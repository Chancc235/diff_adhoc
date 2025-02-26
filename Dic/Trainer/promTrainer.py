import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class BaseTrainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        eval_start = time.time()
        """
        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v
        
        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        """

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

class PromTrainer(BaseTrainer):
    def train(self, episodes_data, device, max_ep_len, max_len):
        self.model.train()
        action_loss = 0.0
        loss = self.train_step(episodes_data, device, max_ep_len, max_len)
        action_loss += loss
        if self.scheduler is not None:
            self.scheduler.step()

        return action_loss

    def train_step(self, episodes_data, device, max_ep_len, max_len):
        states_p, actions_p, rtg_p, dones_p, timesteps_p, attention_mask_p = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=int(max_len / 5))
        states, actions, rtg, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len)
        states = torch.cat((states_p, states), dim=1)
        actions = torch.cat((actions_p, actions), dim=1)
        rtg = torch.cat((rtg_p, rtg), dim=1)
        dones = torch.cat((dones_p, dones), dim=1)
        timesteps = torch.cat((timesteps_p, timesteps), dim=1)
        attention_mask = torch.cat((attention_mask_p, attention_mask), dim=1)

        actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
        action_target = torch.clone(actions)
        state_preds, action_preds, rtg_preds = self.model.forward(
            states, actions, rtg, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item() / max_len

    def evaluate(self, val_loader, device, max_ep_len, max_len):
        self.model.eval()
        action_loss = 0.0
        for batch_idx, episodes_data in enumerate(val_loader):
            """
            states, actions, goal, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len, goal_steps=goal_steps)
            actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
            action_target = torch.clone(actions)
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, goal, timesteps, attention_mask=attention_mask,
            )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            """
            loss = self.eval_step(episodes_data, device, max_ep_len, max_len)
            action_loss += loss

        return action_loss

    def eval_step(self, episodes_data, device, max_ep_len, max_len):
        states_p, actions_p, rtg_p, dones_p, timesteps_p, attention_mask_p = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=int(max_len / 5))
        states, actions, rtg, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len)
        states = torch.cat((states_p, states), dim=1)
        actions = torch.cat((actions_p, actions), dim=1)
        rtg = torch.cat((rtg_p, rtg), dim=1)
        timesteps = torch.cat((timesteps_p, timesteps), dim=1)
        attention_mask = torch.cat((attention_mask_p, attention_mask), dim=1)

        actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
        action_target = torch.clone(actions)
        # 使用 no_grad() 禁用梯度计算
        with torch.no_grad():
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rtg, timesteps, attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        return loss.detach().cpu().item() / max_len



class PromTrainer_lbf(BaseTrainer):
    def train(self, episodes_data, device, max_ep_len, max_len):
        self.model.train()
        action_loss = 0.0
        loss = self.train_step(episodes_data, device, max_ep_len, max_len)
        action_loss += loss
        if self.scheduler is not None:
            self.scheduler.step()

        return action_loss

    def train_step(self, episodes_data, device, max_ep_len, max_len):
        states_p, actions_p, rtg_p, dones_p, timesteps_p, attention_mask_p = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=int(max_len / 5))
        states_p = states_p[..., [-3, -2]]
        states, actions, rtg, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len)
        states = states[..., [-3, -2]]
        states = torch.cat((states_p, states), dim=1)
        actions = torch.cat((actions_p, actions), dim=1)
        rtg = torch.cat((rtg_p, rtg), dim=1)
        dones = torch.cat((dones_p, dones), dim=1)
        timesteps = torch.cat((timesteps_p, timesteps), dim=1)
        attention_mask = torch.cat((attention_mask_p, attention_mask), dim=1)

        actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
        action_target = torch.clone(actions)
        state_preds, action_preds, rtg_preds = self.model.forward(
            states, actions, rtg, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item() / max_len

    def evaluate(self, val_loader, device, max_ep_len, max_len):
        self.model.eval()
        action_loss = 0.0
        for batch_idx, episodes_data in enumerate(val_loader):
            """
            states, actions, goal, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len, goal_steps=goal_steps)
            actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
            action_target = torch.clone(actions)
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, goal, timesteps, attention_mask=attention_mask,
            )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            """
            loss = self.eval_step(episodes_data, device, max_ep_len, max_len)
            action_loss += loss

        return action_loss

    def eval_step(self, episodes_data, device, max_ep_len, max_len):
        states_p, actions_p, rtg_p, dones_p, timesteps_p, attention_mask_p = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=int(max_len / 5))
        states_p = states_p[..., [-3, -2]]
        states, actions, rtg, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len)
        states = states[..., [-3, -2]]
        states = torch.cat((states_p, states), dim=1)
        actions = torch.cat((actions_p, actions), dim=1)
        rtg = torch.cat((rtg_p, rtg), dim=1)
        timesteps = torch.cat((timesteps_p, timesteps), dim=1)
        attention_mask = torch.cat((attention_mask_p, attention_mask), dim=1)

        actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
        action_target = torch.clone(actions)
        # 使用 no_grad() 禁用梯度计算
        with torch.no_grad():
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rtg, timesteps, attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        return loss.detach().cpu().item() / max_len
