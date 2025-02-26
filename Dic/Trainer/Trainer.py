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

class SequenceTrainer(BaseTrainer):
    def __init__(self, 
        model, 
        teammateencoder, 
        adhocencoder, 
        returnnet, 
        goaldecoder,
        optimizer, batch_size, get_batch, loss_fn, 
        alpha, 
        beta, 
        gama, 
        sigma,
        scheduler=None, eval_fns=None):

        self.model = model
        self.teammateencoder = teammateencoder
        self.adhocencoder = adhocencoder 
        self.returnnet = returnnet
        self.goaldecoder = goaldecoder

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.sigma = sigma

        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
    
    def MIE_loss(self, teammateencoder, adhocencoder, beta, gama, s, o):
        # 第一个 KL 项：D_KL(t(z | s) || N(0, I))

        mu1, log_var1 = teammateencoder(s)
        log_var1 = torch.clamp(log_var1, min=-10, max=10)
        std1 = torch.exp(0.5 * log_var1) + 1e-6

        q_z1 = torch.distributions.Normal(mu1, std1)
        p_z1 = torch.distributions.Normal(torch.zeros_like(mu1), torch.ones_like(std1))  # N(0, I)

        kl_term1 = torch.distributions.kl_divergence(q_z1, p_z1).mean()

        # 第二个 KL 项：D_KL(t(z | s) || ad(h | o))，但不计算 t 的梯度
        with torch.no_grad():
            mu_no_grad, log_var_no_grad = teammateencoder(s)
        std_no_grad = torch.exp(0.5 * log_var_no_grad)
        q_z_no_grad = torch.distributions.Normal(mu_no_grad, std_no_grad)

        mu2, log_var2 = adhocencoder(o)
        log_var2 = torch.clamp(log_var2, min=-10, max=10)
        std2 = torch.exp(0.5 * log_var2) + 1e-6
        p_z2 = torch.distributions.Normal(mu2, std2)
        kl_term2 = torch.distributions.kl_divergence(q_z_no_grad, p_z2).mean()

        # 计算最终损失
        loss = beta * kl_term1 + gama * kl_term2
        return loss, q_z1

    def train(self, episodes_data, train_steps, device, max_ep_len, max_len, goal_steps):
        self.model.train()
        self.teammateencoder.train() 
        self.adhocencoder.train()
        self.returnnet.train()
        self.goaldecoder.train()

        action_loss, total_goal_loss, mie_loss, mse_loss_r, mse_loss_g = self.train_step(episodes_data, device, max_ep_len, max_len, goal_steps)
        if self.scheduler is not None:
            self.scheduler.step()
        loss_dict = {
            "action_loss": action_loss,
            "total_goal_loss": total_goal_loss,
            "mie_loss": mie_loss,
            "mse_loss_r": mse_loss_r,
            "mse_loss_g": mse_loss_g
        }
        return loss_dict

    def compute_loss(self, s, o, r_true, g_true):
        # 计算各损失
        mie_loss, q_z = self.MIE_loss(self.teammateencoder, self.adhocencoder, self.beta, self.gama, s, o)
        z = q_z.sample()
        r_pred = self.returnnet(z)
        r_true = r_true.unsqueeze(1)

        mse_loss_r = F.mse_loss(r_pred, r_true)
        g_pred = self.goaldecoder(z, r_pred)
        
        bce_loss_func = nn.BCELoss(reduction='mean')
        bce_loss_g = bce_loss_func(g_pred, g_true)
        # 总损失，加权组合
        total_loss = mie_loss + self.alpha * mse_loss_r +  bce_loss_g
        return total_loss, mie_loss, self.alpha * mse_loss_r, bce_loss_g

    def train_step(self, episodes_data, device, max_ep_len, K, goal_steps):
        # goal的loss
        rand_t = random.randint(0, episodes_data["state"].size(1) - K - 1 - goal_steps +1)
        sum_total_goal_loss = 0.0
        sum_mie_loss = 0.0
        sum_mse_loss_r = 0.0
        sum_mse_loss_g = 0.0
        
        g_list = []
        for ts in range(rand_t, rand_t + K):
            i = ts - rand_t
            # 提取所有批次样本的该时间步的数据    
            s = episodes_data["state"][:, ts, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
            o = episodes_data["obs"][:, ts, :].to(device)          # shape [batch, dim]
            rtg = episodes_data["rtg"][:, ts].to(device)       # shape [batch, 1]
            g = episodes_data["next_state"][:, ts + goal_steps - 1, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]

            # 执行训练步骤并计算损失
            total_loss, mie_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, rtg, g)
            sum_total_goal_loss += total_loss
            sum_mie_loss += mie_loss
            sum_mse_loss_r += mse_loss_r
            sum_mse_loss_g += mse_loss_g
        # 得到第一部分loss
        sum_total_goal_loss /= K
        sum_mie_loss /= K
        sum_mse_loss_r /= K
        sum_mse_loss_g /= K
        # 第二部分loss
        action_loss = 0.0
        states, obs, actions, goal, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=K, goal_steps=goal_steps)
        batch_size, seq_len = states.shape[0], states.shape[1]
        g_list = []
        for t in range(seq_len):
            state_t = states[:, t, ...]
            mu1, log_var1 = self.teammateencoder(state_t.permute(1, 0, 2))
            log_var1 = torch.clamp(log_var1, min=-10, max=10)
            std1 = torch.exp(0.5 * log_var1) + 1e-6
            q_z1 = torch.distributions.Normal(mu1, std1)
            z = q_z1.sample()
            r_pred = self.returnnet(z)
            g_pred = self.goaldecoder(z, r_pred)

            g_list.append(g_pred.reshape(g_pred.shape[1], g_pred.shape[0]*g_pred.shape[2]).unsqueeze(1))

        goal = torch.stack(g_list, dim=1) # (batch_size, K, 2 * state_dim)
        actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            obs, actions, goal, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss /= K
        self.optimizer.zero_grad()
        loss += self.sigma * sum_total_goal_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), sum_total_goal_loss.detach().cpu().item(), mie_loss.detach().cpu().item(), mse_loss_r.detach().cpu().item(), mse_loss_g.detach().cpu().item()

    def evaluate(self, val_loader, device, max_ep_len, max_len, goal_steps):
        self.model.eval()
        self.teammateencoder.eval() 
        self.adhocencoder.eval()
        self.returnnet.eval()
        self.goaldecoder.eval()
        action_loss = 0.0
        sum_total_goal_loss = 0.0
        sum_mie_loss = 0.0
        sum_mse_loss_r = 0.0
        sum_mse_loss_g = 0.0
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
            loss, total_goal_loss, mie_loss, mse_loss_r, mse_loss_g = self.eval_step(episodes_data, device, max_ep_len, max_len, goal_steps)
            action_loss += loss
            sum_total_goal_loss += total_goal_loss
            sum_mie_loss += mie_loss
            sum_mse_loss_r += mse_loss_r
            sum_mse_loss_g += mse_loss_g

        loss_dict = {
            "action_loss": action_loss,
            "total_goal_loss": sum_total_goal_loss,
            "mie_loss": sum_mie_loss,
            "mse_loss_r": sum_mse_loss_r,
            "mse_loss_g": sum_mse_loss_g
        }
        return loss_dict

    def eval_step(self, episodes_data, device, max_ep_len, K, goal_steps):
        with torch.no_grad():
            # goal的loss
            rand_t = random.randint(0, episodes_data["state"].size(1) - K - 1 - goal_steps +1)

            sum_total_goal_loss = 0.0
            sum_mie_loss = 0.0
            sum_mse_loss_r = 0.0
            sum_mse_loss_g = 0.0
            states, obs, actions, goal, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=K, goal_steps=goal_steps)
            g_list = []
            for ts in range(rand_t, rand_t + K):
                i = ts - rand_t
                # 提取所有批次样本的该时间步的数据    
                s = episodes_data["state"][:, ts, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
                o = episodes_data["obs"][:, ts, :].to(device)          # shape [batch, dim]
                rtg = episodes_data["rtg"][:, ts].to(device)       # shape [batch, 1]
                g = episodes_data["next_state"][:, ts + goal_steps - 1, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]

                # 执行训练步骤并计算损失
                total_loss, mie_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, rtg, g)
                sum_total_goal_loss += total_loss
                sum_mie_loss += mie_loss
                sum_mse_loss_r += mse_loss_r
                sum_mse_loss_g += mse_loss_g

            # 得到第一部分loss
            sum_total_goal_loss /= K
            sum_mie_loss /= K
            sum_mse_loss_r /= K
            sum_mse_loss_g /= K

        states, obs, actions, goal, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=K, goal_steps=goal_steps)
        batch_size, seq_len = states.shape[0], states.shape[1]
        g_list = []
        for t in range(seq_len):
            state_t = states[:, t, ...]
            mu1, log_var1 = self.teammateencoder(state_t.permute(1, 0, 2))
            log_var1 = torch.clamp(log_var1, min=-10, max=10)
            std1 = torch.exp(0.5 * log_var1) + 1e-6
            q_z1 = torch.distributions.Normal(mu1, std1)
            z = q_z1.sample()
            r_pred = self.returnnet(z)
            g_pred = self.goaldecoder(z, r_pred)

            g_list.append(g_pred.reshape(g_pred.shape[1], g_pred.shape[0]*g_pred.shape[2]).unsqueeze(1))

        goal = torch.stack(g_list, dim=1) # (batch_size, K, 2 * state_dim)
        
        actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
        action_target = torch.clone(actions)
        # 使用 no_grad() 禁用梯度计算
        with torch.no_grad():
            state_preds, action_preds, reward_preds = self.model.forward(
                obs, actions, goal, timesteps, attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss /= K
        loss += self.sigma * sum_total_goal_loss
        return loss.detach().cpu().item(), sum_total_goal_loss.detach().cpu().item(), mie_loss.detach().cpu().item(), mse_loss_r.detach().cpu().item(), mse_loss_g.detach().cpu().item()



class SequenceTrainer_lbf(BaseTrainer):
    def __init__(self, 
        model, 
        teammateencoder, 
        adhocencoder, 
        returnnet, 
        goaldecoder,
        optimizer, batch_size, get_batch, loss_fn, 
        alpha, 
        beta, 
        gama, 
        sigma,
        scheduler=None, eval_fns=None):

        self.model = model
        self.teammateencoder = teammateencoder
        self.adhocencoder = adhocencoder 
        self.returnnet = returnnet
        self.goaldecoder = goaldecoder

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.sigma = sigma

        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
    
    def MIE_loss(self, teammateencoder, adhocencoder, beta, gama, s, o):
        # 第一个 KL 项：D_KL(t(z | s) || N(0, I))

        mu1, log_var1 = teammateencoder(s)
        log_var1 = torch.clamp(log_var1, min=-10, max=10)
        std1 = torch.exp(0.5 * log_var1) + 1e-6

        q_z1 = torch.distributions.Normal(mu1, std1)
        p_z1 = torch.distributions.Normal(torch.zeros_like(mu1), torch.ones_like(std1))  # N(0, I)

        kl_term1 = torch.distributions.kl_divergence(q_z1, p_z1).mean()

        # 第二个 KL 项：D_KL(t(z | s) || ad(h | o))，但不计算 t 的梯度
        with torch.no_grad():
            mu_no_grad, log_var_no_grad = teammateencoder(s)
        std_no_grad = torch.exp(0.5 * log_var_no_grad)
        q_z_no_grad = torch.distributions.Normal(mu_no_grad, std_no_grad)

        mu2, log_var2 = adhocencoder(o)
        log_var2 = torch.clamp(log_var2, min=-10, max=10)
        std2 = torch.exp(0.5 * log_var2) + 1e-6
        p_z2 = torch.distributions.Normal(mu2, std2)
        kl_term2 = torch.distributions.kl_divergence(q_z_no_grad, p_z2).mean()

        # 计算最终损失
        loss = beta * kl_term1 + gama * kl_term2
        return loss, q_z1

    def train(self, episodes_data, train_steps, device, max_ep_len, max_len, goal_steps):
        self.model.train()
        self.teammateencoder.train() 
        self.adhocencoder.train()
        self.returnnet.train()
        self.goaldecoder.train()

        action_loss, total_goal_loss, mie_loss, mse_loss_r, mse_loss_g = self.train_step(episodes_data, device, max_ep_len, max_len, goal_steps)
        if self.scheduler is not None:
            self.scheduler.step()
        loss_dict = {
            "action_loss": action_loss,
            "total_goal_loss": total_goal_loss,
            "mie_loss": mie_loss,
            "mse_loss_r": mse_loss_r,
            "mse_loss_g": mse_loss_g
        }
        return loss_dict

    def compute_loss(self, s, o, r_true, g_true):
        # 计算各损失
        mie_loss, q_z = self.MIE_loss(self.teammateencoder, self.adhocencoder, self.beta, self.gama, s, o)
        z = q_z.sample()
        r_pred = self.returnnet(z)
        r_true = r_true.unsqueeze(1)

        mse_loss_r = F.mse_loss(r_pred, r_true)
        g_pred = self.goaldecoder(z, r_pred)
        
        bce_loss_func = nn.BCELoss(reduction='mean')
        bce_loss_g = bce_loss_func(g_pred, g_true)
        # 总损失，加权组合
        total_loss = mie_loss + self.alpha * mse_loss_r +  bce_loss_g
        return total_loss, mie_loss, self.alpha * mse_loss_r, bce_loss_g

    def train_step(self, episodes_data, device, max_ep_len, K, goal_steps):
        # goal的loss
        rand_t = random.randint(0, episodes_data["state"].size(1) - K - 1 - goal_steps +1)
        sum_total_goal_loss = 0.0
        sum_mie_loss = 0.0
        sum_mse_loss_r = 0.0
        sum_mse_loss_g = 0.0
        for ts in range(rand_t, rand_t + K):
            # 提取所有批次样本的该时间步的数据    
            states = episodes_data["state"][:, ts, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
            obs = episodes_data["obs"][:, ts, :].to(device)          # shape [batch, dim]
            rtg = episodes_data["rtg"][:, ts].to(device)       # shape [batch, 1]
            goal = episodes_data["next_state"][:, ts + goal_steps - 1, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
            # 手工提取状态中的坐标部分
            states = states[..., [-3, -2]]
            obs = obs[..., [-3, -2]]
            goal = goal[..., [-3, -2]]
            # 转成one-hot
            num_classes = 20

            states = F.one_hot(states.to(torch.long), num_classes).to(torch.float32)
            states = states.view(states.shape[0], states.shape[1], states.shape[2]*states.shape[3])
            obs = F.one_hot(obs.to(torch.long), num_classes).to(torch.float32)
            obs = obs.view(obs.shape[0], obs.shape[1]*obs.shape[2])
            goal = F.one_hot(goal.to(torch.long), num_classes).to(torch.float32)
            goal = goal.view(goal.shape[0], goal.shape[1], goal.shape[2]*goal.shape[3])

            # 执行训练步骤并计算损失
            total_loss, mie_loss, mse_loss_r, mse_loss_g = self.compute_loss(states, obs, rtg, goal)
            sum_total_goal_loss += total_loss
            sum_mie_loss += mie_loss
            sum_mse_loss_r += mse_loss_r
            sum_mse_loss_g += mse_loss_g
        # 得到第一部分loss
        sum_total_goal_loss /= K
        sum_mie_loss /= K
        sum_mse_loss_r /= K
        sum_mse_loss_g /= K
        # 第二部分loss
        action_loss = 0.0
        states, obs, actions, goal, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=K, goal_steps=goal_steps)
        states = states[..., [-3, -2]].to(torch.long)
        states = F.one_hot(states, 20)
        states = states.view(states.shape[0], states.shape[1], states.shape[2], states.shape[3]*states.shape[4]).to(torch.float32)

        obs = obs[..., [-3, -2]].to(torch.long)
        obs = F.one_hot(obs, 20)
        obs = obs.view(obs.shape[0], obs.shape[1], obs.shape[2]*obs.shape[3]).to(torch.float32)
        # goal = goal[..., [-3, -2]]

        batch_size, seq_len = states.shape[0], states.shape[1]
        goals_list = []
        
        for t in range(seq_len):
            state_t = states[:, t, ...]
            mu1, log_var1 = self.teammateencoder(state_t.permute(1, 0, 2))
            log_var1 = torch.clamp(log_var1, min=-10, max=10)
            std1 = torch.exp(0.5 * log_var1) + 1e-6
            q_z1 = torch.distributions.Normal(mu1, std1)
            z = q_z1.sample()
            r_pred = self.returnnet(z)
            g_pred = self.goaldecoder(z, r_pred)

            goals_list.append(g_pred.reshape(g_pred.shape[1], g_pred.shape[0]*g_pred.shape[2]).unsqueeze(1))
            
        
        # 拼接所有时间步的goal
        goal = torch.cat(goals_list, dim=1)


        actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
        action_target = torch.clone(actions)
        state_preds, action_preds, reward_preds = self.model.forward(
            obs, actions, goal, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss /= K
        self.optimizer.zero_grad()
        loss += self.sigma * sum_total_goal_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), sum_total_goal_loss.detach().cpu().item(), mie_loss.detach().cpu().item(), mse_loss_r.detach().cpu().item(), mse_loss_g.detach().cpu().item()

    def evaluate(self, val_loader, device, max_ep_len, max_len, goal_steps):
        self.model.eval()
        self.teammateencoder.eval() 
        self.adhocencoder.eval()
        self.returnnet.eval()
        self.goaldecoder.eval()
        action_loss = 0.0
        sum_total_goal_loss = 0.0
        sum_mie_loss = 0.0
        sum_mse_loss_r = 0.0
        sum_mse_loss_g = 0.0
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
            loss, total_goal_loss, mie_loss, mse_loss_r, mse_loss_g = self.eval_step(episodes_data, device, max_ep_len, max_len, goal_steps)
            action_loss += loss
            sum_total_goal_loss += total_goal_loss
            sum_mie_loss += mie_loss
            sum_mse_loss_r += mse_loss_r
            sum_mse_loss_g += mse_loss_g

        loss_dict = {
            "action_loss": action_loss,
            "total_goal_loss": sum_total_goal_loss,
            "mie_loss": sum_mie_loss,
            "mse_loss_r": sum_mse_loss_r,
            "mse_loss_g": sum_mse_loss_g
        }
        return loss_dict

    def eval_step(self, episodes_data, device, max_ep_len, K, goal_steps):
        with torch.no_grad():
            # goal的loss
            rand_t = random.randint(0, episodes_data["state"].size(1) - K - 1 - goal_steps +1)
            total_goal_loss = 0.0
            mie_loss = 0.0
            mse_loss_r = 0.0
            mse_loss_g = 0.0
            for ts in range(rand_t, rand_t + K):
                # 提取所有批次样本的该时间步的数据    
                states = episodes_data["state"][:, ts, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
                obs = episodes_data["obs"][:, ts, :].to(device)          # shape [batch, dim]
                rtg = episodes_data["rtg"][:, ts].to(device)       # shape [batch, 1]
                goal = episodes_data["next_state"][:, ts + goal_steps - 1, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
                states = states[..., [-3, -2]]
                obs = obs[..., [-3, -2]]
                goal = goal[..., [-3, -2]]
                # 转one-hot
                num_classes = 20
                states = F.one_hot(states.to(torch.long), num_classes).to(torch.float32)
                states = states.view(states.shape[0], states.shape[1], states.shape[2]*states.shape[3])
                obs = F.one_hot(obs.to(torch.long), num_classes).to(torch.float32)
                obs = obs.view(obs.shape[0], obs.shape[1]*obs.shape[2])
                goal = F.one_hot(goal.to(torch.long), num_classes).to(torch.float32)
                goal = goal.view(goal.shape[0], goal.shape[1], goal.shape[2]*goal.shape[3])

                # 执行训练步骤并计算损失
                total_loss, mie_loss, mse_loss_r, mse_loss_g = self.compute_loss(states, obs, rtg, goal)
                total_goal_loss += total_loss
                mie_loss += mie_loss
                mse_loss_r += mse_loss_r
                mse_loss_g += mse_loss_g
            # 得到第一部分loss
            total_goal_loss /= K
            mie_loss /= K
            mse_loss_r /= K
            mse_loss_g /= K

        states, obs, actions, goal, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=K, goal_steps=goal_steps)
        states = states[..., [-3, -2]].to(torch.long)
        states = F.one_hot(states, 20)
        states = states.view(states.shape[0], states.shape[1], states.shape[2], states.shape[3]*states.shape[4]).to(torch.float32)

        obs = obs[..., [-3, -2]].to(torch.long)
        obs = F.one_hot(obs, 20)
        obs = obs.view(obs.shape[0], obs.shape[1], obs.shape[2]*obs.shape[3]).to(torch.float32)
        # goal = goal[..., [-3, -2]]

        batch_size, seq_len = states.shape[0], states.shape[1]
        goals_list = []
        
        for t in range(seq_len):
            state_t = states[:, t, ...]
            mu1, log_var1 = self.teammateencoder(state_t.permute(1, 0, 2))
            log_var1 = torch.clamp(log_var1, min=-10, max=10)
            std1 = torch.exp(0.5 * log_var1) + 1e-6
            q_z1 = torch.distributions.Normal(mu1, std1)
            z = q_z1.sample()
            r_pred = self.returnnet(z)
            g_pred = self.goaldecoder(z, r_pred)

            goals_list.append(g_pred.reshape(g_pred.shape[1], g_pred.shape[0]*g_pred.shape[2]).unsqueeze(1))
            
        
        # 拼接所有时间步的goal
        goal = torch.cat(goals_list, dim=1)
        
        actions = torch.clone(F.one_hot(actions.to(torch.int64), num_classes=self.model.act_dim))
        action_target = torch.clone(actions)
        # 使用 no_grad() 禁用梯度计算
        with torch.no_grad():
            state_preds, action_preds, reward_preds = self.model.forward(
                obs, actions, goal, timesteps, attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss /= K
        loss += self.sigma * total_goal_loss
        return loss.detach().cpu().item(), total_goal_loss.detach().cpu().item(), mie_loss.detach().cpu().item(), mse_loss_r.detach().cpu().item(), mse_loss_g.detach().cpu().item()


class GoalTrainer:
    def __init__(self, teammateencoder, adhocencoder, returnnet, goaldecoder, optimizer, scheduler, alpha=0.1, beta=0.1, gama=0.1,clip_value=1.0):
        # 初始化各网络
        self.teammateencoder = teammateencoder
        self.adhocencoder = adhocencoder
        self.returnnet = returnnet
        self.goaldecoder = goaldecoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        # 设置损失权重
        self.alpha = alpha  # r的MSE损失权重
        self.beta = beta    
        self.gama = gama
        self.clip_value = clip_value  # 梯度裁剪的最大范数


    def kl_divergence(p, q):
        return F.kl_div(p.log(), q, reduction='batchmean')

    def MIE_loss(self, teammateencoder, adhocencoder, beta, gama, s, o):
        # 第一个 KL 项：D_KL(t(z | s) || N(0, I))

        mu1, log_var1 = teammateencoder(s)
        log_var1 = torch.clamp(log_var1, min=-10, max=10)
        std1 = torch.exp(0.5 * log_var1) + 1e-6

        q_z1 = torch.distributions.Normal(mu1, std1)
        p_z1 = torch.distributions.Normal(torch.zeros_like(mu1), torch.ones_like(std1))  # N(0, I)

        kl_term1 = torch.distributions.kl_divergence(q_z1, p_z1).mean()

        # 第二个 KL 项：D_KL(t(z | s) || ad(h | o))，但不计算 t 的梯度
        with torch.no_grad():
            mu_no_grad, log_var_no_grad = teammateencoder(s)
        std_no_grad = torch.exp(0.5 * log_var_no_grad)
        q_z_no_grad = torch.distributions.Normal(mu_no_grad, std_no_grad)

        mu2, log_var2 = adhocencoder(o)
        log_var2 = torch.clamp(log_var2, min=-10, max=10)
        std2 = torch.exp(0.5 * log_var2) + 1e-6
        p_z2 = torch.distributions.Normal(mu2, std2)
        kl_term2 = torch.distributions.kl_divergence(q_z_no_grad, p_z2).mean()

        # 计算最终损失
        loss = beta * kl_term1 + gama * kl_term2
        return loss, q_z1
    

    def compute_loss(self, s, o, r_true, g_true):
        # 计算各损失
        mie_loss, q_z = self.MIE_loss(self.teammateencoder, self.adhocencoder, self.beta, self.gama, s, o)
        z = q_z.sample()
        r_pred = self.returnnet(z)
        r_true = r_true.unsqueeze(1)

        mse_loss_r = F.mse_loss(r_pred, r_true)
        g_pred = self.goaldecoder(z, r_pred)
        
        bce_loss_func = nn.BCELoss(reduction='mean')

        bce_loss_g = bce_loss_func(g_pred, g_true)

        # 总损失，加权组合
        total_loss = mie_loss + self.alpha * mse_loss_r +  bce_loss_g
        return total_loss, mie_loss, self.alpha * mse_loss_r, bce_loss_g
    
    def train_step(self, s, o, r_true, g_true):
        # 清除梯度
        self.optimizer.zero_grad()

        # 计算损失
        total_loss, mie_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, r_true, g_true)
        
        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.teammateencoder.parameters()) + 
            list(self.adhocencoder.parameters()) + 
            list(self.returnnet.parameters()) + 
            list(self.goaldecoder.parameters()), 
            max_norm=self.clip_value
        )

        # 优化步骤
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "mie_loss": mie_loss.item(),
            "mse_loss_r": mse_loss_r.item(),
            "mse_loss_g": mse_loss_g.item()
        }

    def train(self, episodes_data, K, device, goal_steps):
        # 切换到训练模式
        self.teammateencoder.train()
        self.adhocencoder.train()
        self.returnnet.train()
        self.goaldecoder.train()
        # 随机选择一个时间步
        rand_t = random.randint(0, episodes_data["state"].size(1) - K - 1 - goal_steps +1)
        total_goal_loss = 0.0
        mie_loss = 0.0
        mse_loss_r = 0.0
        mse_loss_g = 0.0
        for ts in range(rand_t, rand_t + K):
            # 提取所有批次样本的该时间步的数据
            states = episodes_data["state"][:, ts, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
            obs = episodes_data["obs"][:, ts, :].to(device)          # shape [batch, dim]
            rtg = episodes_data["rtg"][:, ts].to(device)       # shape [batch, 1]
            goal = episodes_data["next_state"][:, ts + goal_steps - 1, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]

            # 执行训练步骤并计算损失
            loss_dict = self.train_step(states, obs, rtg, goal)
            total_goal_loss += loss_dict["total_loss"]
            mie_loss += loss_dict["mie_loss"]
            mse_loss_r += loss_dict["mse_loss_r"]
            mse_loss_g += loss_dict["mse_loss_g"]
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "total_loss": total_goal_loss / K,
            "mie_loss": mie_loss / K,
            "mse_loss_r": mse_loss_r / K,
            "mse_loss_g": mse_loss_g / K
        }

    def eval_step(self, s, o, r_true, g_true):
        # 计算损失
        # 使用 no_grad() 禁用梯度计算
        with torch.no_grad():
            total_loss, mie_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, r_true, g_true)

        # 直接返回损失项，无需反向传播和优化
        return {
            "total_loss": total_loss.item(),
            "mie_loss": mie_loss.item(),
            "mse_loss_r": mse_loss_r.item(),
            "mse_loss_g": mse_loss_g.item()
        }

    def evaluate(self, val_loader, device, goal_steps):
        # 切换到评估模式
        self.teammateencoder.eval()
        self.adhocencoder.eval()
        self.returnnet.eval()
        self.goaldecoder.eval()
        
        total_loss = 0.0
        mie_loss = 0.0
        mse_loss_r = 0.0
        mse_loss_g = 0.0
        num_batches = len(val_loader)
        game_length = 1
        with torch.no_grad():
            for batch_idx, episodes_data in enumerate(val_loader):
                game_length = episodes_data["state"].size(1)
                states = episodes_data["state"]
                obs = episodes_data["obs"]
                rtg = episodes_data["rtg"]
                goal = episodes_data["next_state"]
                
                for ts in range(states.size(1) - goal_steps + 1):
                    s = states[:, ts, :, :].permute(1, 0, 2).to(device)   # shape [batch, num, dim]
                    o = obs[:, ts, :].to(device)   # shape [batch, dim]
                    r_true = rtg[:, ts].to(device)   # shape [batch, 1]
                    g_true = goal[:, ts + goal_steps - 1, :, :].permute(1, 0, 2).to(device)  # shape [batch, num, dim]
                    
                    loss_dict = self.eval_step(s, o, r_true, g_true)
                    total_loss += loss_dict["total_loss"]
                    mie_loss += loss_dict["mie_loss"]
                    mse_loss_r += loss_dict["mse_loss_r"]
                    mse_loss_g += loss_dict["mse_loss_g"]
        
        # 计算验证集上的平均损失
        avg_total_loss = total_loss / (game_length - goal_steps + 1)
        avg_mie_loss = mie_loss / (game_length - goal_steps + 1)
        avg_mse_loss_r = mse_loss_r / (game_length - goal_steps + 1)
        avg_mse_loss_g = mse_loss_g / (game_length - goal_steps + 1)
        
        return {
            "total_loss": avg_total_loss,
            "mie_loss": avg_mie_loss,
            "mse_loss_r": avg_mse_loss_r,
            "mse_loss_g": avg_mse_loss_g
        }

class GoalTrainer_lbf:
    def __init__(self, teammateencoder, adhocencoder, returnnet, goaldecoder, optimizer, scheduler, alpha=0.1, beta=0.1, gama=0.1,clip_value=1.0):
        # 初始化各网络
        self.teammateencoder = teammateencoder
        self.adhocencoder = adhocencoder
        self.returnnet = returnnet
        self.goaldecoder = goaldecoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        # 设置损失权重
        self.alpha = alpha  # r的MSE损失权重
        self.beta = beta    
        self.gama = gama
        self.clip_value = clip_value  # 梯度裁剪的最大范数


    def kl_divergence(p, q):
        return F.kl_div(p.log(), q, reduction='batchmean')

    def MIE_loss(self, teammateencoder, adhocencoder, beta, gama, s, o):
        # 第一个 KL 项：D_KL(t(z | s) || N(0, I))

        mu1, log_var1 = teammateencoder(s)
        log_var1 = torch.clamp(log_var1, min=-10, max=10)
        std1 = torch.exp(0.5 * log_var1) + 1e-6

        q_z1 = torch.distributions.Normal(mu1, std1)
        p_z1 = torch.distributions.Normal(torch.zeros_like(mu1), torch.ones_like(std1))  # N(0, I)

        kl_term1 = torch.distributions.kl_divergence(q_z1, p_z1).mean()

        # 第二个 KL 项：D_KL(t(z | s) || ad(h | o))，但不计算 t 的梯度
        with torch.no_grad():
            mu_no_grad, log_var_no_grad = teammateencoder(s)
        std_no_grad = torch.exp(0.5 * log_var_no_grad)
        q_z_no_grad = torch.distributions.Normal(mu_no_grad, std_no_grad)

        mu2, log_var2 = adhocencoder(o)
        log_var2 = torch.clamp(log_var2, min=-10, max=10)
        std2 = torch.exp(0.5 * log_var2) + 1e-6
        p_z2 = torch.distributions.Normal(mu2, std2)
        kl_term2 = torch.distributions.kl_divergence(q_z_no_grad, p_z2).mean()

        # 计算最终损失
        loss = beta * kl_term1 + gama * kl_term2
        return loss, q_z_no_grad
    

    def compute_loss(self, s, o, r_true, g_true):
        # 计算各损失
        mie_loss, q_z = self.MIE_loss(self.teammateencoder, self.adhocencoder, self.beta, self.gama, s, o)
        z = q_z.sample()
        r_pred = self.returnnet(z)
        r_true = r_true.unsqueeze(1)

        mse_loss_r = F.mse_loss(r_pred, r_true)
        g_pred = self.goaldecoder(z, r_pred)
        
        mse_loss_g = F.mse_loss(g_pred, g_true)

        # 总损失，加权组合
        total_loss = mie_loss + self.alpha * mse_loss_r +  mse_loss_g 
        return total_loss, mie_loss, self.alpha * mse_loss_r, mse_loss_g 
    
    def train_step(self, s, o, r_true, g_true):
        # 清除梯度
        self.optimizer.zero_grad()

        # 计算损失
        total_loss, mie_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, r_true, g_true)
        
        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.teammateencoder.parameters()) + 
            list(self.adhocencoder.parameters()) + 
            list(self.returnnet.parameters()) + 
            list(self.goaldecoder.parameters()), 
            max_norm=self.clip_value
        )

        # 优化步骤
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "mie_loss": mie_loss.item(),
            "mse_loss_r": mse_loss_r.item(),
            "mse_loss_g": mse_loss_g.item()
        }

    def train(self, episodes_data, K, device, goal_steps):
        # 切换到训练模式
        self.teammateencoder.train()
        self.adhocencoder.train()
        self.returnnet.train()
        self.goaldecoder.train()
        # 随机选择一个时间步
        rand_t = random.randint(0, episodes_data["state"].size(1) - K - 1 - goal_steps +1)
        total_goal_loss = 0.0
        mie_loss = 0.0
        mse_loss_r = 0.0
        mse_loss_g = 0.0
        for ts in range(rand_t, rand_t + K):
            # 提取所有批次样本的该时间步的数据
            
            states = episodes_data["state"][:, ts, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
            obs = episodes_data["obs"][:, ts, :].to(device)          # shape [batch, dim]
            rtg = episodes_data["rtg"][:, ts].to(device)       # shape [batch, 1]
            goal = episodes_data["next_state"][:, ts + goal_steps - 1, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
            states = states[..., [-3, -2]]
            obs = obs[..., [-3, -2]]
            goal = goal[..., [-3, -2]]
            # 执行训练步骤并计算损失
            loss_dict = self.train_step(states, obs, rtg, goal)
            total_goal_loss += loss_dict["total_loss"]
            mie_loss += loss_dict["mie_loss"]
            mse_loss_r += loss_dict["mse_loss_r"]
            mse_loss_g += loss_dict["mse_loss_g"]
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "total_loss": total_goal_loss / K,
            "mie_loss": mie_loss / K,
            "mse_loss_r": mse_loss_r / K,
            "mse_loss_g": mse_loss_g / K
        }

    def eval_step(self, s, o, r_true, g_true):
        # 计算损失
        # 使用 no_grad() 禁用梯度计算
        with torch.no_grad():
            total_loss, mie_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, r_true, g_true)

        # 直接返回损失项，无需反向传播和优化
        return {
            "total_loss": total_loss.item(),
            "mie_loss": mie_loss.item(),
            "mse_loss_r": mse_loss_r.item(),
            "mse_loss_g": mse_loss_g.item()
        }

    def evaluate(self, val_loader, device, goal_steps):
        # 切换到评估模式
        self.teammateencoder.eval()
        self.adhocencoder.eval()
        self.returnnet.eval()
        self.goaldecoder.eval()
        
        total_loss = 0.0
        mie_loss = 0.0
        mse_loss_r = 0.0
        mse_loss_g = 0.0
        num_batches = len(val_loader)
        game_length = 1
        with torch.no_grad():
            for batch_idx, episodes_data in enumerate(val_loader):
                game_length = episodes_data["state"].size(1)
                states = episodes_data["state"]
                obs = episodes_data["obs"]
                rtg = episodes_data["rtg"]
                goal = episodes_data["next_state"]
                
                for ts in range(states.size(1) - goal_steps + 1):
                    s = states[:, ts, :, :].permute(1, 0, 2).to(device)   # shape [batch, num, dim]
                    o = obs[:, ts, :].to(device)   # shape [batch, dim]
                    r_true = rtg[:, ts].to(device)   # shape [batch, 1]
                    g_true = goal[:, ts + goal_steps - 1, :, :].permute(1, 0, 2).to(device)  # shape [batch, num, dim]
                    s = s[..., [-3, -2]]
                    o = o[..., [-3, -2]]
                    g_true = g_true[..., [-3, -2]]

                    loss_dict = self.eval_step(s, o, r_true, g_true)
                    total_loss += loss_dict["total_loss"]
                    mie_loss += loss_dict["mie_loss"]
                    mse_loss_r += loss_dict["mse_loss_r"]
                    mse_loss_g += loss_dict["mse_loss_g"]
        
        # 计算验证集上的平均损失
        avg_total_loss = total_loss / (game_length - goal_steps + 1)
        avg_mie_loss = mie_loss / (game_length - goal_steps + 1)
        avg_mse_loss_r = mse_loss_r / (game_length - goal_steps + 1)
        avg_mse_loss_g = mse_loss_g / (game_length - goal_steps + 1)
        
        return {
            "total_loss": avg_total_loss,
            "mie_loss": avg_mie_loss,
            "mse_loss_r": avg_mse_loss_r,
            "mse_loss_g": avg_mse_loss_g
        }


class DtTrainer(BaseTrainer):
    def train(self, episodes_data, device, max_ep_len, max_len):
        self.model.train()
        action_loss = 0.0
        loss = self.train_step(episodes_data, device, max_ep_len, max_len)
        action_loss += loss
        if self.scheduler is not None:
            self.scheduler.step()

        return action_loss

    def train_step(self, episodes_data, device, max_ep_len, max_len):
        
        states, actions, rtg, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len)
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
        
        states, actions, rtg, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len)
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


class DtTrainer_lbf(BaseTrainer):
    def train(self, episodes_data, device, max_ep_len, max_len):
        self.model.train()
        action_loss = 0.0
        loss = self.train_step(episodes_data, device, max_ep_len, max_len)
        action_loss += loss
        if self.scheduler is not None:
            self.scheduler.step()

        return action_loss

    def train_step(self, episodes_data, device, max_ep_len, max_len):
        
        states, actions, rtg, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len)
        states = states[..., [-3, -2]]
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
        
        states, actions, rtg, dones, timesteps, attention_mask = self.get_batch(episodes_data, device=device, max_ep_len=max_ep_len, max_len=max_len)
        states = states[..., [-3, -2]]
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



class GoalTrainer_pp4a_wo_MIE:
    def __init__(self, adhocencoder, returnnet, goaldecoder, optimizer, scheduler, alpha=0.1, beta=0.1, gama=0.1,clip_value=1.0):
        # 初始化各网络
        self.adhocencoder = adhocencoder
        self.returnnet = returnnet
        self.goaldecoder = goaldecoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        # 设置损失权重
        self.alpha = alpha  # r的MSE损失权重
        self.beta = beta    
        self.gama = gama
        self.clip_value = clip_value  # 梯度裁剪的最大范数


    def kl_divergence(p, q):
        return F.kl_div(p.log(), q, reduction='batchmean')

    def compute_z(self, adhocencoder, o):

        mu2, log_var2 = adhocencoder(o)
        log_var2 = torch.clamp(log_var2, min=-10, max=10)
        std2 = torch.exp(0.5 * log_var2) + 1e-6
        p_z2 = torch.distributions.Normal(mu2, std2)
        return p_z2 
    

    def compute_loss(self, s, o, r_true, g_true):
        # 计算各损失
        p_z = self.compute_z(self.adhocencoder, o)
        z = p_z.sample()
        r_pred = self.returnnet(z)
        r_true = r_true.unsqueeze(1)

        mse_loss_r = F.mse_loss(r_pred, r_true)
        g_pred = self.goaldecoder(z, r_pred)
        
        bce_loss_func = nn.BCELoss(reduction='mean')

        bce_loss_g = bce_loss_func(g_pred, g_true)

        # 总损失，加权组合
        total_loss = self.alpha * mse_loss_r +  bce_loss_g
        return total_loss, self.alpha * mse_loss_r, bce_loss_g
    
    def train_step(self, s, o, r_true, g_true):
        # 清除梯度
        self.optimizer.zero_grad()

        # 计算损失
        total_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, r_true, g_true)
        
        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(self.adhocencoder.parameters()) + 
            list(self.returnnet.parameters()) + 
            list(self.goaldecoder.parameters()), 
            max_norm=self.clip_value
        )

        # 优化步骤
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "mse_loss_r": mse_loss_r.item(),
            "mse_loss_g": mse_loss_g.item()
        }

    def train(self, episodes_data, K, device, goal_steps):
        # 切换到训练模式
        self.adhocencoder.train()
        self.returnnet.train()
        self.goaldecoder.train()
        # 随机选择一个时间步
        rand_t = random.randint(0, episodes_data["state"].size(1) - K - 1 - goal_steps +1)
        total_goal_loss = 0.0
        mie_loss = 0.0
        mse_loss_r = 0.0
        mse_loss_g = 0.0
        for ts in range(rand_t, rand_t + K):
            # 提取所有批次样本的该时间步的数据
            states = episodes_data["state"][:, ts, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]
            obs = episodes_data["obs"][:, ts, :].to(device)          # shape [batch, dim]
            rtg = episodes_data["rtg"][:, ts].to(device)       # shape [batch, 1]
            goal = episodes_data["next_state"][:, ts + goal_steps - 1, :, :].to(device).permute(1, 0, 2)  # shape [batch, num, dim]

            # 执行训练步骤并计算损失
            loss_dict = self.train_step(states, obs, rtg, goal)
            total_goal_loss += loss_dict["total_loss"]
            mse_loss_r += loss_dict["mse_loss_r"]
            mse_loss_g += loss_dict["mse_loss_g"]
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "total_loss": total_goal_loss / K,
            "mse_loss_r": mse_loss_r / K,
            "mse_loss_g": mse_loss_g / K
        }

    def eval_step(self, s, o, r_true, g_true):
        # 计算损失
        # 使用 no_grad() 禁用梯度计算
        with torch.no_grad():
            total_loss, mse_loss_r, mse_loss_g = self.compute_loss(s, o, r_true, g_true)

        # 直接返回损失项，无需反向传播和优化
        return {
            "total_loss": total_loss.item(),
            "mse_loss_r": mse_loss_r.item(),
            "mse_loss_g": mse_loss_g.item()
        }

    def evaluate(self, val_loader, device, goal_steps):
        # 切换到评估模式
        self.adhocencoder.eval()
        self.returnnet.eval()
        self.goaldecoder.eval()
        
        total_loss = 0.0
        mie_loss = 0.0
        mse_loss_r = 0.0
        mse_loss_g = 0.0
        num_batches = len(val_loader)
        game_length = 1
        with torch.no_grad():
            for batch_idx, episodes_data in enumerate(val_loader):
                game_length = episodes_data["state"].size(1)
                states = episodes_data["state"]
                obs = episodes_data["obs"]
                rtg = episodes_data["rtg"]
                goal = episodes_data["next_state"]
                
                for ts in range(states.size(1) - goal_steps + 1):
                    s = states[:, ts, :, :].permute(1, 0, 2).to(device)   # shape [batch, num, dim]
                    o = obs[:, ts, :].to(device)   # shape [batch, dim]
                    r_true = rtg[:, ts].to(device)   # shape [batch, 1]
                    g_true = goal[:, ts + goal_steps - 1, :, :].permute(1, 0, 2).to(device)  # shape [batch, num, dim]
                    
                    loss_dict = self.eval_step(s, o, r_true, g_true)
                    total_loss += loss_dict["total_loss"]
                    mse_loss_r += loss_dict["mse_loss_r"]
                    mse_loss_g += loss_dict["mse_loss_g"]
        
        # 计算验证集上的平均损失
        avg_total_loss = total_loss / (game_length - goal_steps + 1)
        avg_mse_loss_r = mse_loss_r / (game_length - goal_steps + 1)
        avg_mse_loss_g = mse_loss_g / (game_length - goal_steps + 1)
        
        return {
            "total_loss": avg_total_loss,
            "mse_loss_r": avg_mse_loss_r,
            "mse_loss_g": avg_mse_loss_g
        }