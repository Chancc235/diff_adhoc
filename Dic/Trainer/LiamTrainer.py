import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class LiamTrainer:
    def __init__(
        self,
        liam_encoder,
        reconstruction_decoder,
        Q_net,
        V_net,
        policy_net,
        optimizer,
        gamma=0.99,
        beta=0.1,
        alpha=0.1,
        sita=0.1,
        batch_size=512,
        act_dim=5,
        update_freq=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        # Models
        self.liam_encoder = liam_encoder.to(device)
        self.reconstruction_decoder = reconstruction_decoder.to(device)
        self.Q_net = Q_net.to(device)
        self.V_net = V_net.to(device)
        self.policy_net = policy_net.to(device)
        
        # Optimizers
        self.optimizer = optimizer
        self.gamma = gamma
        self.beta = beta
        self.batch_size = batch_size
        self.act_dim = act_dim
        self.update_freq = update_freq
        self.alpha = alpha
        self.sita = sita
        
    def train_step(self, episodes_data, max_ep_len):
        
        batch_size = episodes_data["state"].size(0)
        h_0 = torch.zeros(1, batch_size, self.liam_encoder.hidden_dim).to(self.device)
        

        sum_total_loss = 0.0
        sum_reconstruction_loss = 0.0
        sum_a2c_loss = 0.0

        for ts in range(0, max_ep_len-2):
            if ts % self.update_freq == 0:
                h = h_0
                # 提取所有批次样本的该时间步的数据    
                team_states = episodes_data["state"][:, ts].to(self.device) # shape [batch, num, dim]
                obs = episodes_data["obs"][:, ts, :].to(self.device) 
                last_obs = episodes_data["obs"][:, ts-1, :].to(self.device)
                action = torch.clone(F.one_hot(episodes_data["action"][:, ts].to(torch.int64), num_classes=self.act_dim)).to(self.device).unsqueeze(1)  # shape: [batch, 1, dim]
                teammate_actions = torch.clone(F.one_hot(episodes_data["teammate_action"][:, ts].to(torch.int64), num_classes=self.act_dim)).to(self.device)  # shape: [batch, num, dim]
                team_actions = torch.cat([action, teammate_actions], dim=1)  # shape: [batch, num+1, dim]
                reward = episodes_data["reward"][:, ts].to(self.device)

                # 提取t+1时刻的数据
                next_team_states = episodes_data["state"][:, ts+1].to(self.device)  # shape [batch, num, dim]
                next_obs = episodes_data["obs"][:, ts+1, :].to(self.device)
                next_action = torch.clone(F.one_hot(episodes_data["action"][:, ts+1].to(torch.int64), num_classes=self.act_dim)).to(self.device).unsqueeze(1) # shape: [batch, 1, dim]
                next_teammate_actions = torch.clone(F.one_hot(episodes_data["teammate_action"][:, ts+1].to(torch.int64), num_classes=self.act_dim)).to(self.device)  # shape: [batch, num, dim]
                next_team_actions = torch.cat([next_action, next_teammate_actions], dim=1)  # shape: [batch, num+1, dim]
                next_reward = episodes_data["reward"][:, ts+1].to(self.device)

                next_next_obs = episodes_data["obs"][:, ts+2, :].to(self.device)
                

                # Forward pass through encoder
                action = action.squeeze(1)
                next_action = next_action.squeeze(1)
                z, h_new = self.liam_encoder(next_obs, action, h)
                h = h_new

                # Forward pass through reconstruction decoder
                reconstructed_obs, reconstructed_actions = self.reconstruction_decoder(z)
                
                # Calculate reconstruction loss
                obs_flat = next_team_states.reshape(next_team_states.shape[0], -1)
                actions_flat = next_team_actions.reshape(next_team_actions.shape[0], -1).to(torch.float32)
                
                reconstruction_loss = F.mse_loss(reconstructed_obs, obs_flat) + F.cross_entropy(reconstructed_actions, actions_flat, reduction='mean')

                # Get current Q value
                Q = self.Q_net(z, action, obs)

                # Get next state value
                next_z, _ = self.liam_encoder(next_next_obs, next_action, h)
                V = self.V_net(z, next_obs)
                next_V = self.V_net(next_z, next_next_obs)

                # Get policy output for next state and take argmax
                action_logits = self.policy_net(z, next_obs)
                target_value = next_reward.unsqueeze(-1) + self.gamma * next_V
                V_loss = 0.5 * F.mse_loss(V, target_value.detach())
                next_action = next_action.to(torch.float32)

                policy_loss = ((Q - V).detach() * F.cross_entropy(action_logits, next_action, reduction='none') ).mean() + self.alpha * F.cross_entropy(action_logits, next_action, reduction='mean')
                entropy_loss = -torch.sum(F.softmax(action_logits, dim=-1) * F.log_softmax(action_logits, dim=-1), dim=-1).mean()
                a2c_loss = V_loss + policy_loss + self.sita * entropy_loss

                total_loss = reconstruction_loss + self.beta * a2c_loss
                sum_total_loss += total_loss
                sum_reconstruction_loss += reconstruction_loss
                sum_a2c_loss += a2c_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.liam_encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.reconstruction_decoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.V_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()
        
        return {
            'total_loss': sum_total_loss.item(),
            'reconstruction_loss': sum_reconstruction_loss.item(),
            'a2c_loss': sum_a2c_loss.item()
        }

    def train(self, episodes_data, max_ep_len):
        self.liam_encoder.train()
        self.reconstruction_decoder.train()
        self.Q_net.train()
        self.V_net.train()
        self.policy_net.train()
        loss_dict = self.train_step(episodes_data, max_ep_len)
        return loss_dict

  