import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ODITSTrainer:
    def __init__(
        self,
        teamwork_encoder,
        proxy_encoder,
        teamwork_decoder,
        proxy_decoder,
        integrating_net,
        marginal_net,
        optimizer,
        gamma=0.99,
        beta=0.1,
        batch_size=512,
        act_dim=5,
        update_freq=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        # Models
        self.teamwork_encoder = teamwork_encoder.to(device)
        self.proxy_encoder = proxy_encoder.to(device)
        self.teamwork_decoder = teamwork_decoder.to(device)
        self.proxy_decoder = proxy_decoder.to(device)
        self.integrating_net = integrating_net.to(device)
        self.marginal_net = marginal_net.to(device)
        
        # Optimizers
        self.optimizer = optimizer
        self.gamma = gamma
        self.beta = beta
        self.batch_size = batch_size
        self.act_dim = act_dim
        self.update_freq = update_freq
    def train_step(self, episodes_data, max_ep_len):
        
        batch_size = episodes_data["state"].size(0)
        h_0 = torch.zeros(1, batch_size, self.marginal_net.hidden_dim).to(self.device)
        
        sum_total_loss = 0.0
        sum_Q_loss = 0.0
        sum_MI_loss = 0.0
        for ts in range(0, max_ep_len-2):
            if ts % self.update_freq == 0:
                total_loss = 0.0
                h = h_0
                # 提取所有批次样本的该时间步的数据    
                team_states = episodes_data["state"][:, ts].to(self.device) # shape [batch, num, dim]
                obs = episodes_data["obs"][:, ts, :].to(self.device) 
                last_obs = episodes_data["obs"][:, ts-1, :].to(self.device)
                action = torch.clone(F.one_hot(episodes_data["action"][:, ts].to(torch.int64), num_classes=self.act_dim)).to(self.device).unsqueeze(1)  # shape: [batch, 1, dim]
                last_action = torch.clone(F.one_hot(episodes_data["action"][:, ts-1].to(torch.int64), num_classes=self.act_dim)).to(self.device)  # shape: [batch, 1, dim]
                teammate_actions = torch.clone(F.one_hot(episodes_data["teammate_action"][:, ts].to(torch.int64), num_classes=self.act_dim)).to(self.device)  # shape: [batch, num, dim]
                team_actions = torch.cat([action, teammate_actions], dim=1)  # shape: [batch, num+1, dim]
                reward = episodes_data["reward"][:, ts].to(self.device)

                # 提取t+1时刻的数据
                next_team_states = episodes_data["state"][:, ts+1].to(self.device)  # shape [batch, num, dim]
                next_obs = episodes_data["obs"][:, ts+1, :].to(self.device)
                next_action = torch.clone(F.one_hot(episodes_data["action"][:, ts+1].to(torch.int64), num_classes=self.act_dim)).to(self.device).unsqueeze(1)  # shape: [batch, 1, dim]
                next_teammate_actions = torch.clone(F.one_hot(episodes_data["teammate_action"][:, ts+1].to(torch.int64), num_classes=self.act_dim)).to(self.device)  # shape: [batch, num, dim]
                next_team_actions = torch.cat([next_action, next_teammate_actions], dim=1)  # shape: [batch, num+1, dim]

                # 前向传播 - 团队编码器
                team_mu, team_logvar = self.teamwork_encoder(team_states, team_actions)
                team_std = torch.exp(0.5 * team_logvar)
                team_dist = torch.distributions.Normal(team_mu, team_std)
                team_z = team_dist.rsample()
                
                # 前向传播 - 代理编码器
                action = action.squeeze(1)
                proxy_mu, proxy_logvar = self.proxy_encoder(last_obs, obs, last_action)
                proxy_std = torch.exp(0.5 * proxy_logvar)
                proxy_dist = torch.distributions.Normal(proxy_mu, proxy_std)
                proxy_z = proxy_dist.rsample()
                
                # 计算proxy_z和team_z之间的KL散度
                MI_loss = torch.distributions.kl_divergence(proxy_dist, team_dist).mean()
                
                # 计算边际效用
                marginal_input = torch.cat([obs, action], dim=-1).unsqueeze(1)
                marginal_utility, h_new = self.marginal_net(marginal_input, proxy_z, h)
                h = h_new

                # 计算整合效用
                integrated_utility = self.integrating_net(marginal_utility, team_z)
                
                # 算 u (t+1)
                num_actions = action.shape[-1]  # Get action dimension
                
                # Get next state utilities for all possible actions
                next_utilities = []

                next_action = next_action.squeeze(1)

                for a in range(num_actions):
                    # One-hot encode the action
                    candidate_action = torch.zeros(next_action.shape).to(self.device)
                    candidate_action[..., a] = 1
                    
                    # Recalculate z for next state
                    next_proxy_mu, next_proxy_logvar = self.proxy_encoder(obs, next_obs, action)
                    next_proxy_std = torch.exp(0.5 * next_proxy_logvar)
                    next_proxy_dist = torch.distributions.Normal(next_proxy_mu, next_proxy_std)
                    next_proxy_z = next_proxy_dist.rsample()

                    next_team_mu, next_team_logvar = self.teamwork_encoder(next_team_states, next_team_actions)
                    next_team_std = torch.exp(0.5 * next_team_logvar)
                    next_team_dist = torch.distributions.Normal(next_team_mu, next_team_std)
                    next_team_z = next_team_dist.rsample()

                    next_marginal_input = torch.cat([next_obs, candidate_action], dim=-1).unsqueeze(1)
                    next_marginal_utility, _ = self.marginal_net(next_marginal_input, next_proxy_z, h)
                    next_integrated_utility = self.integrating_net(next_marginal_utility, next_team_z)
                    next_utilities.append(next_integrated_utility)
                
                # Stack and get maximum utility
                next_utilities = torch.stack(next_utilities, dim=1)  # [batch_size, num_actions, 1]
                next_integrated_utility, _ = next_utilities.max(dim=1)  # Get max utility for each batch

                # 计算贝尔曼方程的损失
                # U(s,a) = r + gamma * max_a' U(s',a')
                # 当前状态-动作对的效用
                current_utility = integrated_utility.squeeze(-1)  # [batch_size]
                # 下一状态的最大效用
                next_max_utility = next_integrated_utility.squeeze(-1)  # [batch_size]
                # 计算目标值：即时奖励 + 折扣的下一状态最大效用
                reward = reward.unsqueeze(-1)
                target_utility = reward + self.gamma * next_max_utility.detach()
                # MSE损失
                Q_loss = torch.nn.functional.mse_loss(current_utility, target_utility)
                
                # 总损失
                total_loss = Q_loss + self.beta * MI_loss
                sum_total_loss += total_loss
                sum_Q_loss += Q_loss
                sum_MI_loss += MI_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.teamwork_encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.proxy_encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.teamwork_decoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.proxy_decoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.integrating_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.marginal_net.parameters(), max_norm=1.0)
                self.optimizer.step()
        
        num_updates = (max_ep_len - 2) // self.update_freq
        return {
            'total_loss': sum_total_loss.item() / num_updates,
            'Q_loss': sum_Q_loss.item() / num_updates,
            'MI_loss': sum_MI_loss.item() / num_updates
        }

    def train(self, episodes_data, max_ep_len):
        self.teamwork_encoder.train()
        self.proxy_encoder.train()
        self.teamwork_decoder.train()
        self.proxy_decoder.train()
        self.integrating_net.train()
        self.marginal_net.train()

        loss_dict = self.train_step(episodes_data, max_ep_len)
        return loss_dict

  