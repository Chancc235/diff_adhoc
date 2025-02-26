from torch import nn
import torch


class TeamworkSituationEncoder(nn.Module):
    def __init__(self, state_dim=75, action_dim=5, num_agents=2, output_dim=32, hidden_dim=128):
        super(TeamworkSituationEncoder, self).__init__()
        input_dim = (state_dim + action_dim) * num_agents
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # input: (batch_size, input_dim) -> output: (batch_size, hidden_dim) 
            nn.BatchNorm1d(hidden_dim),            # (batch_size, hidden_dim)
            nn.LeakyReLU(),                        # (batch_size, hidden_dim)
            nn.Linear(hidden_dim, output_dim),     # (batch_size, hidden_dim) -> (batch_size, output_dim)
            nn.BatchNorm1d(output_dim),            # (batch_size, output_dim)
            nn.LeakyReLU()                         # (batch_size, output_dim)
        )

        # 输出层: (batch_size, output_dim) -> (batch_size, output_dim * 2)
        self.output_layer = nn.Linear(output_dim, output_dim * 2)  # 全连接层输出均值和方差
        self.layer_norm = nn.LayerNorm(output_dim) # (batch_size, output_dim)
        self.output_bn = nn.BatchNorm1d(output_dim * 2)

    def forward(self, states, actions):  # states: (batch_size, num_agents, state_dim), actions: (batch_size, num_agents, action_dim)
        # 拼接状态和动作
        #print(states.shape, actions.shape)
        x = torch.cat([states, actions], dim=2)    # (batch_size, num_agents, state_dim+action_dim)
        x = x.reshape(x.shape[0], -1)
        
        # 编码
        encoded = self.encoder(x)                  # (batch_size, output_dim)
        encoded = self.layer_norm(encoded)         # (batch_size, output_dim)
        
        # 输出层
        output = self.output_layer(encoded)        # (batch_size, output_dim * 2)
        output = self.output_bn(output)
        
        # 分割输出得到均值和对数方差
        mean = output[..., :output.size(1)//2]     # (batch_size, output_dim)  前半部分作为均值
        log_var = output[..., output.size(1)//2:]  # (batch_size, output_dim)  后半部分作为对数方差
        
        return mean, log_var

class ProxyEncoder(nn.Module):
    def __init__(self, state_dim=75, action_dim=5, output_dim=32, hidden_dim=128):
        super(ProxyEncoder, self).__init__()
        input_dim = 2 * state_dim + action_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # input: (batch_size, input_dim) -> output: (batch_size, hidden_dim) 
            nn.BatchNorm1d(hidden_dim),            # (batch_size, hidden_dim)
            nn.LeakyReLU(),                        # (batch_size, hidden_dim)
            nn.Linear(hidden_dim, output_dim),     # (batch_size, hidden_dim) -> (batch_size, output_dim)
            nn.BatchNorm1d(output_dim),            # (batch_size, output_dim)
            nn.LeakyReLU()                         # (batch_size, output_dim)
        )

        # 输出层: (batch_size, output_dim) -> (batch_size, output_dim * 2)
        self.output_layer = nn.Linear(output_dim, output_dim * 2)  # 全连接层输出均值和方差
        self.layer_norm = nn.LayerNorm(output_dim) # (batch_size, output_dim)
        self.output_bn = nn.BatchNorm1d(output_dim * 2)

    def forward(self, last_states, states, last_actions):  # states: (batch_size, state_dim), actions: (batch_size, action_dim)
        # 拼接状态和动作
        # print(states.shape, actions.shape)
        x = torch.cat([last_states, states, last_actions], dim=1)    # (batch_size, state_dim+action_dim)
        
        # 编码
        x = x.to(torch.float32)
        encoded = self.encoder(x)                  # (batch_size, output_dim)
        encoded = self.layer_norm(encoded)         # (batch_size, output_dim)
        
        # 输出层
        output = self.output_layer(encoded)        # (batch_size, output_dim * 2)
        output = self.output_bn(output)
        
        # 分割输出得到均值和对数方差
        mean = output[..., :output.size(1)//2]     # (batch_size, output_dim)  前半部分作为均值
        log_var = output[..., output.size(1)//2:]  # (batch_size, output_dim)  后半部分作为对数方差
        
        return mean, log_var

if __name__ == "__main__":
    # 创建一个测试实例
    state_dim = 75
    action_dim = 5
    output_dim = 128
    hidden_dim = 256
    batch_size = 32
   # encoder = TeamworkSituationEncoder(state_dim=state_dim, action_dim=action_dim, num_agents=2, output_dim=output_dim, hidden_dim=hidden_dim)
    encoder = ProxyEncoder(state_dim=state_dim, action_dim=action_dim, output_dim=output_dim, hidden_dim=hidden_dim)
    # 创建随机输入数据
    test_states = torch.randn(batch_size, state_dim)
    test_actions = torch.randn(batch_size, action_dim)
    
    # 前向传播
    mean, log_var = encoder(test_states, test_actions)
    # 构造正态分布
    std = torch.exp(0.5 * log_var)  # 计算标准差
    dist = torch.distributions.Normal(mean, std)
    # 从构造的正态分布中采样
    z = dist.rsample()  # 使用rsample()进行重参数化采样

    print("\nSampled latent vector statistics:")
    print("Latent vector shape:", z.shape)
    print("Latent vector range:", torch.min(z).item(), "to", torch.max(z).item())
    print("Latent vector average:", torch.mean(z).item())
    # 验证输出维度
    print("States shape:", test_states.shape)
    print("Actions shape:", test_actions.shape)
    print("Mean shape:", mean.shape)
    print("Log variance shape:", log_var.shape)
    
    # 验证输出是否为有效值
    print("\nMean statistics:")
    print("Mean value range:", torch.min(mean).item(), "to", torch.max(mean).item())
    print("Mean average:", torch.mean(mean).item())
    
    print("\nLog variance statistics:")
    print("Log variance range:", torch.min(log_var).item(), "to", torch.max(log_var).item())
    print("Log variance average:", torch.mean(log_var).item())
    
    # 验证模型参数
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal number of parameters: {total_params}")