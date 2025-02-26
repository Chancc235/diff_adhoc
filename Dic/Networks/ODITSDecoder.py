import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# HyperNetwork: 生成主网络的权重
class ProxyDecoder(nn.Module):
    def __init__(self, hyper_input_dim, primary_layer_dims):
        super(ProxyDecoder, self).__init__()
        self.hyper_fc = nn.Sequential(
            nn.Linear(hyper_input_dim, sum([d[0] * d[1] for d in primary_layer_dims])),  # 展平成向量
            nn.BatchNorm1d(sum([d[0] * d[1] for d in primary_layer_dims]))
        )
        self.primary_layer_dims = primary_layer_dims

    def forward(self, z):
        # z is (batch_size, hyper_input_dim)
        flat_params = self.hyper_fc(z)  # (batch_size, total_params)
        
        # Initialize list to store parameters for each sample in batch
        batch_params = []
        for sample_params in flat_params:  # Process each sample separately
            params = []
            start = 0
            for dim_in, dim_out in self.primary_layer_dims:
                end = start + dim_in * dim_out
                params.append(sample_params[start:end].view(dim_out, dim_in))
                start = end
            batch_params.append(params[0])  # Since we only have one layer
        
        return torch.stack(batch_params)  # (batch_size, dim_out, dim_in)

# 主网络：包含 GRU 和由 HyperNetwork 提供参数的 FC
class MarginalUtilityNet(nn.Module):
    def __init__(self,  state_dim, action_dim, hidden_dim, fc_input_dim, fc_output_dim, hypernetwork):
        super(MarginalUtilityNet, self).__init__()
        # input_dim: 输入特征维度
        # hidden_dim: GRU隐藏状态维度
        # fc_input_dim: 全连接层输入维度 (等于hidden_dim)
        # fc_output_dim: 全连接层输出维度
        input_dim = state_dim + action_dim
        # 添加GRU前的全连接层
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.pre_bn = nn.BatchNorm1d(hidden_dim)
        self.pre_relu = nn.ReLU()
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.hypernetwork = hypernetwork  # 超网络用于生成全连接层权重
        self.fc_input_dim = fc_input_dim
        self.fc_output_dim = fc_output_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, z, h_0=None):
        # x: (batch_size, 1, input_dim) - 输入序列
        # z: (batch_size, hyper_input_dim) - 超网络的输入向量
        # h_0: (num_layers=1, batch_size, hidden_dim) - GRU初始隐藏状态,可选

        # 如果没有提供初始隐藏状态 h_0，则初始化为全零张量
        if h_0 is None:
            batch_size = x.size(0)
            h_0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)  # (1, batch_size, hidden_dim)
        # 通过GRU前的全连接层
        x = x.to(torch.float32)
        x = x.squeeze(1)  # Remove time dimension for BatchNorm
        x = self.pre_fc(x)
        x = self.pre_bn(x)
        x = self.pre_relu(x)
        x = x.unsqueeze(1)  # Add time dimension back
            
        # GRU 提取特征
        _, h_n = self.gru(x, h_0)  # h_n: (1, batch_size, hidden_dim)
        h_n = h_n.squeeze(0)  # (batch_size, hidden_dim)

        # 通过 HyperNetwork 生成 FC 层权重
        weight = self.hypernetwork(z)  # (batch_size, fc_output_dim, fc_input_dim)

        # 应用 FC 层
        output = torch.bmm(h_n.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)  # (batch_size, fc_output_dim)
        return output, h_n.unsqueeze(0)


# HyperNetwork: 生成主网络的权重
class TeamworkSituationDecoder(nn.Module):
    def __init__(self, hyper_input_dim, primary_layer_dims):
        super(TeamworkSituationDecoder, self).__init__()
        self.hyper_fc = nn.Sequential(
            nn.Linear(hyper_input_dim, sum([d[0] * d[1] for d in primary_layer_dims])),  # 展平成向量
            nn.BatchNorm1d(sum([d[0] * d[1] for d in primary_layer_dims]))
        )
        self.primary_layer_dims = primary_layer_dims

    def forward(self, z):
        # z is (batch_size, hyper_input_dim)
        flat_params = self.hyper_fc(z)  # (batch_size, total_params)
        
        # Initialize list to store parameters for each sample in batch
        batch_params = []
        for sample_params in flat_params:  # Process each sample separately
            params = []
            start = 0
            for dim_in, dim_out in self.primary_layer_dims:
                end = start + dim_in * dim_out
                params.append(sample_params[start:end].view(dim_out, dim_in))
                start = end
            batch_params.append(params[0])  # Since we only have one layer
        
        return torch.stack(batch_params)  # (batch_size, dim_out, dim_in)

# 主网络：包含 GRU 和由 HyperNetwork 提供参数的 FC
class IntegratingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_input_dim, fc_output_dim, hypernetwork):
        super(IntegratingNet, self).__init__()
        # input_dim: 输入特征维度
        # hidden_dim: GRU隐藏状态维度
        # fc_input_dim: 全连接层输入维度 (等于hidden_dim)
        # fc_output_dim: 全连接层输出维度

        # 添加GRU前的全连接层
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.pre_bn = nn.BatchNorm1d(hidden_dim)
        self.pre_relu = nn.ReLU()

        self.hypernetwork = hypernetwork  # 超网络用于生成全连接层权重
        self.fc_input_dim = fc_input_dim
        self.fc_output_dim = fc_output_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, z):
        # x: (batch_size, 1, input_dim) - 输入序列
        # z: (batch_size, hyper_input_dim) - 超网络的输入向量

        x = self.pre_fc(x)
        x = self.pre_bn(x)
        x = self.pre_relu(x)
        x = x.unsqueeze(1)
            
        # 通过 HyperNetwork 生成 FC 层权重
        weight = self.hypernetwork(z)  # (batch_size, fc_output_dim, fc_input_dim)
        # Now x is 3D and compatible with bmm
        output = torch.bmm(x, weight.transpose(1, 2))  # (batch_size, 1, fc_output_dim)
        return output

if __name__ == "__main__":

    # 测试参数设置
    input_dim = 32  # 输入特征维度
    hidden_dim = 64  # GRU隐藏状态维度
    fc_input_dim = hidden_dim  # 全连接层输入维度等于hidden_dim
    fc_output_dim = 1  # 全连接层输出维度
    hyper_input_dim = 32  # 超网络输入维度
    batch_size = 8

    # 创建超网络
    primary_layer_dims = [(fc_input_dim, fc_output_dim)]  # FC层的维度
    hypernetwork = TeamworkSituationDecoder(hyper_input_dim, primary_layer_dims)

    # 创建主网络
    decoder = IntegratingNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        fc_input_dim=fc_input_dim,
        fc_output_dim=fc_output_dim,
        hypernetwork=hypernetwork
    )

    # 创建测试数据
    x = torch.randn(batch_size, input_dim)  # 输入序列
    z = torch.randn(batch_size, hyper_input_dim)  # 超网络输入

    # 前向传播
    output = decoder(x, z).squeeze(1)  # Add squeeze(1) if you want 2D output

    # 打印网络信息
    print("\nNetwork Architecture:")
    print(f"Input Dimension: {input_dim}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"FC Input Dimension: {fc_input_dim}")
    print(f"FC Output Dimension: {fc_output_dim}")
    
    # 打印输入输出形状
    print("\nTensor Shapes:")
    print(f"Input x shape: {x.shape}")
    print(f"Hypernetwork input z shape: {z.shape}")
    print(f"Output shape: {output.shape}")
    
    # 验证输出
    print("\nOutput Statistics:")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")

    # 验证模型参数
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal number of parameters: {total_params}")

