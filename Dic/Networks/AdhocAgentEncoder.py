import torch
import torch.nn as nn

class AdhocAgentEncoder(nn.Module):
    """
    个体编码器
    输入：
        state: 个体的状态，形状为 (batch_size, state_dim)
    输出：
        全局编码，形状为 (batch_size, embed_dim)
    """
    def __init__(self, state_dim, embed_dim, hidden_dim1=512, hidden_dim2=256):
        super(AdhocAgentEncoder, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        
        self.fc3 = nn.Linear(hidden_dim2, embed_dim * 2)
        #self.bn3 = nn.BatchNorm1d(embed_dim)
        
        # 激活函数
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, state):
        # 输入 state 的形状：(batch_size, state_dim)
        
        x = self.fc1(state)        # (batch_size, 512)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.fc2(x)            # (batch_size, 256)
        x = self.bn2(x)
        x = self.activation(x)
        
        x = self.fc3(x)            # (batch_size, embed_dim * 2)
                
        # 分割输出，得到均值和对数方差
        mean = x[..., :x.size(1) // 2]  # 取前一半作为均值
        log_var = x[..., x.size(1) // 2:]  # 取后一半作为对数方差
 
        return mean, log_var
