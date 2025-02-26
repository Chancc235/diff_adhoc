import torch
import torch.nn as nn

class TeammateEncoder(nn.Module):
    """
    队友编码器
    输入：
        states: 队友的状态，形状为 (num_agents, batch_size, state_dim)
    输出：
        全局编码，形状为 (batch_size, embed_dim)
    """
    def __init__(self, state_dim, embed_dim, num_heads):
        super(TeammateEncoder, self).__init__()
        self.embedding = nn.Linear(state_dim, embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.output_layer = nn.Linear(embed_dim, embed_dim * 2)  # 全连接层
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 平均池化
        self.embed_layer_norm = nn.LayerNorm(embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)


    def forward(self, states):
        # 输入 states 的形状：(num_agents, batch_size, state_dim)

        # embedding
        embedded_states = self.embedding(states)  # (num_agents, batch_size, embed_dim)
        embedded_states = self.embed_layer_norm(embedded_states)
        # embedded_states = torch.clamp(embedded_states, min=-1e6, max=1e6)


        # Multihead Attention
        attn_output, _ = self.self_attention(embedded_states, embedded_states, embedded_states)
        attn_output = self.attn_layer_norm(attn_output)

        # transfer to pooling
        attn_output = attn_output.permute(1, 2, 0)  # (batch_size, embed_dim, num_agents)
        # attn_output = torch.clamp(attn_output, min=-1e6, max=1e6)

        # average pooling
        pooled_output = self.pooling(attn_output).squeeze(-1)  # (batch_size, embed_dim)
        # pooled_output  = torch.clamp(pooled_output , min=-1e6, max=1e6)


        # output
        output = self.output_layer(pooled_output)  # (batch_size, embed_dim * 2)
        # 分割输出，得到均值和对数方差
        mean = output[..., :output.size(1) // 2]  # 取前一半作为均值
        log_var = output[..., output.size(1) // 2:]  # 取后一半作为对数方差
 
        return mean, log_var

