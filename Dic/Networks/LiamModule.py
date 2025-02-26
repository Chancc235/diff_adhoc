import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim=128):
        super().__init__()
        
        # Input layer to get to hidden size
        self.input_layer = nn.Linear(obs_dim + action_dim, 256)  # [batch_size, obs_dim + action_dim] -> [batch_size, 256]
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=256,  # [batch_size, seq_len, 256]
            hidden_size=256,  # [batch_size, seq_len, 256]
            num_layers=1,
            batch_first=True
        )
        self.hidden_dim = 256
        # Output projection
        self.output_layer = nn.Linear(256, z_dim)  # [batch_size, 256] -> [batch_size, z_dim]
        
    def forward(self, obs, last_action, h=None):
        # Concatenate observation and action
        x = torch.cat([obs, last_action], dim=-1)  # [batch_size, obs_dim + action_dim]
        
        # Initial projection
        x = torch.relu(self.input_layer(x))  # [batch_size, 256]
        
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, 256]
            
        # Pass through GRU
        x, new_h = self.gru(x, h)  # x: [batch_size, seq_len, 256], new_h: [1, batch_size, 256]
        
        # Take final sequence output
        x = x[:,-1]  # [batch_size, 256]
        
        # Project to latent dimension
        z = self.output_layer(x)  # [batch_size, z_dim]
        
        return z, new_h

class ReconstructionDecoder(nn.Module):
    def __init__(self, z_dim, obs_dim, action_dim, num_agents):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        
        # Decoder network for reconstructing observations and actions
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # [batch_size, z_dim] -> [batch_size, 256]
            nn.ReLU(),
            nn.Linear(256, 256),  # [batch_size, 256] -> [batch_size, 256]
            nn.ReLU(),
            nn.Linear(256, (obs_dim + action_dim) * num_agents)  # [batch_size, 256] -> [batch_size, (obs_dim + action_dim) * num_agents]
        )
        
    def forward(self, z):
        # Decode latent z to concatenated obs and actions
        decoded = self.decoder(z)  # [batch_size, (obs_dim + action_dim) * num_agents]
        
        # Split and reshape output into obs and actions
        obs = decoded[..., :self.obs_dim * self.num_agents]  # [batch_size, obs_dim * num_agents]
        actions = decoded[..., self.obs_dim * self.num_agents:]  # [batch_size, action_dim * num_agents]
        
        return obs, actions
    
class PolicyNetwork(nn.Module):
    def __init__(self, z_dim, action_dim, obs_dim):
        super().__init__()
        
        self.policy = nn.Sequential(
            nn.Linear(z_dim+obs_dim, 256),  # [batch_size, z_dim+obs_dim] -> [batch_size, 256]
            nn.ReLU(),
            nn.Linear(256, action_dim),  # [batch_size, 256] -> [batch_size, action_dim]
            nn.Softmax(dim=-1)
        )
        
    def forward(self, z, obs):
        x = torch.cat([z, obs], dim=-1)  # [batch_size, z_dim + obs_dim]
        action_logits = self.policy(x)  # [batch_size, action_dim]
        return action_logits


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim):
        super().__init__()
        
        self.q_net = nn.Sequential(
            nn.Linear(z_dim + action_dim + obs_dim, 256),  # [batch_size, z_dim + action_dim + obs_dim] -> [batch_size, 256]
            nn.ReLU(),
            nn.Linear(256, 1)  # [batch_size, 256] -> [batch_size, 1]
        )
    def forward(self, z, action, obs):
        x = torch.cat([z, action, obs], dim=-1)  # [batch_size, z_dim + action_dim + obs_dim]
        q_value = self.q_net(x)  # [batch_size, 1]
        return q_value

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, z_dim):
        super().__init__()
        
        self.value = nn.Sequential(
            nn.Linear(z_dim + obs_dim, 256),  # [batch_size, z_dim + obs_dim] -> [batch_size, 256]
            nn.ReLU(),
            nn.Linear(256, 1)  # [batch_size, 256] -> [batch_size, 1]
        )
        
    def forward(self, z, obs):
        x = torch.cat([z, obs], dim=-1)  # [batch_size, z_dim + obs_dim]
        state_value = self.value(x)  # [batch_size, 1]
        return state_value

# Test cases
if __name__ == "__main__":
    # Test parameters
    batch_size = 32
    obs_dim = 21
    action_dim = 6
    z_dim = 128
    num_agents = 2
    
    # Test Encoder
    encoder = Encoder(obs_dim, action_dim, z_dim)
    obs = torch.randn(batch_size, obs_dim)
    last_action = torch.randn(batch_size, action_dim)
    z, h = encoder(obs, last_action)
    print("Encoder output shape:", z.shape)  # Should be [batch_size, z_dim]
    print("Hidden state shape:", h.shape)  # Should be [2, batch_size, 256]
    
    # Test ReconstructionDecoder
    decoder = ReconstructionDecoder(z_dim, obs_dim, action_dim, num_agents)
    decoded_obs, decoded_actions = decoder(z)
    print("Decoded obs shape:", decoded_obs.shape)  # Should be [batch_size, obs_dim * num_agents]
    print("Decoded actions shape:", decoded_actions.shape)  # Should be [batch_size, action_dim * num_agents]
    
    # Test PolicyNetwork
    policy = PolicyNetwork(z_dim, action_dim)
    action_logits = policy(z)
    print("Policy output shape:", action_logits.shape)  # Should be [batch_size, action_dim]
    
    # Test QNetwork
    q_net = QNetwork(obs_dim, action_dim, z_dim)
    q_value = q_net(z, last_action, obs)
    print("Q-value shape:", q_value.shape)  # Should be [batch_size, 1]
    
    # Test ValueNetwork
    value_net = ValueNetwork(obs_dim, z_dim)
    state_value = value_net(z, obs)
    print("State value shape:", state_value.shape)  # Should be [batch_size, 1]