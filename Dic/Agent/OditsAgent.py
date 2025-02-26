import torch.nn.functional as F
import torch
class OditsAgent:
    def __init__(self, proxy_encoder, proxy_decoder, MarginalUtilityNet, env_type='PP4a'):
        self.proxy_encoder = proxy_encoder
        self.proxy_decoder = proxy_decoder
        self.marginal_net = MarginalUtilityNet
        self.env_type = env_type

    def take_action(self, last_obs, obs, last_action, h, num_actions):
        self.proxy_encoder.eval()
        self.proxy_decoder.eval()
        self.marginal_net.eval()
        # Convert last_action to one-hot encoding
        last_action_onehot = F.one_hot(torch.tensor(last_action), num_classes=num_actions).float().to("cuda").unsqueeze(0)
        last_obs = torch.tensor(last_obs).to("cuda").clone().detach().unsqueeze(0)
        obs = torch.tensor(obs).to("cuda").clone().detach().unsqueeze(0)

        proxy_mu, proxy_logvar = self.proxy_encoder(last_obs, obs, last_action_onehot)
        proxy_std = torch.exp(0.5 * proxy_logvar)
        proxy_dist = torch.distributions.Normal(proxy_mu, proxy_std)
        proxy_z = proxy_dist.rsample()

        utilities = []
        for a in range(num_actions):
            # One-hot encode the action
            candidate_action = F.one_hot(torch.tensor(a), num_classes=num_actions).float().to("cuda").unsqueeze(0)
            
            marginal_input = torch.cat([obs, candidate_action], dim=-1).unsqueeze(1)
            if h is None:
                marginal_utility, h_new = self.marginal_net(marginal_input, proxy_z)
            else:
                marginal_utility, h_new = self.marginal_net(marginal_input, proxy_z, h)
            utilities.append(marginal_utility)
        utilities = torch.stack(utilities, dim=1)
        utility, _ = utilities.max(dim=1)
        action = torch.argmax(utility, dim=1)
        # Convert action tensor to scalar
        action = action.item()
        return [action], h_new