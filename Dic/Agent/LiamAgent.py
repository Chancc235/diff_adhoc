import torch.nn.functional as F
import torch
class LiamAgent:
    def __init__(self, liam_encoder, policy_net, env_type='PP4a'):
        self.liam_encoder = liam_encoder
        self.policy_net = policy_net
        self.env_type = env_type

    def take_action(self, obs, last_action, h, num_actions):
        self.liam_encoder.eval()
        self.policy_net.eval()
        # Convert last_action to one-hot encoding
        last_action_onehot = F.one_hot(torch.tensor(last_action), num_classes=num_actions).float().to("cuda").unsqueeze(0).to(torch.float32)
        obs = torch.tensor(obs).to("cuda").clone().detach().unsqueeze(0).to(torch.float32)
        if h is None:
            z, h_new = self.liam_encoder(obs, last_action_onehot)
        else:
            z, h_new = self.liam_encoder(obs, last_action_onehot, h)

        action_logits = self.policy_net(z, obs)
        action = torch.argmax(action_logits, dim=-1)
        # Convert action tensor to scalar
        action = action.item()
        return [action], h_new