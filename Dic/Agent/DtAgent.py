from Networks.dt_models.decision_transformer import DecisionTransformer
import torch.nn.functional as F
import torch
class DtAgent:
    def __init__(self, dt_model, env_type='PP4a'):
        self.dt_model = dt_model
        self.env_type = env_type

    def to_onehot(self, onehot_tensor, indices_to_onehot=[9, 10, 20, 23, 24]):
        onehot_tensor = onehot_tensor[..., :29]
        onehot_tensor = torch.clamp(onehot_tensor, min=0)
        indices_to_onehot = [v + i * 10 for i, v in enumerate(indices_to_onehot)] # [9, 20, 43, 54]
        for index in indices_to_onehot:
            values = onehot_tensor[..., index].to(torch.int64)
            # 将该维度的值转为 one-hot 编码
            onehot_values = F.one_hot(values, num_classes=11) 
            onehot_tensor = torch.cat([onehot_tensor[..., :index], onehot_values, onehot_tensor[..., index + 1:]], dim=-1)

        return onehot_tensor.to(torch.float32)

    def take_action(self, o_list, a_list, R_list, t_list, act_dim):

        self.dt_model.eval()

        # 将列表转换为tensor list
        o_list = [torch.tensor(o).to("cuda").clone().detach() for o in o_list]
        a_list = [torch.tensor(a).to("cuda").clone().detach() for a in a_list]
        R_list = [torch.tensor(R).to("cuda").clone().detach() for R in R_list]
        t_list = [torch.tensor(t).to("cuda").clone().detach() for t in t_list]
        # 当前时间步的状态
        obs = o_list[0].view(1, o_list[0].shape[0])
        # 转换为可用的o ,a, s 序列
        o = torch.cat(o_list, dim=0)
        if len(a_list) != 0:
            a = torch.tensor(a_list).to(device="cuda")
        else:
            a = torch.zeros(len(o_list), 1).to(device="cuda")
        o = o.to(device="cuda")
        o = o.view(len(o_list), o_list[0].shape[0])
        t = torch.tensor(t_list).to(device="cuda")
        
        if self.env_type == "LBF":
            o = o[..., [-3, -2]]
        if self.env_type == "overcooked":
            o = self.to_onehot(o)
        R_list = [r.unsqueeze(0) if r.dim() == 0 else r for r in R_list]
        R = torch.cat(R_list, dim=0).unsqueeze(0)
        o = o.unsqueeze(0)
        a = a.unsqueeze(0)
        t = t.unsqueeze(0)
        # print(new_g.shape)
        # print(a.shape)
        # print(o.shape)
        # print(t)
        a_onehot = F.one_hot(a.to(torch.int64), num_classes=act_dim)
        ac_pred = self.dt_model.get_action(o, a_onehot, R, t)
        ac_pred = torch.argmax(ac_pred)
        # print(ac_pred)
        return [ac_pred.detach().cpu().numpy()]


class DtAgent_prom_ok:
    def __init__(self, dt_model, env_type='PP4a'):
        self.dt_model = dt_model
        self.env_type = env_type

    def to_onehot(self, onehot_tensor, indices_to_onehot=[9, 10, 20, 23, 24]):
        onehot_tensor = onehot_tensor[..., :29]
        onehot_tensor = torch.clamp(onehot_tensor, min=0)
        indices_to_onehot = [v + i * 10 for i, v in enumerate(indices_to_onehot)] # [9, 20, 43, 54]
        for index in indices_to_onehot:
            values = onehot_tensor[..., index].to(torch.int64)
            # 将该维度的值转为 one-hot 编码
            onehot_values = F.one_hot(values, num_classes=11) 
            onehot_tensor = torch.cat([onehot_tensor[..., :index], onehot_values, onehot_tensor[..., index + 1:]], dim=-1)

        return onehot_tensor.to(torch.float32)

    def take_action(self, states_p, o_list, actions_p, a_list, rtg_p, R_list, t_p, t_list, act_dim):

        self.dt_model.eval()
        # 将列表转换为tensor list
        o_list = [torch.tensor(o).to("cuda").clone().detach() for o in o_list]
        o_list = [self.to_onehot(o) for o in o_list]
        a_list = [torch.tensor(a).to("cuda").clone().detach() for a in a_list]
        R_list = [torch.tensor(R).to("cuda").clone().detach() for R in R_list]
        t_list = [torch.tensor(t).to("cuda").clone().detach() for t in t_list]

        if len(a_list) != 0:
            a = torch.tensor(a_list).to(device="cuda")
        else:
            a = torch.zeros(len(o_list), 1).to(device="cuda")

        states_p = [torch.tensor(sp).to("cuda").clone().detach() for sp in states_p]
        actions_p = [torch.tensor(sp).to("cuda").clone().detach() for sp in states_p]
        t_p = [torch.tensor(sp).to("cuda").clone().detach() for sp in t_p]
        rtg_p = [torch.tensor(sp).to("cuda").clone().detach() for sp in rtg_p]
        o_list = states_p + o_list
        a_list = actions_p + a_list
        R_list = rtg_p + R_list 
        t_list = t_p + t_list

        # 转换为可用的o ,a, s 序列
        o = torch.cat(o_list, dim=0)
        
        o = o.to(device="cuda")
        o = o.view(len(o_list), o_list[0].shape[0])
        t = torch.tensor(t_list).to(device="cuda")

        '''
        if self.env_type == "overcooked":
            o = self.to_onehot(o)
        '''
        R_list = [r.unsqueeze(0) if r.dim() == 0 else r for r in R_list]
        R = torch.cat(R_list, dim=0).unsqueeze(0)
        o = o.unsqueeze(0)
        a = a.unsqueeze(0)
        t = t.unsqueeze(0)
        # print(new_g.shape)
        # print(a.shape)
        # print(o.shape)
        # print(t)
        a_onehot = F.one_hot(a.to(torch.int64), num_classes=act_dim)
        ac_pred = self.dt_model.get_action(o, a_onehot, R, t)
        ac_pred = torch.argmax(ac_pred)
        # print(ac_pred)
        return [ac_pred.detach().cpu().numpy()]