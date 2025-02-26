from TestGame import Test
from Agent.Adhoc_DT import Adhoc_DT
from utils_dt import load_config
from Networks.ReturnNet import ReturnNet
from Networks.TeammateEncoder import TeammateEncoder
from Networks.AdhocAgentEncoder import AdhocAgentEncoder
from Networks.GoalDecoder import GoalDecoder
from Networks.dt_models.decision_transformer import DecisionTransformer
import torch
from Networks.dt_models.rtg_dt import RTG_DT_lbf
from Agent.DtAgent import DtAgent
from Agent.RandomAgent import RandomAgent

config = load_config(f"./test_ckpt/config.yaml")
device = config['device']
# dt = RTG_DT_lbf(
#     state_dim=config["state_dim"],
#     num_agents=config["num_agents"],
#     act_dim=config["act_dim"],
#     max_length=config["K"],
#     max_ep_len=config["max_ep_len"],
#     hidden_size=config['dt_embed_dim'],
#     n_layer=config['n_layer'],
#     n_head=config['n_head'],
#     n_inner=4*config['dt_embed_dim'],
#     activation_function=config['dt_activation_function'],
#     n_positions=1024,
#     resid_pdrop=config['dt_dropout'],
#     attn_pdrop=config['dt_dropout'],
#     ).to(device)

adhocagentEncoder = AdhocAgentEncoder(state_dim=config["state_dim"], embed_dim=config["embed_dim"]).to(device)
returnnet = ReturnNet(input_dim=config["embed_dim"]).to(device)
goaldecoder = GoalDecoder(input_dim=config["embed_dim"], scalar_dim=1, hidden_dim=512, output_dim=config["state_dim"], num=config["num_agents"], state_dim=config["state_dim"]).to(device)
dt = DecisionTransformer(
    state_dim=config["state_dim"],
    num_agents=config["num_agents"],
    act_dim=config["act_dim"],
    max_length=config["K"],
    max_ep_len=config["max_ep_len"],
    hidden_size=config['dt_embed_dim'],
    n_layer=config['n_layer'],
    n_head=config['n_head'],
    n_inner=4*config['dt_embed_dim'],
    activation_function=config['dt_activation_function'],
    n_positions=1024,
    resid_pdrop=config['dt_dropout'],
    attn_pdrop=config['dt_dropout'],
    ).to(device)
import os
import csv

# 创建结果保存文件
results_file = './test_ckpt/test_results.csv'
with open(results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Returns', 'Variance'])

# 遍历权重文件夹
weight_dir = './test_ckpt/weight'
for filename in sorted(os.listdir(weight_dir)):
    if filename.endswith('.pth'):
        # 加载模型权重
        checkpoint = torch.load(os.path.join(weight_dir, filename))
        dt.load_state_dict(checkpoint['model_state_dict']['model'])
        adhocagentEncoder.load_state_dict(checkpoint['model_state_dict']['adhocencoder'])
        returnnet.load_state_dict(checkpoint['model_state_dict']['returnnet'])
        goaldecoder.load_state_dict(checkpoint['model_state_dict']['goaldecoder'])
        
        # 获取epoch数
        epoch = int(filename.split('_')[1].split('.')[0])
        print(f"Testing model from epoch {epoch}")

        # 创建agent并测试
        agent = Adhoc_DT(
            dt_model=dt,
            state_encoder=adhocagentEncoder,
            return_net=returnnet,
            goal_decoder=goaldecoder,
            env_type="PP4a"
        )

        # agent = DtAgent(dt, "LBF")
        test = Test('PP4a')
        returns, var = test.test_game(1, agent, config["K"])
        # print(f"Epoch {epoch} - Returns: {returns}, Variance: {var}")

        # 保存结果
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, returns, var])