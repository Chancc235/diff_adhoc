import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import torch
import os

# 定义文件路径列表
dir_root_path = '../saves/LBF/'
dir_path_list = ["LBF_trajectorys"]

# 初始化列表，用于拼接原始数据
actions_list = []
state_list = []
reward_list = []
done_list = []

# 读所有数据
for dir_path in dir_path_list:
    dir_path = os.path.join(dir_root_path, dir_path)
    all_files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    for file_path in all_files:
        # 打开并读取.plk 文件
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            reward_list.append(data["transition_data"]["reward"].squeeze(-1)[:1024])
            state_list.append(data["transition_data"]["obs"].squeeze(-1)[:1024])
            actions_list.append(data["transition_data"]["actions"].squeeze(-1)[:1024])
            done_list.append(data["transition_data"]["terminated"].squeeze(-1)[:1024])
        print(f"{file_path} done")

# 拼接所有数据
actions = torch.cat(actions_list, dim=0)
state = torch.cat(state_list, dim=0)
reward = torch.cat(reward_list, dim=0)
terminated = torch.cat(done_list, dim=0)

# 检查拼接后的数据形状
print(actions.shape)

num_episodes = state.shape[0]
max_steps = state.shape[1]
zero_state = torch.zeros(1, state.shape[2], state.shape[3])  # 创建一个全零的 state 用于填充 next_state
print(zero_state.shape)

episodes = []

total_reward = 0
cnt = 0
for i in range(num_episodes):
    
    if reward[i, :max_steps].sum() == 0:
        continue
    
    for agent_idx in range(2):
        cnt += 1
        total_reward += reward[i, :max_steps].sum()
        episode_data = {
            'state': state[i, :max_steps, ...],            # 当前状态，固定长度为 max_steps
            'obs': state[i, :max_steps, agent_idx, ...],  # adhoc agent状态，固定长度为 max_steps
            'action': actions[i, :max_steps, agent_idx,...],       # 动作，固定长度为 max_steps
            'reward': reward[i, :max_steps],             # 奖励，固定长度为 max_steps
            'next_state': state[i, 1:max_steps, ...].clone(),  # 下一个状态，最后时间步设为全零
            'done': terminated[i, :max_steps],            # 结束标志，固定长度为 max_steps
            'teammate_action': actions[i, :max_steps, [j for j in range(actions.shape[2]) if j != agent_idx], ...]
        }

        # 将 next_state 的最后一个时间步设置为全零
        episode_data['next_state'] = torch.cat((episode_data['next_state'], zero_state), dim=0)
        episodes.append(episode_data)

    print(f"episode {i}/{num_episodes} done")

print("有效episode数", cnt)
print("平均reward", total_reward/cnt)

torch.save(episodes, 'data/LBF_episodes_datas.pt')
