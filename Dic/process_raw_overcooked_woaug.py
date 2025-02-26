import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import torch
import os
from utils_dt import preprocess_data
import torch.nn.functional as F
from tqdm import tqdm

def to_onehot(onehot_tensor, indices_to_onehot=[9, 10, 19, 23, 24]):
    onehot_tensor = onehot_tensor[..., :29]
    onehot_tensor = torch.clamp(onehot_tensor, min=0)
    indices_to_onehot = [v + i * 10 for i, v in enumerate(indices_to_onehot)]
    for index in indices_to_onehot:
        values = onehot_tensor[..., index].to(torch.int64)
        # 将该维度的值转为 one-hot 编码
        onehot_values = F.one_hot(values, num_classes=11) 
        onehot_tensor = torch.cat([onehot_tensor[..., :index], onehot_values, onehot_tensor[..., index + 1:]], dim=-1)

    return onehot_tensor.to(torch.float32)


# 定义文件路径列表
dir_root_path = '../saves/overcooked/'

# dir_path_list = ['overcooked_trajectorys1', 'overcooked_trajectorys2', 'overcooked_trajectorys3', 'overcooked_trajectorys4', 'overcooked_trajectorys5']
dir_path_list = ['overcooked_trajectorys6']

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
            k = 1024
            if dir_path[-1] == '5' or dir_path[-1] == '6':
                k = 512
            try:
                data = pickle.load(file)
            except:
                print(file_path)
            
            reward_list.append(data["transition_data"]["reward"].squeeze(-1)[:k])
            state_list.append(data["transition_data"]["obs"].squeeze(-1)[:k])
            actions_list.append(data["transition_data"]["actions"].squeeze(-1)[:k])
            done_list.append(data["transition_data"]["terminated"].squeeze(-1)[:k])
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
    if reward[i, :max_steps].sum() == 10:
        continue
    for agent_idx in range(1):
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


data = preprocess_data(episodes)
print(data[0]["rtg"])
print(data[0]["reward"])


new_data = []

for i, v in enumerate(data):
    s = to_onehot(v['state'])
    n = to_onehot(v['next_state'])
    o = to_onehot(v['obs'])

    data_dict = {
        'state': s,
        'next_state': n,
        'obs': o,
        'action': v['action'],
        'reward': v['reward'],
        'done': v['done'],
        'teammate_action': v['teammate_action'],
        'rtg': v['rtg']
    }
    new_data.append(data_dict)
    print(f"{i} finished")


torch.save(new_data, 'data/overcooked_wo_aug.pt')