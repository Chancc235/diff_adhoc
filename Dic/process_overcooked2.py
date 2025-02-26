import torch
import torch.nn.functional as F
from tqdm import tqdm

def to_onehot(onehot_tensor, indices_to_onehot, num_classes=10):
    for index in indices_to_onehot:
        # 获取指定维度的值
        values = onehot_tensor[..., index].to(torch.long) + 1  # 假设值需要加1
        # 将该维度的值转为 one-hot 编码
        onehot_values = F.one_hot(values, num_classes=num_classes)
        # 用 one-hot 编码的结果替换原 tensor 中的相应部分
        onehot_tensor = onehot_tensor.clone()
        onehot_tensor[..., index:index+1] = onehot_values
    return onehot_tensor.to(torch.float32)

data_path = "data/overcooked_episodes_datas_rtg_new.pt"
sav_data_path = "data/overcooked_episodes_datas_rtg_new2.pt"
data = torch.load(data_path)
new_data = []
print("Loaded data.")

indices_to_onehot = [9, 10, 23, 24]
for i, v in tqdm(enumerate(data), total=len(data), desc="Processing data"):
    s = to_onehot(v['state'][..., :29], indices_to_onehot)
    n = to_onehot(v['next_state'][..., :29], indices_to_onehot)
    o = to_onehot(v['obs'][..., :29], indices_to_onehot)

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

    if (i + 1) % 1000 == 0:
        torch.save(new_data, sav_data_path)  # 可选择在一定间隔后保存

print("Final data shape:", new_data[-1]['state'].shape)
torch.save(new_data, sav_data_path)
print(f"Data saved to {sav_data_path}.")
