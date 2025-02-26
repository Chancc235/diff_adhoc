import torch
from utils_dt import preprocess_data
data_path = "data/PP4a_episodes_datas_rtg.pt"
sav_data_path = "data/PP4a_episodes_datas_rtg_new.pt"
data = torch.load(data_path)
new_data = data[::2]
torch.save(new_data, sav_data_path)