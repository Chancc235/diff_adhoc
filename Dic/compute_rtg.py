import torch
from utils_dt import preprocess_data
data_path = "data/overcooked_episodes_datas2.pt"
sav_data_path = "data/overcooked_episodes_datas2_rtg.pt"
# data_path = "data/PP4a_test.pt"
# sav_data_path = "data/PP4a_test_rtg.pt"
data = torch.load(data_path)
# print(data[10000]["rtg"])
data = preprocess_data(data)
print(data[0]["rtg"])
print(data[0]["reward"])
torch.save(data, sav_data_path)