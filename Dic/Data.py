import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, episodes_data):
        
        self.episodes_data = episodes_data
    
    def __len__(self):
        # 返回数据集的样本数量
        return len(self.episodes_data)
    
    def __getitem__(self, idx):
        # 根据索引获取单个样本
        episodes_data = self.episodes_data[idx]
        
        return episodes_data
