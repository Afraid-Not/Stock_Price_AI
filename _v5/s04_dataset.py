import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, file_path, window_size=60):
        self.df = pd.read_csv(file_path)
        self.window_size = window_size
        
        # Target은 첫 번째 컬럼, 나머지는 Feature
        self.y = self.df['target'].values
        self.X = self.df.drop(columns=['target']).values

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        # idx부터 window_size만큼의 데이터를 가져옴
        x = self.X[idx : idx + self.window_size]
        y = self.y[idx + self.window_size] # 윈도우 바로 다음 날의 상승/하락
        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()

def get_dataloaders(file_path, batch_size=32, train_split=0.8):
    dataset = StockDataset(file_path)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader