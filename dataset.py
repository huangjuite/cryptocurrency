import torch
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class KlineDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('./data/ETHUSDT_historical_klines_15m.csv')
        self.data = self.data.drop(['Time'], axis=1)
        self.data = self.data.to_numpy()[:122880, :]
        self.data = self.data.reshape((-1, 96, 5))

    def __getitem__(self, index):
        base_pos = 191  # 96*2-1=191
        tmp = np.vstack((self.data[index], self.data[index+1], self.data[index+2]))

        base = tmp[base_pos, -2]
        tmp[:, :4] -= base
        tmp[:, :4] /= base
        tmp[:, -1] /= tmp[base_pos, -1]
        x = tmp[:base_pos+1]
        y = tmp[base_pos+1:]
        y = np.mean(y[:, :-1], axis=1)

        return x, y

    def __len__(self):
        return len(self.data)-2
