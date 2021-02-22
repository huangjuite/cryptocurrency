
import torch
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from utils import *
import datetime
import mplfinance as mpf
import matplotlib.pyplot as plt

data = pd.read_csv('./data/ETHUSDT_historical_klines_15m.csv')

data = data.drop(['Time'], axis=1)
data = data.to_numpy()[:122880, :]

data = data.reshape((-1, 96, 5))

dl = 96
t0 = datetime.date.today()
t1 = t0 + datetime.timedelta(days=1)
t2 = t1 + datetime.timedelta(days=1)
t3 = t2 + datetime.timedelta(days=1)


lengths = len(data)-2
index = lengths-1

base_pos = 191  # 96*2-1=191
tmp = np.vstack((data[index], data[index+1], data[index+2]))

base = tmp[base_pos, -2]
tmp[:, :4] -= base
tmp[:, :4] /= base
tmp[:, -1] /= tmp[base_pos, -1]
# draw_klines_np(tmp, t0, t3, dl*3)

x = tmp[:base_pos+1]
y = tmp[base_pos+1:]

# draw_klines_np(x, t0, t2, dl*2)
# draw_klines_np(y, t0, t1, dl)
y = np.mean(y[:, :-1], axis=1)

lines = []
datelist = pd.date_range(t2, t3, dl)
for v, t in zip(y, datelist):
    lines.append([str(t), v])
draw_dataset(tmp, t0, t3, dl*3, lines)
