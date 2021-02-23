import os
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

from dataset import KlineDataset
from models import CandleTransformer
from utils import *

hyper_parameter = dict(
    batch_size=64,
    sequence_length=192,
    output_sequence=96,
    nhead_attention=8,
    transformer_layer=6,
    line_order=32,
    feature_dim=5,
    embed_dim=128,
    split_ratio=0.8,
    epoch=40,
    beta1=0.5,
    learning_rate=0.0002,
)

wandb.init(config=hyper_parameter,
           project="candle-prediction", name='candle-transformer-bigger')
config = wandb.config


torch.random.manual_seed(777)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)

# dataset
candle_dataset = KlineDataset()

train_len = int(config.split_ratio*len(candle_dataset))
test_len = len(candle_dataset) - train_len

train_dataset, test_dataset = random_split(
    candle_dataset, [train_len, test_len])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
)

# model
model = CandleTransformer(
    output_sequence=config.output_sequence,
    sequence_length=config.sequence_length,
    features=config.feature_dim,
    embed_dim=config.embed_dim,
    nhead=config.nhead_attention,
    num_layers=config.transformer_layer,
    order=config.line_order,
).to(device)
wandb.watch(model)


# train
optimizer = optim.Adam(model.parameters(),
                       lr=config.learning_rate, betas=(config.beta1, 0.999))
criterion = nn.MSELoss()
t = trange(config.epoch)
step = 0

for epoch in t:
    # train
    for x, y in train_loader:
        x = x[:, :config.sequence_length]
        x = x.to(device)
        x = x.transpose(0, 1)
        y = y.to(device)

        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, y)
        loss.backward()
        optimizer.step()

        step += 1
        metrics = {
            'train_loss': loss,
            'custom_step': step,
        }
        wandb.log(metrics)
        t.set_description("step %d, loss %.4f" % (step, loss.item()))

    # test
    model.eval()
    test_loss = []
    for fx, y in test_loader:
        x = fx[:, :config.sequence_length]
        x = x.to(device)
        x = x.transpose(0, 1)
        y = y.to(device)

        x_hat = model(x)
        loss = criterion(x_hat, y)
        test_loss.append(loss.item())

    model.train()
    test_loss = np.mean(test_loss)
    metrics = {
        'test_loss': test_loss,
        'custom_step': step,
    }
    wandb.log(metrics)

# visualization
fx = fx[0].numpy()
x_hat = x_hat[0].cpu().detach().cpu().numpy().astype(np.float64)
img = draw_dataset(fx, x_hat, config.output_sequence, save=True)
wandb.log({"example%d"%step: wandb.Image(img)})
    
torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth"))
