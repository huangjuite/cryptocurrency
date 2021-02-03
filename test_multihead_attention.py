
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MHAblock(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        feature_dim=5,
    ):
        super(MHAblock, self).__init__()

        self.linear_q = nn.Linear(feature_dim, embed_dim)
        self.linear_k = nn.Linear(feature_dim, embed_dim)
        self.linear_v = nn.Linear(feature_dim, embed_dim)

        self.multihead_attention1 = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        atten_output, atten_output_weights = self.multihead_attention1(q, k, v)

        return atten_output


class Encoder(nn.Module):
    def __init__(
        self,
        features=5,
        embed_dim=128,
        nhead=8,
        num_layers=6,
    ):
        super(Encoder, self).__init__()

        self.linear = nn.Linear(features, embed_dim)

        single_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead)
        self.Encoder = nn.TransformerEncoder(
            single_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.linear(x)
        x = self.Encoder(x)
        return x


if __name__ == '__main__':

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    batch_size = 32
    sequence_length = 128
    feature_dim = 5

    x = np.zeros([sequence_length, batch_size, feature_dim])
    x = torch.Tensor(x).to(device)

    # model
    model = Encoder(
        features=5,
        embed_dim=128,
        nhead=8,
        num_layers=6,
    ).to(device)
    
    x = model(x)

    print(x.shape)
