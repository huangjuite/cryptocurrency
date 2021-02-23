
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


class CandleTransformer(nn.Module):
    def __init__(
        self,
        sequence_length=192,
        features=5,
        embed_dim=64,
        nhead=8,
        num_layers=6,
        order=3,
        output_sequence=96,
    ):
        super(CandleTransformer, self).__init__()

        # linear projection layer
        self.linear_projection = nn.Linear(features, embed_dim)

        # learnable extra token positioned at very first
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))

        # positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(sequence_length+1, 1, embed_dim))

        # transformer encder block
        single_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            single_layer, num_layers=num_layers)

        # output regression parameter mlp layer
        self.regression_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, order),
            nn.Linear(order, output_sequence),
        )

    def forward(self, x):
        x = self.linear_projection(x)
        _, b, _ = x.shape
        cls_tokens = self.cls.repeat(1, b, 1)

        # concat extra learnable token to first position
        x = torch.cat((cls_tokens, x), dim=0)

        # add positional embedding
        x += self.pos_embedding

        # transformer encoder
        x = self.transformer_encoder(x)

        # mlp head for n-th order linear regression
        mlp_head = x[0, :, :]
        output = self.regression_mlp(mlp_head)

        return output