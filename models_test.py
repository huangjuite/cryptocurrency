
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import CandleTransformer

def test_mode() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(device)

    batch_size = 64
    sequence_length = 192
    output_sequence = 96
    feature_dim = 5

    x = np.zeros([sequence_length, batch_size, feature_dim])
    x = torch.Tensor(x).to(device)

    # model
    model = CandleTransformer(
        output_sequence=output_sequence,
        sequence_length=sequence_length,
        features=feature_dim,
        embed_dim=32,
        nhead=8,
        num_layers=6,
        order=3,
    ).to(device)

    x_hat = model(x)

    assert x_hat.shape == (batch_size, output_sequence)
