
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

from dataset import KlineDataset

torch.random.manual_seed(777)


def test_dataset_init() -> None:
    candle_dataset = KlineDataset()

    ratio = 0.8
    train_len = int(ratio*len(candle_dataset))
    test_len = len(candle_dataset) - train_len

    batch_size = 64
    train_dataset, test_dataset = random_split(
        candle_dataset, [train_len, test_len])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    for x, y in train_loader:
        assert y.shape == (batch_size, 96)
        assert type(y) == torch.Tensor
        assert x.shape == (batch_size, 192, 5)
        assert type(x) == torch.Tensor
        break
