from pathlib import Path

import torch as t
from torch.utils.data import Dataset


class OlfactionDataset(Dataset):
    def __init__(self, root, channels=3, train=True, transform=None, shuffled=False):
        root = Path(root)
        root = root / "train" if train else root / "val"
        self.files = [p for p in Path(root).rglob("*.pt")]
        self.transform = transform
        self.channels = channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x, y = t.load(self.files[index], weights_only=True, map_location="cpu")
        x, y = x.reshape((self.channels, 32, 32)), y.squeeze(0)

        if self.transform is not None:
            x = self.transform(x)

        return x, y
