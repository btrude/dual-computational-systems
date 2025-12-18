from pathlib import Path

import torch as t
from torch.utils.data import Dataset


class HippocampusDataset(Dataset):
    def __init__(self, root, train=True, transform=None, labels_are_primitives=False):
        self.labels_are_primitives = labels_are_primitives

        if train:
            root = Path(root) / "train"
        else:
            root = Path(root) / "val"

        self.files = [p for p in Path(root).rglob("*.pt")]
        self.transform = transform
        self.primitives = t.load(Path(root) / ".." / "primitives.pt", weights_only=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x, y = t.load(self.files[index], weights_only=True)
        x = x.to(t.float32).reshape((32, 32, 1))

        if not self.labels_are_primitives:
            y = x.clone().reshape((x.shape[0] * x.shape[1],))

        x, y = x.numpy(), y.numpy()

        if self.transform is not None:
            x = self.transform(x)

        return x, y
