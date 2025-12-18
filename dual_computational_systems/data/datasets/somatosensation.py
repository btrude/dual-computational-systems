from pathlib import Path

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

from dual_computational_systems.util.tactile_glove import SOMATOSENSATION_SUBSET_LABELS


class SomatosensationDataset(Dataset):
    def __init__(self, root, train=True, transform=None, subset=True, rotate_thumb=True):
        self.metadata = loadmat(f"{root}/metadata.mat")
        self.data = self.metadata["pressure"]
        self.labels = self.metadata["objectId"]
        self.transform = transform
        self._rotate_thumb = rotate_thumb

        if subset:
            if train:
                self.subset = np.load(Path(root) / "subset_50k.npy")
            else:
                self.subset = np.load(Path(root) / "subset_test_10k.npy")

            self.data = self.data[self.subset]
            self.labels = self.labels[self.subset]

        self.data = ((self.data - self.data.min()) / (self.data.max() - self.data.min())).astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    @property
    def rotate_thumb(self):
        return self._rotate_thumb

    @rotate_thumb.setter
    def rotate_thumb(self, value):
        self._rotate_thumb = value

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index].item()
        x = x[..., np.newaxis]
        y = SOMATOSENSATION_SUBSET_LABELS[y]

        if self.rotate_thumb:
            temp = x.copy()
            thumb = np.rot90(temp[18:, 25:29, :], k=3)
            x[18:, 16:, :] = temp[18:, :16, :]
            x[18:22, 14:28, :] = thumb

        if self.transform is not None:
            x = self.transform(x)

        return x, y
