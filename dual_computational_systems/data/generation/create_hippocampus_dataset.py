import os
from pathlib import Path
from uuid import uuid4

import fire
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dual_computational_systems.util.constants import HIPPOCAMPUS_DATASET_PATH
from dual_computational_systems.util import set_seed


PRIMITIVES = torch.randint(0, 2, (100, 1, 128))


class RandomPrimitivesDataset(Dataset):
    def __init__(self, primitives, N, train=True):
        self.primitives = primitives
        self.N = N
        self.train = train

        self.samples = []
        self.labels = []

        rng = torch.Generator()
        seen_samples = set()

        with tqdm(total=N, desc=f"Generating {'train' if train else 'test'} samples") as pbar:
            while len(self.samples) < N:
                indices = torch.randint(0, len(self.primitives), (8,), generator=rng)
                selected_primitives = self.primitives[indices].squeeze(1).clone()

                if not self.train:
                    ones_indices = (selected_primitives == 1).nonzero(as_tuple=False)
                    num_ones = len(ones_indices)
                    num_to_zero = num_ones // 2

                    if num_to_zero > 0:
                        zero_indices = ones_indices[torch.randperm(num_ones, generator=rng)[:num_to_zero]]
                        selected_primitives[zero_indices[:, 0], zero_indices[:, 1]] = 0

                sample_hash = (tuple(indices.tolist()), selected_primitives.flatten().tolist().__repr__())

                if sample_hash not in seen_samples:
                    seen_samples.add(sample_hash)
                    self.samples.append(selected_primitives)
                    self.labels.append(indices)
                    pbar.update(1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def main():
    seed = 12345
    save_dir = Path(HIPPOCAMPUS_DATASET_PATH)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/train", exist_ok=True)
    os.makedirs(f"{save_dir}/val", exist_ok=True)

    set_seed(seed)
    dataset_train = RandomPrimitivesDataset(PRIMITIVES, 50000, train=True)
    set_seed(seed + 1)
    dataset_test = RandomPrimitivesDataset(PRIMITIVES, 10000, train=False)

    dir_subdivision = 1000
    subdirs_train = [str(uuid4())[:8] for _ in range(50000 // dir_subdivision)]
    subdirs_test = [str(uuid4())[:8] for _ in range(10000 // dir_subdivision)]

    for i, subdir in tqdm(enumerate(subdirs_train), desc="Saving train data"):
        os.makedirs(f"{save_dir}/train/{subdir}")

        for j in range(dir_subdivision):
            idx = j + (i * dir_subdivision)
            train_data, train_labels = dataset_train[idx]
            torch.save((train_data, train_labels), f"{save_dir}/train/{subdir}/{str(uuid4())[:8]}.pt")

    for i, subdir in tqdm(enumerate(subdirs_test), desc="Saving test data"):
        os.makedirs(f"{save_dir}/val/{subdir}")

        for j in range(dir_subdivision):
            idx = j + (i * dir_subdivision)
            test_data, test_labels = dataset_test[idx]
            torch.save((test_data, test_labels), f"{save_dir}/val/{subdir}/{str(uuid4())[:8]}.pt")

    torch.save(PRIMITIVES, f"{save_dir}/primitives.pt")


if __name__ == "__main__":
    fire.Fire(main)
