import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as t
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from dual_computational_systems.util.fsd50k import AUDITION_RELABEL


class AuditionDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        root = Path(root) if isinstance(root, str) else root
        base_dir = Path(root).name
        vocabulary_path = Path(str(root).replace(base_dir, "fsd50k")) / "ground_truth" / "vocabulary.csv"

        self.vocabulary = {}
        with open(vocabulary_path) as vocab_csv:
            reader = csv.reader(vocab_csv)
            for token, label, _ in reader:
                self.vocabulary[label] = int(token)

        train_labels_path = Path(str(root).replace(base_dir, "fsd50k")) / "ground_truth" / "dev.csv"
        train_lookup = self._create_lookup(train_labels_path)
        train_data_path = root / "train"
        train_data = [p for p in Path(str(train_data_path)).rglob("*.pt")]
        train_labels = self._create_labels(train_data, train_lookup)

        test_labels_path = Path(str(root).replace(base_dir, "fsd50k")) / "ground_truth" / "eval.csv"
        test_lookup = self._create_lookup(test_labels_path)
        test_data_path = root / "val"
        test_data = [p for p in Path(str(test_data_path)).rglob("*.pt")]
        test_labels = self._create_labels(test_data, test_lookup)

        self.transform = transform

        train_indices, test_indices = train_test_split(
            list(range(len(train_data) + len(test_data))),
            train_size=0.9,
            test_size=0.1,
            stratify=train_labels + test_labels,
            random_state=0,
        )

        if train:
            self.data = np.array(train_data + test_data)[train_indices]
            self.labels = np.array(train_labels + test_labels)[train_indices]
        else:
            self.data = np.array(train_data + test_data)[test_indices]
            self.labels = np.array(train_labels + test_labels)[test_indices]

    def _create_lookup(self, labels_path):
        lookup = {}
        with open(labels_path) as vocab_csv:
            reader = csv.reader(vocab_csv)
            for i, (fname, labels, *_) in enumerate(reader):
                if not i:
                    continue

                labels = labels.split(",")

                tokens = defaultdict(int)
                for label in labels:
                    token = AUDITION_RELABEL[label]
                    tokens[token] += 1

                max_ = -1
                category = None
                for token, count in tokens.items():
                    if count > max_:
                        max_ = count
                        category = token

                lookup[int(fname)] = category

        return lookup

    def _create_labels(self, data, lookup):
        labels = []
        for cochleagram_path in data:
            file_num = int(str(cochleagram_path).split("/")[-1].split(".")[0])
            labels.append(lookup[file_num])
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = t.load(self.data[index], weights_only=True, map_location="cpu")
        y = self.labels[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y
