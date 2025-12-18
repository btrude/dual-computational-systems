from copy import deepcopy

from torchvision import transforms
from torchvision.datasets import CIFAR10

from dual_computational_systems.data.datasets import AuditionDataset
from dual_computational_systems.data.datasets import HippocampusDataset
from dual_computational_systems.data.datasets import OlfactionDataset
from dual_computational_systems.data.datasets import SomatosensationDataset
from dual_computational_systems.util.constants import CIFAR10_PATH
from dual_computational_systems.util.constants import AUDITION_DATASET_PATH
from dual_computational_systems.util.constants import HIPPOCAMPUS_DATASET_PATH
from dual_computational_systems.util.constants import OLFACTION_DATASET_PATH
from dual_computational_systems.util.constants import SOMATOSENSATION_DATASET_PATH


def prepare_dataset(
        dataset_name,
        test=True,
        test_only=False,
        disable_transform=False,
        **overrides,
    ):
    entry = deepcopy(REGISTRY[dataset_name])

    if test_only:
        train_or_test_or_both = "test"
    elif test:
        train_or_test_or_both = "train and test"
    else:
        train_or_test_or_both = "train"

    print(f"Loading {train_or_test_or_both} data for {dataset_name}...")

    if "train_root" in overrides:
        entry["train_kwargs"]["root"] = overrides["train_root"]

    if "test_root" in overrides:
        entry["test_kwargs"]["root"] = overrides["test_root"]

    if "train_channels" in overrides:
        entry["train_kwargs"]["channels"] = overrides["train_channels"]

    if "test_channels" in overrides:
        entry["test_kwargs"]["channels"] = overrides["test_channels"]

    if "train_transform" in overrides:
        entry["train_kwargs"]["transform"] = overrides["train_transform"]

    if "test_transform" in overrides:
        entry["test_kwargs"]["transform"] = overrides["test_transform"]

    if disable_transform:
        entry["train_kwargs"] = {
            k: v for k, v in entry["train_kwargs"].items()
            if k != "transform"
        }
        entry["test_kwargs"] = {
            k: v for k, v in entry["test_kwargs"].items()
            if k != "transform"
        }

    if test_only:
        test_dataset = entry["test_dataset_class"](**entry["test_kwargs"])
        setattr(test_dataset, "dataset_name", dataset_name)
        return test_dataset

    elif test:
        train_dataset = entry["train_dataset_class"](**entry["train_kwargs"])
        test_dataset = entry["test_dataset_class"](**entry["test_kwargs"])
        setattr(train_dataset, "dataset_name", dataset_name)
        setattr(test_dataset, "dataset_name", dataset_name)
        return train_dataset, test_dataset

    else:
        train_dataset = entry["train_dataset_class"](**entry["train_kwargs"])
        setattr(train_dataset, "dataset_name", dataset_name)
        return train_dataset


REGISTRY = {
    "cifar10": {
        "train_dataset_class": CIFAR10,
        "train_kwargs": {
            "root": CIFAR10_PATH,
            "train": True,
            "transform": transforms.Compose([
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            "download": True,
        },
        "test_dataset_class": CIFAR10,
        "test_kwargs": {
            "root": CIFAR10_PATH,
            "train": False,
            "transform": transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            "download": True,
        },
    },
    "olfaction": {
        "train_dataset_class": OlfactionDataset,
        "train_kwargs": {
            "root": OLFACTION_DATASET_PATH,
            "channels": 1,
            "train": True,
        },
        "test_dataset_class": OlfactionDataset,
        "test_kwargs": {
            "root": OLFACTION_DATASET_PATH,
            "channels": 1,
            "train": False,
        },
    },
    "somatosensation": {
        "train_dataset_class": SomatosensationDataset,
        "train_kwargs": {
            "root": SOMATOSENSATION_DATASET_PATH,
            "train": True,
            "transform": transforms.Compose([
                transforms.ToTensor(),
            ]),
        },
        "test_dataset_class": SomatosensationDataset,
        "test_kwargs": {
            "root": SOMATOSENSATION_DATASET_PATH,
            "train": False,
            "transform": transforms.Compose([
                transforms.ToTensor(),
            ]),
        },
    },
    "audition": {
        "train_dataset_class": AuditionDataset,
        "train_kwargs": {
            "root": AUDITION_DATASET_PATH,
            "train": True,
        },
        "test_dataset_class": AuditionDataset,
        "test_kwargs": {
            "root": AUDITION_DATASET_PATH,
            "train": False,
        },
    },
    "hippocampus": {
        "train_dataset_class": HippocampusDataset,
        "train_kwargs": {
            "root": HIPPOCAMPUS_DATASET_PATH,
            "train": True,
            "transform": transforms.Compose([
                transforms.ToTensor(),
            ]),
        },
        "test_dataset_class": HippocampusDataset,
        "test_kwargs": {
            "root": HIPPOCAMPUS_DATASET_PATH,
            "train": False,
            "transform": transforms.Compose([
                transforms.ToTensor(),
            ]),
        },
    },
}
