import os
from pathlib import Path
from uuid import uuid4

import fire
import torch as t
from torch.distributions import Normal, Uniform
from tqdm import tqdm

from dual_computational_systems.util import set_seed
from dual_computational_systems.util.constants import OLFACTION_DATASET_PATH


def group_prototypes(
    prototypes: t.Tensor,
    n_classes: int,
):
    n_proto, _ = prototypes.shape
    k = n_proto // n_classes
    assert n_proto % n_classes == 0

    prototypes_norm = t.nn.functional.normalize(prototypes, dim=1)

    sim = prototypes_norm @ prototypes_norm.T

    used = t.zeros(n_proto, dtype=t.bool)
    groups = []

    for _ in range(n_classes):
        seed_idx = t.where(~used)[0][0]

        sim_to_seed = sim[seed_idx].clone()
        sim_to_seed[used] = -t.inf
        topk = t.topk(sim_to_seed, k=k).indices

        groups.append(prototypes[topk])
        used[topk] = True

    return t.stack(groups)


def generate_samples(
    n_samples: int,
    n_classes: int,
    n_OR: int,
    prototype_assignments: t.Tensor,
    uniform: Uniform,
    normal: Normal = None,
):
    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    all_data = []
    all_labels = []

    for class_idx in tqdm(range(n_classes)):
        n_class_samples = samples_per_class + (1 if class_idx < remainder else 0)

        for _ in range(n_class_samples):
            data = uniform.sample((1, n_OR)) + (normal.sample((1, n_OR)) if normal is not None else 0)
            label = t.argmin(t.cdist(data, prototype_assignments).sum(dim=-1), dim=0)

            max_attempts = 100
            attempts = 0
            while label != class_idx and attempts < max_attempts:
                data = uniform.sample((1, n_OR)) + (normal.sample((1, n_OR)) if normal is not None else 0)
                label = t.argmin(t.cdist(data, prototype_assignments).sum(dim=-1), dim=0)
                attempts += 1

            if label == class_idx:
                all_data.append(data)
                all_labels.append(label)

    return all_data, all_labels


def main(
    n_proto=20,
    n_OR=1024,
    n_classes=10,
    n_train=int(50000),
    n_test=int(10000),
    sigma_ORN=0.1,
    seed=12345,
    min_distance_threshold=0.1,
):
    '''
    From e1, https://www.sciencedirect.com/science/article/pii/S0896627321006826
    To generate the standard dataset we first generate n_proto = 200 odor prototypes. Each prototype (x_tilde^(i)) activates n_OR = 50 ORN
    types or ORs, and the activation of each ORN type is sampled independently from a uniform distribution between 0 and 1, x~U(0,1). The
    200 prototypes are randomly assigned to n_classes = 100, with each class containing two prototypes. A given odor x_tilde is a vector
    in the 50-dimensional ORN-type space sampled the same way as the prototypes. When the network's input layer corresponds to ORNS, each
    ORN receives the activation of its OR plus an independent gaussian noise epsilon ~ N(0, sigma^2_ORN), where sigma^2_ORN = 0 by default.
    Its associated ground-truth class c is set to be the class of closest prototype, as measured by Euclidian distance in the ORN-type space.
    The training set consists of 1 million odors, the validation set consists of 8192.
    '''

    base_dir = Path(OLFACTION_DATASET_PATH)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(base_dir / "train", exist_ok=True)
    os.makedirs(base_dir / "val", exist_ok=True)

    set_seed(seed)

    if sigma_ORN > 0:
        normal = Normal(0, sigma_ORN)
    else:
        normal = None

    uniform = Uniform(0, 1)

    prototypes = uniform.sample((n_proto, n_OR))

    prototype_assignments = group_prototypes(
        prototypes=prototypes,
        n_classes=n_classes,
    )

    t.save(prototype_assignments, base_dir / "prototypes.pt")

    print("Generating training samples...")
    train_data_list, train_labels_list = generate_samples(
        n_samples=n_train,
        n_classes=n_classes,
        n_OR=n_OR,
        prototype_assignments=prototype_assignments,
        uniform=uniform,
        normal=normal,
    )

    train_data_stacked = t.cat(train_data_list, dim=0)

    print("Generating test samples...")
    test_data_list = []
    test_labels_list = []

    samples_per_class = n_test // n_classes
    remainder = n_test % n_classes

    for class_idx in tqdm(range(n_classes)):
        n_class_samples = samples_per_class + (1 if class_idx < remainder else 0)

        for _ in range(n_class_samples):
            data = uniform.sample((1, n_OR)) + (normal.sample((1, n_OR)) if normal is not None else 0)
            label = t.argmin(t.cdist(data, prototype_assignments).sum(dim=-1), dim=0)

            max_attempts = 1000
            attempts = 0

            while attempts < max_attempts:
                if label == class_idx:
                    distances = t.cdist(data, train_data_stacked)
                    min_dist = distances.min().item()

                    if min_dist >= min_distance_threshold:
                        break

                data = uniform.sample((1, n_OR)) + (normal.sample((1, n_OR)) if normal is not None else 0)
                label = t.argmin(t.cdist(data, prototype_assignments).sum(dim=-1), dim=0)
                attempts += 1

            if attempts < max_attempts:
                test_data_list.append(data)
                test_labels_list.append(label)

    train_indices = t.randperm(len(train_data_list))
    test_indices = t.randperm(len(test_data_list))

    train_data_list = [train_data_list[i] for i in train_indices]
    train_labels_list = [train_labels_list[i] for i in train_indices]
    test_data_list = [test_data_list[i] for i in test_indices]
    test_labels_list = [test_labels_list[i] for i in test_indices]

    dir_subdivision = 1000
    print("Saving training data...")
    for i in tqdm(range(0, len(train_data_list), dir_subdivision)):
        subdir = str(uuid4())[:8]
        os.makedirs(f"{base_dir}/train/{subdir}")

        for j in range(min(dir_subdivision, len(train_data_list) - i)):
            idx = i + j
            t.save(
                (train_data_list[idx], train_labels_list[idx]),
                f"{base_dir}/train/{subdir}/{str(uuid4())[:8]}.pt"
            )

    for i in tqdm(range(0, len(test_data_list), dir_subdivision)):
        subdir = str(uuid4())[:8]
        os.makedirs(f"{base_dir}/val/{subdir}")

        for j in range(min(dir_subdivision, len(test_data_list) - i)):
            idx = i + j
            t.save(
                (test_data_list[idx], test_labels_list[idx]),
                f"{base_dir}/val/{subdir}/{str(uuid4())[:8]}.pt"
            )

    print("\nClass distribution:")
    train_label_counts = t.bincount(t.stack(train_labels_list).view(-1))
    test_label_counts = t.bincount(t.stack(test_labels_list).view(-1))

    print(f"Train: {train_label_counts.tolist()}")
    print(f"Test: {test_label_counts.tolist()}")
    print(f"\nTotal train samples: {len(train_data_list)}")
    print(f"Total test samples: {len(test_data_list)}")


if __name__ == "__main__":
    fire.Fire(main)
