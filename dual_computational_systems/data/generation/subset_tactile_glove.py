import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dual_computational_systems.data.datasets import SomatosensationDataset
from dual_computational_systems.util import set_seed
from dual_computational_systems.util.constants import SOMATOSENSATION_DATASET_PATH
from dual_computational_systems.util.tactile_glove import SOMATOSENSATION_SUBSET_LABELS


def main():
    seed = 0
    set_seed(seed)
    dataset = SomatosensationDataset(SOMATOSENSATION_DATASET_PATH, subset=False)
    samples, labels, orig_labels = [], [], []

    for i, y in enumerate(tqdm(dataset.labels)):
        if y.item() not in SOMATOSENSATION_SUBSET_LABELS:
            continue

        orig_labels.append(y.item())
        y = SOMATOSENSATION_SUBSET_LABELS[y.item()]

        samples.append(i)
        labels.append(y)

    train, test = train_test_split(
        samples,
        train_size=50_000,
        test_size=10_000,
        stratify=labels,
        random_state=seed,
    )

    assert np.max(labels) == list(SOMATOSENSATION_SUBSET_LABELS.values())[-1]
    train = np.array(train)
    test = np.array(test)
    print(f"Train samples: {len(train)}, Test Samples: {len(test)}")

    for i in range(0, 27):
        print(((np.array(orig_labels)[np.array(orig_labels) == i].shape, i)))

    np.save(f"{SOMATOSENSATION_DATASET_PATH}/subset_50k.npy", train)
    np.save(f"{SOMATOSENSATION_DATASET_PATH}/subset_test_10k.npy", test)


if __name__ == "__main__":
    main()
