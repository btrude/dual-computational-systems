import random

import numpy as np
import torch as t


def set_seed(seed=None):
    if seed is None:
        seed = np.random.randint(99999)

    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    np.random.seed(seed)

    print(f"Seed set to: {seed} ({'{0:b}'.format(int(seed))})")
