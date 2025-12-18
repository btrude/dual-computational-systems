import json
from copy import deepcopy

import torch

from dual_computational_systems.models import DistributedNetwork
from dual_computational_systems.models import LocallyConnectedNetwork


def load_model_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only=True)
    ckpt["model"] = {
        k[10:] if k[:10] == "_orig_mod." else k: v
        for k, v in ckpt["model"].items()
    }

    model.load_state_dict(ckpt["model"])


def print_model(model, write_func=None):
    if write_func is None:
        write_func = print
    write_func(str(model))
    write_func(f"# Params: {sum([p.numel() for p in model.parameters() if p.requires_grad]):,}")


def prepare_model(model_name, device="cpu", do_print_model=True, **overrides):
    cls, kwargs = deepcopy(REGISTRY[model_name])
    kwargs = {} if kwargs is None else kwargs

    for key, value in overrides.items():
        kwargs[key] = value

    if do_print_model:
        print(f"Using {model_name} on {device} with config {json.dumps(kwargs, indent=4)}")

    model = cls(**kwargs)
    if device != "cpu":
        model = model.to(device)

    if do_print_model:
        print_model(model)

    return model


REGISTRY = {
    "distributed_network": (DistributedNetwork, dict(channels=1)),
    "locally_connected_network": (LocallyConnectedNetwork, dict(channels=1)),
}
