import json

import torch.optim
from dual_computational_systems.optim import get_cosine_lr


def load_scheduler_checkpoint(scheduler, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    scheduler.load_state_dict(ckpt["scheduler"])


def load_opt_checkpoint(opt, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    opt.load_state_dict(ckpt["opt"])


def prepare_opt(model, optimizer_name, lr, **overrides):
    cls, kwargs = REGISTRY[optimizer_name]

    for key, value in overrides.items():
        kwargs[key] = value

    print(f"Using optimizer '{optimizer_name}' with config {json.dumps(kwargs, indent=4)}")
    return cls(model.parameters(), lr=lr, **kwargs)


def prepare_scheduler(scheduler_name, opt, num_training_steps, num_warmup_steps=100, min_lr=0.0):
    cls = SHD_REGISTRY[scheduler_name]
    return cls(
        opt,
        num_warmup_steps,
        num_training_steps,
        min_lr,
    )


REGISTRY = {
    "adamw": (torch.optim.AdamW, {}),
    "adam": (torch.optim.Adam, {}),
    "sgd": (torch.optim.SGD, {}),
}

SHD_REGISTRY = {
    "cosine": get_cosine_lr,
}
