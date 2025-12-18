import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_min_lr_and_warmup_lambda(
    current_step,
    *,
    num_warmup_steps,
    num_training_steps,
    min_lr,
    num_cycles,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(min_lr, min_lr + 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * (2.0) * progress)) * (1.0 - min_lr))


def cosine_schedule_with_min_lr_and_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr=0.1,
    num_cycles=0.5,
    last_epoch=-1,
):
    lr_lambda = partial(
        _get_cosine_schedule_with_min_lr_and_warmup_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_lr(
    opt,
    num_warmup_steps,
    num_training_steps,
    min_lr=0.1,
):
    return cosine_schedule_with_min_lr_and_warmup(
        opt,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr,
    )
