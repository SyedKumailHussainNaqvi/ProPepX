import math
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=0.5,
    last_epoch=-1
):
    """
    Cosine annealing schedule with linear warmup.

    Args:
        optimizer: torch optimizer
        num_warmup_steps: number of warmup steps
        num_training_steps: total training steps
        num_cycles: number of cosine cycles (0.5 = half cosine)
        last_epoch: last epoch index for resume support
    """

    def lr_lambda(current_step):
        # linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)