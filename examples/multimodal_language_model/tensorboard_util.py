import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(self):
        self.writer = SummaryWriter()

    def write_summary(
        self,
        num_iteration: int,
        iteration_time: float,
        loss: float,
        learning_rate: float,
        num_patches,
    ):
        self.writer.add_scalar("Loss", loss, num_iteration)
        self.writer.add_scalar("Learning Rate", learning_rate, num_iteration)
        self.writer.add_scalar("Iteration Time", iteration_time, num_iteration)
        self.writer.add_scalars(
            "Memory",
            {
                "Max": torch.cuda.max_memory_allocated() / (1000**2),
                "Current": torch.cuda.memory_allocated() / (1000**2),
            },
            num_iteration,
        )
        torch.cuda.reset_max_memory_allocated()
        self.writer.add_scalar("Num Patches", num_patches, num_iteration)
