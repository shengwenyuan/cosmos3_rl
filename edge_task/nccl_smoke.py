"""Minimal single-node NCCL all-reduce smoke for the Cosmos training image."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=120),
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    torch.testing.assert_close(value.cpu(), torch.tensor(expected))

    dist.barrier()
    if rank == 0:
        print(
            f"PASS nccl all-reduce world_size={world_size} "
            f"sum={value.item():.0f} torch={torch.__version__} cuda={torch.version.cuda}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
