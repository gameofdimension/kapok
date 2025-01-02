import os

import torch
import torch.distributed as dist


def is_main_process():
    try:
        if dist.get_rank() == 0:
            return True
        else:
            return False
    except Exception:
        return True


def cleanup():
    dist.destroy_process_group()


def init_distributed():

    # Initializes the distributed backend
    # which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=rank,
        device_id=torch.device(f"cuda:{torch.cuda.current_device()}"),
    )

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    return world_size, rank, local_rank
