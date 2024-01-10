import os

import torch
import torch.distributed as dist

def print_only_in_master(is_master):
    """
    disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def init_distributed_mode(args, only_master_can_print=True):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print("case 1")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        print("case 2")
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank%torch.cuda.device_count()
    elif hasattr(args, "rank"):
        print("case 3")
        pass
    else:
        print("case 4")
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    
    if only_master_can_print:
        print_only_in_master(args.rank == 0)


if __name__ == '__main__':
    import argparse 
    
    parser = argparse.ArgumentParser(description="PyTorch Classification Training")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    # parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="url used to set up distributed training")
    
    args = parser.parse_args()
    
    rank = get_rank()
    world_size = get_world_size()
    
    # print("* rank: ", rank)
    # print("* world_size: ", world_size)
    print("=====================================================")
    init_distributed_mode(args)
    print(args)
    
