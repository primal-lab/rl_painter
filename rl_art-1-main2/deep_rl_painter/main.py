"""
Main Entry Point for Deep RL Painter

This script:
- Parses command-line arguments
- Updates config dictionary
- Auto-spawns DistributedDataParallel workers (no torchrun needed)
- Delegates full training to `train(config)` from train.py

Usage examples:
  python main.py                               # auto: use up to --num_gpus (default 2)
  python main.py --num_gpus 1                  # force single GPU
  python main.py --gpus "0,2"                  # pick specific GPUs out of 4
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import config
from train import train


# ------------------------ Utils ------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep RL Painter")

    # Painting environment
    parser.add_argument('--target_image_path', type=str, default=config["target_image_path"], help='Path to the target image')
    parser.add_argument('--max_strokes', type=int, default=config["max_strokes"], help='Maximum number of strokes per episode')

    # Model parameters
    parser.add_argument('--model_name', type=str, default=config["model_name"], help='Model architecture')
    parser.add_argument('--action_dim', type=int, default=config["action_dim"], help='Dimension of the action space')
    parser.add_argument('--reward_method', type=str, default=config["reward_method"], help='Reward function')

    # Multi-GPU control
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='How many GPUs to use (auto-picks first N visible GPUs).')
    parser.add_argument('--gpus', type=str, default="",
                        help='Comma-separated GPU IDs to use, e.g. "0,2". If set, overrides --num_gpus.')

    return parser.parse_args()
# -------------------------------------------------------


# -------------------- DDP worker -----------------------
def ddp_worker(local_rank: int, world_size: int, run_config: dict):
    # Required for torch.distributed init in spawn mode
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # Per-process config wiring
    run_config = dict(run_config)  # shallow copy for safety
    run_config["device"] = f"cuda:{local_rank}"
    run_config["local_rank"] = local_rank
    run_config["world_size"] = world_size
    run_config["is_ddp"] = True

    set_seed(run_config["seed"])
    train(run_config)

    dist.destroy_process_group()
# -------------------------------------------------------


def main():
    args = parse_args()

    # Update config with command-line arguments
    config["target_image_path"] = args.target_image_path
    config["max_strokes"]      = args.max_strokes
    config["model_name"]       = args.model_name
    config["action_dim"]       = args.action_dim
    config["reward_method"]    = args.reward_method

    # ----- GPU selection -----
    # If --gpus provided, we set CUDA_VISIBLE_DEVICES so that indices [0..N-1]
    # map to the chosen physical GPUs. Example: --gpus "0,2" will expose two
    # visible devices where local_rank 0 -> physical 0, local_rank 1 -> physical 2.
    if args.gpus.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()

    ngpus_visible = torch.cuda.device_count()
    desired = args.num_gpus if not args.gpus.strip() else len(args.gpus.split(","))
    world_size = min(ngpus_visible, max(1, desired))

    # Single-process fallback (CPU or 1 GPU)
    if world_size <= 1 or not torch.cuda.is_available():
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        run_config = dict(config)
        run_config["device"] = device_str
        run_config["local_rank"] = 0
        run_config["world_size"] = 1
        run_config["is_ddp"] = False

        set_seed(run_config["seed"])
        if run_config.get("train", True):
            print(f"Training mode (single process on {device_str})")
            train(run_config)
        else:
            print("Production mode")
        return

    # Multi-GPU spawn (no torchrun needed)
    print(f"Training mode (DDP spawn) on {world_size} GPU(s); visible GPUs: {ngpus_visible}")
    mp.spawn(ddp_worker, nprocs=world_size, args=(world_size, dict(config)))


if __name__ == "__main__":
    main()
