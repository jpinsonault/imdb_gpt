import argparse
import time

import torch

from .config import TrainConfig
from .experiments import run_all_experiments
from .utils import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Numeric representation benchmark.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--full", action="store_true", help="Use full task/rep suite.")
    parser.add_argument("--only", nargs="*", help="Subset: task, rep, or task__rep__sX.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume-from-status.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds per (task, rep).",
    )
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TrainConfig(n_seeds=max(1, int(args.seeds)))

    set_global_seed(cfg.seed)

    run_all_experiments(
        run_id=run_id,
        full=bool(args.full),
        only=args.only or [],
        resume=bool(args.resume),
        device_str=device,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
