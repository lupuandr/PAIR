import argparse
import logging
import os
import shutil
import socket
from pathlib import Path

import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def prepare_run(args):
    # Set up logging
    xpid = get_xpid(args)

    # Checkpoint dir is args.checkpoint_dir/$USER
    checkpoint_dir = os.path.join(args.checkpoint_dir, os.environ["USER"])

    run_dir = Path(
        os.path.join(
            checkpoint_dir,
            args.wandb_entity,
            args.wandb_project,
            args.wandb_group,
            xpid,
        )
    )
    os.makedirs(str(run_dir), exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "checkpoint.tar")

    if args.use_wandb:
        wandb.login(host="https://fairwandb.org")
        run = wandb.init(
            config=OmegaConf.to_container(args),
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            notes=socket.gethostname(),
            dir=str(run_dir),
            name=xpid,
            reinit=True,
        )
    else:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run = None
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        run = None

    # Log the config
    job_info = HydraConfig.get()["job"]
    logging.info(f"Job details:\n{OmegaConf.to_yaml(job_info)}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(args)}")
    logging.info(f"Run directory: {run_dir}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return run, run_dir, checkpoint_path


def get_xpid(args):
    xpid = ""
    xpid_keys = args.path_keys.split(":")
    for key in xpid_keys:
        if "." in key:
            k1, k2 = key.split(".")[0], key.split(".")[1]
            val = args[k1][k2]
            key = k1 + k2
        else:
            val = args[key]
        xpid += key + "_" + str(val) + "_"
    return xpid[:-1].replace("experiment_name_", "")


def safe_checkpoint(state_dict, path, index=None, archive_interval=None):
    filename, ext = os.path.splitext(path)
    path_tmp = f"{filename}_tmp{ext}"
    torch.save(state_dict, path_tmp)

    os.replace(path_tmp, path)

    if index is not None and archive_interval is not None and archive_interval > 0:
        if index % archive_interval == 0:
            archive_path = f"{filename}_{index}{ext}"
            shutil.copy(path, archive_path)
