"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random
import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import pefomed.tasks as tasks
from pefomed.common.config import Config
from pefomed.common.dist_utils import get_rank, init_distributed_mode
from pefomed.common.logger import setup_logger
from pefomed.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from pefomed.common.registry import registry
from pefomed.common.utils import now

# imports modules for registration
from pefomed.datasets.builders import *
from pefomed.models import *
from pefomed.processors import *
from pefomed.runners import *
from pefomed.tasks import *

import warnings
# wandb.login()
# 忽略特定的警告
warnings.filterwarnings("ignore", message="huggingface/tokenizers: The current process just got forked,")

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    # parser.add_argument("--cfg-path", required=False, default="pefomed/projects/vim_mamba/vim_mamba.yaml", help="path to configuration file.")
    parser.add_argument("--cfg-path", required=False, default="",help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    print(model)
    runner = get_runner_class(cfg)(
        cfg=cfg, task=task, model=model, datasets=datasets
    )


    runner.train()


if __name__ == "__main__":
    main()
