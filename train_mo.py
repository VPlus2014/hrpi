# train MORL PPO agent
from __future__ import annotations

import os
import sys
import time
import torch
import numpy as np
import gymnasium as gym
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple
from collections import deque
from torch.utils.tensorboard.writer import SummaryWriter

from ppo import PPO
from morl import MORL
from util_tools import init_seed
from utils import ReplayBuffer, make_env, make_vec_envs


def get_args():
    parser = argparse.ArgumentParser(description="Train MORL PPO agent")
    parser.add_argument(
        "--env-name",
        default="gymnasium-v0",
        help="environment to train on (default: gymnasium-v0)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="N", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        metavar="N",
        help="how many training CPU processes to use (default: 16)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        metavar="N",
        help="number of forward steps in PPO (default: 2048)",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=10,
        metavar="N",
        help="number of updates to perform in PPO (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        metavar="GAE",
        help="lambda parameter for GAE (default: 0.95)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        metavar="E",
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        metavar="V",
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        metavar="M",
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        metavar="E",
        help="clipping epsilon for PPO (default: 0.2)",
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=32,
        metavar="N",
        help="number of mini-batches (default: 32)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="log interval, one log per n updates (default: 1)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        metavar="N",
        help="save interval, one save per n updates (default: 100)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        metavar="N",
        help="eval interval, one eval per n updates (default: None)",
    )
    parser.add_argument(
        "--num-env-steps",
        type=int,
        default=10e6,
        metavar="N",
        help="number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--env-base-dir",
        default="./env",
        help="directory to save environment data (default: ./env)",
    )
    parser.add_argument(
        "--log-dir",
        default="./logs",
        help="directory to save agent logs (default: ./logs)",
    )
    parser.add_argument(
        "--save-dir",
        default="./trained_models",
        help="directory to save agent checkpoints (default: ./trained_models)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--use-morl", action="store_true", default=False, help="use MORL algorithm"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.env_base_dir, exist_ok=True)

    writer = SummaryWriter(args.log_dir)

    init_seed(args.seed)

    env = make_env(args.env_name, args.seed, args.env_base_dir)
    env.seed(args.seed)


if __name__ == "__main__":
    main()
