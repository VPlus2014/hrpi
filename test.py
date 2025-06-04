import random
import gymnasium as gym
from pathlib import Path

import numpy as np
import environments

# import torch
from torch.utils.tensorboard.writer import SummaryWriter


def main():
    n = 8
    dimX = 10
    x = np.random.random((n, dimX))
    y = np.random.random((n, dimX))
    done = np.random.randint(0, 2, (n, 1)).astype(bool)
    obs2 = np.where(done, x, y)
    idxs = np.where(done.ravel())[0]
    obs22 = obs2[idxs]
    obs23 = x[idxs]
    print(obs2.shape)
    assert np.equal(obs22,obs23).all()
    return

    x = torch.randint(0, 10, (1, 2), dtype=torch.int64)
    y = torch.randint(1, 10, (1, 2), dtype=torch.int64)
    z = x / y
    print(z)

    from agents.modules.actor import GaussianDActor

    dv = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtp = torch.float32

    dimX = 10
    dimA = 3
    Anvec = [100 for _ in range(dimA)]

    actor = GaussianDActor(
        dimX,
        Anvec,
        action_min=torch.tensor([-2.0 for _ in range(dimA)]),
        action_max=torch.tensor([3.0 for _ in range(dimA)]),
        hidden_sizes=[128, 128],
    ).to(dv, dtp)

    n = random.randint(1, 4) * 2
    x = torch.randn((n, 1, dimX), device=dv, dtype=dtp)
    pix = actor.get_dist(x)
    a = pix.sample()
    print("x: ", x)
    print("a: ", a)
    print("log_prob: ", pix.log_prob(a))
    print("entropy: ", pix.entropy())
    print("nvec: ", pix.nvec)
    return

    writer = SummaryWriter()
    env = gym.make(
        "Evasion-v1",
        agent_step_size_ms=50,
        sim_step_size_ms=50,
        position_min_limit=[-5000, -5000, -10000],
        position_max_limit=[5000, 5000, 0],
        writer=writer,
        render_mode="tacview",
        render_dir=Path.cwd() / "results",
        num_envs=20,
        device=torch.device("cuda"),
    )

    # env.reset()
    print("(IN)state_dim is {}".format(env.observation_space.shape[0]))
    print("(OUT)action_dim is {}".format(env.action_space.shape[0]))
    print("action_min is {}".format(env.action_space.low))
    print("action_max is {}".format(env.action_space.high))
    # env.step()


if __name__ == "__main__":
    main()
