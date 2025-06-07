import itertools
import random
from timeit import timeit
import gymnasium as gym
from pathlib import Path
import numpy as np

# import torch
# from torch.utils.tensorboard.writer import SummaryWriter
import torch
from numpy.typing import NDArray

_reshape = np.reshape
_where = np.where
_clip = np.clip
_norm = np.linalg.norm
# NDArray = torch.Tensor
# _reshape = torch.reshape
# _where = torch.where
# _clip = torch.clamp
# _norm = torch.norm


def main():
    n = 2
    dimX = 3
    x = np.random.randn(n, dimX)
    for i in range(3):
        y = np.expand_dims(x,-i)
        print(x.shape, y.shape,i)

    p_np = x
    n,e,d = np.split(p_np, 3, axis=-1)
    print(n.shape, e.shape, d.shape)
    print(x)
    print(np.concatenate([n, e, d], axis=-1))
    return

    def data_gen():
        candet = torch.randint(0, 2, (n, m, 1), dtype=torch.bool)
        cantrk = torch.randint(0, 2, (n, m, 1), dtype=torch.bool)
        anytrk = cantrk.any(dim=-2, keepdim=True)
        x = torch.randn((n, m, dimX))
        return x, candet, cantrk, anytrk

    def tf1():
        x, candet, cantrk, anytrk = data_gen()
        w1 = anytrk & cantrk
        w2 = (~anytrk) & candet
        sig = w1 | w2
        y = sig * x

    def tf2():
        x, candet, cantrk, anytrk = data_gen()
        sig = torch.where(anytrk, cantrk, candet)
        y = sig * x

    def tf3():
        x, candet, cantrk, anytrk = data_gen()
        c1f = cantrk.to(torch.float32)
        c11f = anytrk.to(torch.float32)
        c2f = candet.to(torch.float32)
        w1 = c11f * c1f
        w2 = (1 - c11f) * c2f
        sig = w1 + w2
        y = sig * x

    for tf in [tf1, tf2, tf3]:
        dt = timeit(tf, number=100)
        print(f"{tf.__name__}: {dt:.5f}s")

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
