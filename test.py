import random
import gymnasium as gym
from pathlib import Path
import environments
import torch
from torch.utils.tensorboard.writer import SummaryWriter


def crossmat(v: torch.Tensor):
    """左叉积矩阵"""
    v1, v2, v3 = v.unbind(dim=-1)
    _0 = torch.zeros_like(v1)
    return torch.cat([_0, -v3, v2, v3, _0, -v1, -v2, v1, _0])


def main():
    x = torch.randn((1, 1, 2, 3), dtype=torch.float32)
    y = crossmat(x)
    print(y.shape)
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
