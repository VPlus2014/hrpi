"""
Pretrain world model on NavHeadingEnv trajectories.

This script trains only the world model (RSSM + decoder + reward/continue heads),
without actor-critic updates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from codes.envs_np import NavHeadingEnv
from codes.world_model import WorldModel, WorldModelCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain world model with NavHeadingEnv."
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--agent-step-ms", type=int, default=100)
    parser.add_argument("--sim-step-ms", type=int, default=20)
    parser.add_argument("--max-sim-ms", type=int, default=600_000)

    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=100.0)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="runs/world_model_nav")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_env(args: argparse.Namespace) -> NavHeadingEnv:
    return NavHeadingEnv(
        num_envs=args.num_envs,
        agent_step_size_ms=args.agent_step_ms,
        sim_step_size_ms=args.sim_step_ms,
        max_sim_ms=args.max_sim_ms,
    )


def build_world_model(env: NavHeadingEnv, args: argparse.Namespace) -> WorldModel:
    return WorldModel(
        obs_shape=env.single_observation_space.shape,
        action_dim=env.single_action_space.shape[0],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        reward_hidden_dims=(args.hidden_dim, args.hidden_dim),
        use_cnn=False,
    )


def save_checkpoint(
    path: Path,
    step: int,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    env = build_env(args)
    model = build_world_model(env, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    collector = WorldModelCollector(env=env, device=device, normalize_obs=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] device={device}, save_dir={save_dir}")
    print(f"[setup] model params={sum(p.numel() for p in model.parameters()):,}")

    try:
        for step in range(1, args.steps + 1):
            collector.rollout.clear()
            data = collector.collect_rollout(num_steps=args.rollout_steps)

            obs = data["observations"].to(device)  # (T, B, obs_dim)
            action = data["actions"].to(device)  # (T, B, act_dim)
            reward = data["rewards"].to(device).squeeze(-1)  # (T, B)
            done = data["dones_float"].to(device).squeeze(-1)  # (T, B), 1 means done
            continue_flag = 1.0 - done

            model.train()
            loss, metrics = model.compute_loss(
                obs=obs,
                action=action,
                reward=reward,
                continue_flag=continue_flag,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_every == 0 or step == 1:
                print(
                    f"[train] step={step:6d} "
                    f"total={metrics['total_loss']:.4f} "
                    f"obs={metrics['obs_loss']:.4f} "
                    f"rew={metrics['reward_loss']:.4f} "
                    f"con={metrics['continue_loss']:.4f} "
                    f"kl={metrics['kl_loss']:.4f}"
                )

            if step % args.save_every == 0 or step == args.steps:
                ckpt = save_dir / f"world_model_step_{step}.pt"
                save_checkpoint(ckpt, step, model, optimizer)
                print(f"[save] {ckpt}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
