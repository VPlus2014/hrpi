import time
import torch
import gymnasium as gym
from tqdm import tqdm


def main():
    use_cuda = False
    dv = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    dtp = torch.float32
    total_frames = 100000
    nenvs = 20000
    env_mode = "pytorch"
    max_steps = total_frames // nenvs
    render_mode = ["tacview", None][-1]

    from environments.navigation import NavigationEnv

    envcls = NavigationEnv
    env = envcls(
        agent_step_size_ms=50,
        sim_step_size_ms=50,
        navigation_points_total_num=10,
        navigation_points_visible_num=1,
        position_min_limit=[-5000, -5000, -10000],
        position_max_limit=[5000, 5000, 0],
        render_mode=render_mode,
        num_envs=nenvs,
        device=dv,
        mode=env_mode,
    )
    _t0 = time.time()
    env.reset()
    for i in tqdm(range(max_steps)):
        action = env.action_space.sample()
        action = torch.asarray(action, device=dv, dtype=dtp)
        obs, rew, term, trunc, info = env.step(action)

    dt = time.time() - _t0
    print(f"Time elapsed: {dt:.2f}s, FPS: {nenvs*max_steps/dt:.2f}")


if __name__ == "__main__":
    main()
