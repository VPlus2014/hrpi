from contextlib import ContextDecorator
from datetime import datetime
import time
from typing import cast
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from environments import NavigationEnv, EvasionEnv
from agents import PPOContinuous


def init_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"Seed initialized to {seed}")


def main():
    from environments.utils import log_ext

    ROOT_DIR = Path(__file__).parent
    nenvs = 2000
    env_out_torch = True
    # env_out_mode = "pytorch" if env_out_torch else "numpy"
    env_device = [
        torch.device("cpu"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ][-1]
    th_float = torch.float32
    algoname = "ppo"
    envcls = NavigationEnv
    RUNS_DIR = (
        ROOT_DIR
        / "runs"
        / "{}_{}_{}".format(
            envcls.__name__,
            algoname,
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    )
    WEIGHTS_DIR = RUNS_DIR / "weights"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    init_seed(10086)

    writer = SummaryWriter(RUNS_DIR / "tb")
    train_env = envcls(
        agent_step_size_ms=50,
        sim_step_size_ms=50,
        max_sim_ms=60000,
        waypoints_total_num=10,
        waypoints_visible_num=3,
        position_min_limit=[-5000, -5000, -10000],
        position_max_limit=[5000, 5000, 0],
        render_mode="tacview",
        render_dir=RUNS_DIR / "acmi",
        num_envs=nenvs,
        device=env_device,
        dtype=th_float,
        out_torch=env_out_torch,
    )
    # train_env = EvasionEnv(
    #     agent_step_size_ms=50,
    #     sim_step_size_ms=10,
    #     position_min_limit=[-10000, -10000, -10000],
    #     position_max_limit=[10000, 10000, 0],
    #     writer=writer,
    #     render_mode="tacview",
    #     render_dir=Path.cwd() / "results_2",
    #     num_envs=100,
    #     device=torch.device("cuda:0"),
    #     mode="pytorch",
    # )

    batch_size = 200

    agent = PPOContinuous(
        name="planner",
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        batch_size=batch_size,
        mini_batch_size=100,
        lr_a=5e-4,
        actor_hidden_sizes=[128, 128, 128, 128, 128],
        lr_c=5e-4,
        critic_hidden_sizes=[128, 128, 128, 128, 128],
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        policy_entropy_coef=0.01,
        use_grad_clip=False,
        use_adv_norm=False,
        repeat=10,
        adam_eps=1e-5,
        num_envs=train_env.num_envs,
        writer=writer,
        device=env_device,
        dtype=th_float,
        env_out_torch=env_out_torch,
    )

    obs, _ = train_env.reset()

    wallt_sim = 0.0
    wallt_learn = 0.0
    wallt_infer = 0.0

    # train
    max_train_episodes = int(1e6)
    progress_bar = tqdm(total=max_train_episodes, desc="Train")
    global_step = 0
    pre_udpate_n = progress_bar.n
    _save_k0 = 0
    _save_k = 0
    _tmp_t00 = time.time()
    echo_interval = 1.0
    _echo_k0 = 0
    _echo_k = 0
    while progress_bar.n < max_train_episodes:
        _tmp_t0 = time.time()
        _tmp_t1 = time.time()
        # 更新智能体
        if (
            pre_udpate_n != progress_bar.n
            and global_step % (agent.replay_buffer.max_size) == 0
        ):
            agent.update(progress_bar.n)

        # 存储智能体
        _save_k = progress_bar.n // 1000
        if _save_k > _save_k0:
            _save_k0 = _save_k
            torch.save(agent.actor, WEIGHTS_DIR / "actor.pt")
            torch.save(agent.actor.state_dict(), WEIGHTS_DIR / "actor.pth")

        wallt_learn += time.time() - _tmp_t1

        _tmp_t1 = time.time()

        # obs_norm = normalize(obs, low, high)
        act, act_log_prob = agent.choose_action(obs)

        wallt_infer += time.time() - _tmp_t1

        _tmp_t1 = time.time()

        obs_next, rew, terminated, truncated, _ = train_env.step(act)
        train_env.render()

        wallt_sim += time.time() - _tmp_t1

        _tmp_t1 = time.time()

        agent.post_act(
            obs_next=obs_next,
            act=act,
            act_log_prob=act_log_prob,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            global_step=progress_bar.n,
        )

        obs = obs_next
        done = truncated | truncated

        wallt_learn += time.time() - _tmp_t1

        global_step += 1

        _tmp_t1 = time.time()
        _echo_k = int((_tmp_t1 - _tmp_t00) / echo_interval)
        if _echo_k > _echo_k0:
            _echo_k0 = _echo_k
            tms = np.asarray([wallt_sim, wallt_infer, wallt_learn])
            tms_ratio = tms / np.sum(tms)
            sec_per_batch = (_tmp_t1 - _tmp_t00) / global_step
            ms_per_batch = int(sec_per_batch * 1000)
            progress_bar.set_postfix(
                {
                    "ms/B": f"{ms_per_batch}",
                    "Sim": f"{tms_ratio[0]:.1%}",  # sim
                    "Infer": f"{tms_ratio[1]:.1%}",  # infer
                    "Learn": f"{tms_ratio[2]:.1%}",  # learn
                }
            )

        dn = done.sum().item()
        if dn > 0:
            progress_bar.update(dn)


if __name__ == "__main__":
    main()
