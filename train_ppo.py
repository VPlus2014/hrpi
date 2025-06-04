from contextlib import ContextDecorator
from datetime import datetime
import logging
import time
import traceback
from typing import cast
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from environments import NavigationEnv, EvasionEnv
from agents import PPOContinuous
from tools import as_np, as_tsr, init_seed, ConextTimer, set_max_threads
from decimal import getcontext


def main():
    getcontext().prec = 4

    from environments.utils import log_ext

    ROOT_DIR = Path(__file__).parent
    nenvs = 4000
    batch_size = 1000
    buffer_size = 2 * batch_size  # 回放池最大轨迹数
    env_out_torch = False
    env_max_steps = 100
    # env_out_mode = "pytorch" if env_out_torch else "numpy"
    env_device = [
        torch.device("cpu"),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ][-1]
    th_float = torch.float32
    algoname = "ppo"
    envcls = NavigationEnv
    RUNS_DIR = ROOT_DIR / "runs"
    TASK_DIR = RUNS_DIR / "{}_{}_{}".format(
        envcls.__name__,
        algoname,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    WEIGHTS_DIR = TASK_DIR / "weights"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    pretrn_dir = TASK_DIR / "NavigationEnv_ppo_20250604_190053" / "weights"
    logr = log_ext.reset_logger(
        f"{__name__}",
        level=logging.DEBUG,
        file_path=str(TASK_DIR / "main.log"),
    )

    set_max_threads(8)
    init_seed(10086)

    writer = SummaryWriter(TASK_DIR / "tb")
    train_env = envcls(
        agent_step_size_ms=50,
        sim_step_size_ms=50,
        max_sim_ms=50 * env_max_steps,
        waypoints_total_num=10,
        waypoints_visible_num=3,
        position_min_limit=[-5000, -5000, -10000],
        position_max_limit=[5000, 5000, 0],
        render_mode="tacview",
        render_dir=TASK_DIR / "acmi",
        num_envs=nenvs,
        device=env_device,
        dtype=th_float,
        out_torch=env_out_torch,
        logconfig=log_ext.LogConfig(
            f"{envcls.__name__}",
            level=logging.DEBUG,
            file_path=str(TASK_DIR / "env.log"),
        ),
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

    agent = PPOContinuous(
        name="planner",
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        buffer_size=buffer_size,
        learn_batch_size=batch_size,
        mini_batch_size=max(int(batch_size // 8), 1) * 2,
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
        max_steps=env_max_steps,
        logr=log_ext.LogConfig(
            f"{algoname}_agent",
            level=logging.DEBUG,
            file_path=str(TASK_DIR / "agent.log"),
        ).make(),
    )

    if isinstance(pretrn_dir, Path):
        for model, fname in [
            (agent.actor, "actor.pth"),
            (agent.critic, "critic.pth"),
        ]:
            try:
                model = cast(torch.nn.Module, model)
                fin = str(pretrn_dir / fname)
                model.load_state_dict(torch.load(fin))
                logr.info((f"{model.__class__.__name__}<<", fin))
            except Exception as e:
                logr.info(f"Failed to load pretrain {model.__class__.__name__}: {e}")

    tmr_env = ConextTimer("Sim")
    tmr_learn = ConextTimer("Learn")
    tmr_infer = ConextTimer("Infer")
    wallt_sim = 0.0
    wallt_learn = 0.0
    wallt_infer = 0.0
    _tmp_t00 = time.time()

    with tmr_env:
        obs, _ = train_env.reset()

    # train
    max_train_episodes = int(1e6)
    progress_bar = tqdm(total=max_train_episodes, desc="Train")
    global_step = 0
    global_episode = 0
    _save_k0 = 0
    _save_k = 0
    echo_interval = 1.0
    _echo_k0 = 0
    _echo_k = 0
    _learn_k = 0
    _learn_k0 = 1
    while progress_bar.n < max_train_episodes:
        with tmr_learn:
            # 更新智能体
            _learn_k = global_episode // batch_size
            if _learn_k > _learn_k0:
                assert (
                    len(agent.replay_buffer) >= batch_size
                ), "Replay buffer is too small"
                _learn_k0 = _learn_k
                agent.update(progress_bar.n)

            # 存储智能体
            _save_k = progress_bar.n // 1000
            if _save_k > _save_k0:
                _save_k0 = _save_k
                # torch.save(agent.actor, WEIGHTS_DIR / "actor.pt")
                for model, fname in [
                    (agent.actor, "actor.pth"),
                    (agent.critic, "critic.pth"),
                ]:
                    model = cast(torch.nn.Module, model)
                    fo = str(WEIGHTS_DIR / fname)
                    torch.save(model.state_dict(), fo)
                    logr.debug((f"{model.__class__.__name__}>>", fo))

        with tmr_infer:
            # obs_norm = normalize(obs, low, high)
            act, act_log_prob = agent.choose_action(obs)

        with tmr_env:
            obs_next, rew, terminated, truncated, info = train_env.step(act)
            train_env.render()

        with tmr_learn:
            final_obs = info[train_env.KEY_FINAL_OBS]

            agent.post_act(
                obs=obs,
                obs_next=obs_next,
                final_obs=final_obs,
                act=act,
                act_log_prob=act_log_prob,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                global_step=progress_bar.n,
            )

            obs = obs_next
            done = truncated | truncated

        global_step += 1

        _tmp_t1 = time.time()
        _echo_k = int((_tmp_t1 - _tmp_t00) / echo_interval)
        if _echo_k > _echo_k0:
            _echo_k0 = _echo_k
            wallt_sim = tmr_env.t
            wallt_infer = tmr_infer.t
            wallt_learn = tmr_learn.t
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

        ndone = done.sum().item()
        if ndone > 0:
            global_episode += ndone
            progress_bar.update(ndone)


if __name__ == "__main__":
    main()
