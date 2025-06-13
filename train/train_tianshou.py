from datetime import datetime
import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def _setup():  # 将项目根节点加入 sys.path
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


ROOT_DIR = _setup()

import logging
import time
import traceback
from typing import Any, cast
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from environments_th import NavigationEnv
from agents import PPOContinuous
from decimal import getcontext
from tools import as_np, as_tsr, init_seed, ConextTimer


from tianshou.highlevel.config import SamplingConfig
from tianshou.highlevel.env import EnvFactoryRegistered, VectorEnvType, EnvPoolFactory
from tianshou.highlevel.experiment import (
    DQNExperimentBuilder,
    ExperimentConfig,
    PPOExperimentBuilder,
)
from tianshou.highlevel.persistence import PolicyPersistence
from tianshou.highlevel.env import EnvMode
from tianshou.highlevel.params.policy_params import DQNParams, PPOParams
from tianshou.highlevel.trainer import (
    EpochTestCallbackDQNSetEps,
    EpochTrainCallbackDQNSetEps,
    EpochStopCallbackRewardThreshold,
)
import tianshou as tianshou
from tianshou.data import CollectStats
from tianshou.utils.space_info import SpaceInfo
from environments_th.utils import log_ext


import gymnasium as gym
import torch
from torch.utils.tensorboard.writer import SummaryWriter


def main() -> None:
    runs_dir = Path(__file__).parent / "tmp" / "tianshou"
    fin_model = runs_dir / "dqn.pth"

    task = "CartPole-v1"
    lr, epoch, batch_size = 1e-3, 10, 64
    train_num, test_num = 10, 100
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 20000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 10000, 10

    # For other loggers, see https://tianshou.readthedocs.io/en/master/tutorials/logger.html

    # You can also try SubprocVectorEnv, which will use parallelization
    train_envs = tianshou.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(train_num)]
    )
    test_envs = tianshou.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(test_num)]
    )
    test_envs.reset()
    tianshou.env.ShmemVectorEnv

    from tianshou.utils.net.common import Net

    # Note: You can easily define other networks.
    # See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
    env = gym.make(task, render_mode="human")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    net = Net(
        state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128]
    )
    if fin_model.exists():
        try:
            net.load_state_dict(torch.load(fin_model))
            print(f"Loaded model from {fin_model}")
        except Exception as e:
            print(f"Failed to load model from {fin_model}: {e}")
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = tianshou.policy.DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=gamma,
        action_space=env.action_space,
        estimation_step=n_step,
        target_update_freq=target_freq,
    )
    train_collector = tianshou.data.Collector(
        policy,
        train_envs,
        tianshou.data.VectorReplayBuffer(buffer_size, train_num),
        exploration_noise=True,
    )
    test_collector = tianshou.data.Collector(
        policy,
        test_envs,
        exploration_noise=True,
    )  # because DQN uses epsilon-greedy method

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    def save_best_fn(policy: tianshou.policy.BasePolicy) -> None:
        fo = runs_dir / "best_policy.pth"
        try:
            torch.save(policy.state_dict(), fo)
            print(f"Saved best policy to {fo}")
        except Exception as e:
            print(f"Failed to save best policy to {fo}: {e}")

    trainer = tianshou.trainer.OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        episode_per_test=test_num,
        batch_size=batch_size,
        update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        save_best_fn=save_best_fn,
        stop_fn=stop_fn,
        logger=tianshou.utils.TensorboardLogger(
            SummaryWriter(runs_dir / "log/dqn")
        ),  # TensorBoard is supported!,
    )
    try:
        result = trainer.run()
        print(f"Finished training in {result.timing.total_time} seconds")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Exception occurred: {e}")
    # watch performance
    policy.set_eps(eps_test)
    collector = tianshou.data.Collector(policy, env, exploration_noise=True)

    try:
        collector.collect(n_episode=100, render=1 / 35)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Exception occurred: {e}")


def main2():
    getcontext().prec = 4
    init_seed(10086)
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    nenvs = 10
    envcls = NavigationEnv
    env_max_steps = 100
    algoname = "ppo"
    max_train_episodes = int(1e6)

    batch_size = 1000
    buffer_size = 2 * (batch_size + nenvs)  # 回放池最大轨迹数
    RUNS_DIR = ROOT_DIR / "runs"
    TASK_DIR = RUNS_DIR / "{}_{}_{}".format(
        envcls.__name__,
        algoname,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    logr = log_ext.LogConfig(
        f"train_{__name__}",
        file_path=str(TASK_DIR / "main.log"),
    ).remake()
    env_logr = log_ext.LogConfig(
        f"{envcls.__name__}",
        level=logging.DEBUG,
        file_path=str(TASK_DIR / "env.log"),
    ).remake()
    env_writer = SummaryWriter(TASK_DIR / "tb_env")

    env_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    agent_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    th_float = torch.float32
    np_float = np.float32

    pretrn_dir = str(TASK_DIR / "NavigationEnv_ppo_20250604_191442" / "weights")

    env_simdt_ms = 50
    rmax = 2000
    NavigationEnv  # @config
    envcfg: dict[str, Any] = dict(
        agent_step_size_ms=500,
        sim_step_size_ms=env_simdt_ms,
        max_sim_ms=env_simdt_ms * env_max_steps,
        waypoints_total_num=10,
        waypoints_visible_num=3,
        waypoints_dR_ratio_min=1e-2,
        waypoints_dR_ratio_max=2e-2,
        position_min_limit=[-rmax, -rmax, -rmax],
        position_max_limit=[rmax, rmax, rmax],
        render_mode="tacview_local",  # tianshou下被屏蔽
        render_dir=str(TASK_DIR / "acmi"),
        num_envs=nenvs,
        device=env_device,
        dtype=th_float,
        np_float=np_float,
        out_torch=False,
        logname=env_logr.name,
        version="2.0",
    )

    envfactory = EnvFactoryRegistered(
        task="Navigation-v1",
        train_seed=0,
        test_seed=0,
        venv_type=VectorEnvType.DUMMY,
        #
        render_mode_train=None,
        render_mode_test=None,
        render_mode_watch="tacview_local",
        #
        **envcfg,
    )
    # for mode in [EnvMode.TRAIN, EnvMode.TEST, EnvMode.WATCH]:
    #     env = envfactory.create_env(mode)
    #     env.close()
    #     print(env.observation_space, env.action_space)
    if not os.path.exists(pretrn_dir):
        pretrn_dir = None
    exprcfg = ExperimentConfig(
        device=agent_device,
        persistence_enabled=False,
        policy_restore_directory=pretrn_dir,
        policy_persistence_mode=PolicyPersistence.Mode.POLICY,  # 策略存储模式 完整/状态字典
        train=True,
        watch=False,
        watch_render=1 / 35,
        watch_num_episodes=100,
        log_file_enabled=True,
        persistence_base_dir=str(TASK_DIR / "log"),
    )
    sampler_config = SamplingConfig(
        num_epochs=10,
        step_per_epoch=500,
        batch_size=500,
        num_train_envs=1,
        num_test_envs=1,
        buffer_size=20000,
        step_per_collect=10,
        update_per_step=1 / 10,
    )

    expriment = (
        PPOExperimentBuilder(
            envfactory,
            exprcfg,
            sampler_config,
        )
        .with_ppo_params(
            PPOParams(
                lr=1e-3,
                discount_factor=0.99,
                gae_lambda=0.95,
                eps_clip=0.2,
                vf_coef=0.5,
                # max_grad_norm=1.0,
            ),
        )
        .with_actor_factory_default(
            hidden_sizes=(128, 128),
        )
        .with_critic_factory_default.with_model_factory_default(hidden_sizes=(64, 64))
        .with_epoch_train_callback(EpochTrainCallbackDQNSetEps(0.3))
        .with_epoch_test_callback(EpochTestCallbackDQNSetEps(0.0))
        .with_epoch_stop_callback(EpochStopCallbackRewardThreshold(195))
        .build()
    )

    expriment.run()


if __name__ == "__main__":
    main()
