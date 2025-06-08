from datetime import datetime
import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
import logging
import time
import traceback
from typing import Any, cast
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from environments import NavigationEnv
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
from environments.utils import log_ext


def main():
    getcontext().prec = 4
    init_seed(10086)
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    nenvs = 10
    envcls = NavigationEnv
    env_max_steps = 100
    algoname = "ppo"
    ROOT_DIR = Path(__file__).parent
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
