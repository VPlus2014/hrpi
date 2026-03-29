from __future__ import annotations
from datetime import datetime


def _setup():  # 将项目根节点加入 sys.path
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


_DEBUG = True
ROOT_DIR = _setup()

from torch import nn, optim
import logging
import time
import traceback
from typing import Any, Callable, SupportsFloat, cast
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from decimal import getcontext
import util_tools
from util_tools import as_np, as_tsr, init_seed

util_tools.set_use_cuda_dsa(True)

# from tianshou.policy import BasePolicy
from tianshou.data.utils.converter import to_numpy, to_torch, to_torch_as, to_hdf5
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
from tianshou.utils.space_info import SpaceInfo
from tianshou.utils import DummyTqdm
from tianshou.data import Batch

import gymnasium as gym
from gymnasium import Wrapper, spaces
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from codes.agents.replay_buffer.np_trajbuffer import RETrajReplayBuffer
from codes.utils import log_ext
from codes.agents.policy import BaseNNPolicy
from codes.agents.types import (
    TimingStats,
    InfoStats,
    ActBatchProtocol,
    ActStateBatchProtocol,
    ObsBatchProtocol,
    TTrainingStats,
)
from codes.envs_np.proto4venv import NPSyncVecEnv
from codes.envs_np.wrappers import (
    LinspaceActionWrapper,
    FlattenMultiDiscreteActionWrapper,
    ObsNormWrapper,
)
from codes.agents.policy.utils import calc_gae
from codes.utils.time_ext import Timer_Context as ConextTimer, Timer_Pulse

INFOKEY_REWS = "reward_components"


def parse_space_vec_shape(
    obs_space: spaces.Space, use_flatten: bool = False
) -> tuple[int, ...]:
    assert not use_flatten, NotImplementedError("flatten not supported")
    if isinstance(obs_space, spaces.Box):
        obs_shape = obs_space.shape
    elif isinstance(obs_space, spaces.Discrete):
        obs_shape = (1,)
    elif isinstance(obs_space, spaces.MultiDiscrete):
        obs_shape = (len(obs_space.nvec),)
    else:
        raise NotImplementedError(f"Unsupported observation space: {obs_space}")
    return obs_shape


def make_policy(
    device: str,
    dtype: torch.dtype,
    envs: gym.vector.VectorEnv,
    logger: logging.Logger,
    use_multi_reward: bool,
    dimV: int,
    optimizer_maker: Callable[[torch.nn.Module], torch.optim.Optimizer],
    lr_scheduler_maker: Callable[
        [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
    ],
    target_update_freq=50,
    grad_max: float = 0,
) -> BaseNNPolicy:
    # epoch = 10
    # batch_size = 64
    # train_num = 10
    horizon = 100
    gamma = 1 - 1 / horizon
    n_step = 3

    _dimV = dimV if use_multi_reward else 1

    # For other loggers, see https://tianshou.readthedocs.io/en/master/tutorials/logger.html

    # You can also try SubprocVectorEnv, which will use parallelization
    tianshou.env.ShmemVectorEnv

    from codes.agents.policy.utils.net import MLP, MLP_NoState

    # Note: You can easily define other networks.
    # See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
    state_shape = envs.single_observation_space.shape
    assert state_shape is not None, "invalid observation space: None"
    assert len(state_shape) == 1, "only support 1D observation space"

    act_space = envs.single_action_space
    assert isinstance(
        act_space, gym.spaces.Discrete
    ), "only support discrete action space"
    # space_info = SpaceInfo.from_env(envs)

    action_shape = int(act_space.n)
    # assert action_shape is not None, "invalid action space: None"
    # assert len(action_shape) == 1, "only support 1D action space"

    mlp_kern = MLP(
        input_dim=state_shape[0],
        output_dim=_dimV * (1 + action_shape),
        hidden_sizes=[128, 128],
        device=device,
        dtype=dtype,
    )
    net = MLP_NoState(mlp_kern)

    # optimizers
    optr = optimizer_maker(net)
    lr_scheduler = lr_scheduler_maker(optr)

    from codes.agents.policy.mv_dqn import MV_DQNPolicy

    policy = MV_DQNPolicy(
        model=net,
        optim=optr,
        discount_factor=gamma,
        action_space=act_space,
        estimation_step=n_step,
        target_update_freq=target_update_freq,
        lr_scheduler=lr_scheduler,
        observation_space=envs.single_observation_space,
        device=device,
        dtype=dtype,
        use_multi_reward=use_multi_reward,
        values_dim=_dimV,
        clip_loss_grad=True,
        gae_lambda=0.8,
        greedy_eps=0.1,
    )
    policy = policy.to(device=device, dtype=dtype)

    from codes.agents.policy.utils.net import orthogonal_init_

    policy = policy.apply(orthogonal_init_)
    return policy


def make_envs(
    TASK_DIR: Path,
    nenvs: int,
    max_episode_steps: int,
    trainings: bool = True,
    use_multi_reward=True,
    writer: SummaryWriter | None = None,
) -> tuple[gym.vector.VectorEnv, int]:
    from codes.envs_np import NavHeadingEnv as Env_

    env_sim_dt_ms = 50
    env_desc_ms = env_sim_dt_ms * 10
    env_desc_max_steps = max_episode_steps
    rmax = 2000.0

    _mode = "train" if trainings else "test"
    envs = Env_(
        num_envs=nenvs,
        agent_step_size_ms=env_desc_ms,
        sim_step_size_ms=env_sim_dt_ms,
        max_sim_ms=env_desc_ms * env_desc_max_steps,
        waypoints_total_num=1,
        waypoints_visible_num=1,
        xmax=rmax,
        # pos_e_nvec=[100, 100, 100],
        render_mode="tacview_local",
        render_dir=str(TASK_DIR / f"acmi_{_mode}"),
        easy_mode=True,
        debug=False,
        logconfig=log_ext.LogConfig(
            __name__, level=logging.DEBUG, file_path=str(TASK_DIR / f"env_{_mode}.log")
        ),
        writer=writer,
        use_multi_reward=use_multi_reward,
    )

    envs = LinspaceActionWrapper(envs, nvec=[2, 2, 2, 2])
    envs = FlattenMultiDiscreteActionWrapper(envs)
    envs = ObsNormWrapper(envs)

    # env = cast(NPVecEnv, env)
    dimV = len(cast(Env_, envs.unwrapped).reward_fns())
    return envs, dimV


def make_buffer(
    TASK_DIR: Path,
    envs: gym.vector.VectorEnv,
    nenvs: int,
    buffer_size: int,
    max_steps: int,
    observation_space: spaces.Box | spaces.Space,
    action_space: spaces.Discrete | spaces.Box | spaces.MultiDiscrete | spaces.Space,
    training: bool = True,
    float_dtype: type[np.floating] = np.float32,
    int_dtype: type[np.integer] = np.int64,
    aux_infos: dict[str, Any] = {},
):
    _mode = "train" if training else "test"

    assert isinstance(observation_space, spaces.Box), (
        "Only support Box observation space",
        observation_space,
    )
    assert isinstance(
        action_space, (spaces.Discrete, spaces.Box, spaces.MultiDiscrete)
    ), ("Only support Discrete/Box/MultiDiscrete action space", action_space)

    act_dtype = float_dtype if isinstance(action_space, spaces.Box) else int_dtype

    debug = False
    buffer = RETrajReplayBuffer(
        max_trajs=buffer_size,
        max_steps=max_steps,
        size_stack_in=nenvs,
        obs_shape=parse_space_vec_shape(observation_space),
        act_shape=parse_space_vec_shape(action_space),
        logger=log_ext.LogConfig(
            f"{__name__}_{_mode}",
            level=logging.DEBUG if debug else logging.INFO,
            file_path=str(TASK_DIR / "buffer.log"),
        ).remake(),
        obs_dtype=float_dtype,
        act_dtype=act_dtype,
        aux_infos=aux_infos,
        debug=debug,
    )
    return buffer


def _update(
    policy: BaseNNPolicy[TTrainingStats],
    collector: RETrajReplayBuffer,
    sample_size: int,
    step: int,
    writer: SummaryWriter | None,
):
    policy.is_within_training_step = True
    rst = policy.update(sample_size=sample_size, buffer=collector)
    policy.is_within_training_step = False

    if writer is not None:
        policy.write_stats(rst, writer=writer, step=step)
    return rst


def train_one_epoch(
    current_epoch: int,
    envs: gym.vector.VectorEnv,
    policy: BaseNNPolicy[TTrainingStats],
    train_collector: RETrajReplayBuffer,
    step_per_epoch: int,
    update_freq_steps: int,
    writer: SummaryWriter | None,
    sample_size: int,
    training: bool = True,
    device: torch.device | int | str = "cpu",
    dtype: torch.dtype = torch.float32,
    show_progress: bool = True,
    seed: int = 0,
):
    global_step0 = current_epoch * step_per_epoch
    assert update_freq_steps >= 0
    local_step = 0
    k_update = 0
    episodes = 0
    dv = torch.device(device)
    nenvs = envs.num_envs

    tmr_infer = ConextTimer("infer")
    tmr_step = ConextTimer("sim")
    tmr_collect = ConextTimer("collect")
    tmr_update = ConextTimer("update")
    _tmrs = [tmr_infer, tmr_step, tmr_collect, tmr_update]

    tmr_echo = Timer_Pulse("echo")

    if show_progress:
        pbar = tqdm(total=step_per_epoch, desc=f"Epoch {current_epoch+1}")
    else:
        pbar = DummyTqdm(total=step_per_epoch)
    #
    with tmr_collect:
        train_collector.reset()

    with tmr_step:
        obs1_np, info = envs.reset(seed=seed)  # (B, *obs_shape)
    obs1_np = np.repeat(envs.single_observation_space.sample()[None], nenvs, axis=0)
    # done_np = np.zeros(nenvs, dtype=np.bool_)
    # term_np = np.zeros(nenvs, dtype=np.bool_)
    # trunc_np = np.zeros(nenvs, dtype=np.bool_)

    # x_batch.to_torch_(device=dv, dtype=dtype)
    mout: ActBatchProtocol
    with pbar:
        while True:
            with tmr_infer:
                X1_batch4nn = cast(
                    ObsBatchProtocol,
                    Batch(
                        obs=to_torch(obs1_np, device=dv, dtype=dtype),
                        info=[None] * nenvs,  # TODO: add info to policy input?
                    ),
                )
                mout = policy(X1_batch4nn)
                act = mout.act  # (nenvs, *act_shape)
                act_np = as_np(act)

            with tmr_step:
                obs_next_np, rew, term, trunc, info = envs.step(act_np)

            with tmr_collect:
                term = term.reshape(-1, 1)
                trunc = trunc.reshape(-1, 1)
                done = term | trunc
                final_obs = info.get("final_observation", obs_next_np)
                obs2_np = np.where(done, final_obs, obs_next_np)

                rew_comps = info.get(INFOKEY_REWS, None)
                aux = {INFOKEY_REWS: rew_comps}

                train_collector.add(
                    obs=obs1_np,
                    act=act_np,
                    obs_next=obs2_np,
                    rew=rew,
                    term=term,
                    trunc=trunc,
                    # info=[None] * nenvs,  # TODO: add info to buffer
                    **aux,
                )
                if done.any():
                    _ie = np.where(done)[0]
                    train_collector

            obs1_np = obs_next_np

            if (
                training
                and update_freq_steps > 0
                and local_step % update_freq_steps == 0
            ):
                with tmr_update:
                    rst = _update(
                        policy=policy,
                        collector=train_collector,
                        step=global_step0 + local_step,
                        writer=writer,
                        sample_size=sample_size,
                    )
                    k_update += 1

            n_dones = done.sum()
            if n_dones > 0:
                episodes += n_dones

            pbar.update()
            local_step += 1

            if tmr_echo.step() and isinstance(pbar, tqdm):
                _ts = np.asarray([tmr.t for tmr in _tmrs])
                _ts_ratio = _ts / float(np.clip(_ts.sum(), 1e-6, None))
                pbar.set_postfix_str(
                    ",".join(
                        [
                            f"Ep:{episodes}",
                            f"update:{k_update}",
                            "ts:"
                            + ("/".join(["{:.0%}".format(_dr) for _dr in _ts_ratio])),
                        ]
                    )
                )

            # TODO: reset policy with hidden state if anydone
            # anydone = done.any()
            if local_step >= step_per_epoch:
                break
    return rst


def train_run(
    envs: gym.vector.VectorEnv,
    policy: BaseNNPolicy[TTrainingStats],
    train_collector: RETrajReplayBuffer,
    # test_collector: RETrajReplayBuffer | None,
    max_epoch: int,
    step_per_epoch: int,
    update_freq_epoch_step: int,  # unit: epoch step
    # step_per_collect=step_per_collect,
    # episode_per_test: int,
    batch_size: int,  # sample
    # logger=tianshou.utils.TensorboardLogger(
    #     SummaryWriter(runs_dir / "log/dqn")
    # ),  # TensorBoard is supported!,
    ini_epoch: int = 1,
    #
    train_fn: Callable[[int, int], None] | None = None,
    test_fn: Callable[[int, int | None], None] | None = None,
    stop_fn: Callable[[float], bool] | None = None,
    save_best_fn: Callable[[BaseNNPolicy], None] | None = None,
    save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    resume_from_log: bool = False,
    reward_metric: Callable[[np.ndarray], np.ndarray] | None = None,
    training: bool = True,
    writer: SummaryWriter | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    show_progress=True,
) -> InfoStats:
    stats = InfoStats(
        gradient_step=0,
        best_reward=0,
        best_reward_std=0,
        train_step=0,
        train_episode=0,
        test_step=0,
        test_episode=0,
        timing=TimingStats(),
    )
    assert update_freq_epoch_step < step_per_epoch, (
        "expect update_in_steps < step_per_epoch, but got",
        update_freq_epoch_step,
        step_per_epoch,
    )
    tianshou.trainer.BaseTrainer.run
    policy.to(device=device, dtype=dtype)
    for epoch in range(max_epoch):
        rst = train_one_epoch(
            envs=envs,
            current_epoch=epoch,
            policy=policy,
            train_collector=train_collector,
            step_per_epoch=step_per_epoch,
            update_freq_steps=update_freq_epoch_step,
            training=training,
            writer=writer,
            device=device,
            dtype=dtype,
            sample_size=batch_size,
            show_progress=show_progress,
        )
        stats.timing.train_time += rst.train_time
        if save_best_fn is not None:
            save_best_fn(policy)
    return stats


def main() -> None:

    getcontext().prec = 4
    seed = int(time.time()) % (2 << 31)
    init_seed(seed)
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    nenvs = 4000

    max_episode_steps = 100
    algoname = "dqn"
    max_epochs = 100
    use_multi_reward = True
    train_episodes_per_env = 20
    assert train_episodes_per_env > 0
    agent_sync_freq = 100  # unit:更新步数
    agent_grad_max = 1.0  # 一次更新的梯度上限

    batch_size = max(1, nenvs // 2)
    # assert batch_size <= train_episodes_per_epoch
    buffer_size = 2 * max(batch_size, nenvs)  # 回放池最大轨迹数
    RUNS_DIR = ROOT_DIR / "runs"

    lr = 1e-3
    update_freq_epoch_step = 2
    step_per_epoch = train_episodes_per_env * max_episode_steps

    from codes.envs_np.nav_heading import NavHeadingEnv as _Env

    TASK_DIR = RUNS_DIR / "{}_{}_{}".format(
        _Env.__name__,
        algoname,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    logr = log_ext.LogConfig(
        __name__ + "_main", level=logging.DEBUG, file_path=str(TASK_DIR / "main.log")
    ).remake()

    agent_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    th_float = torch.float32
    # assert th_float == torch.float32, "SB tianshou MLP only support float32"
    np_float = np.float64

    pretrn_dir = str(RUNS_DIR / "NavHeadingEnv_dqn_20250625_233142" / "policy_41.pth")

    sw = SummaryWriter(str(TASK_DIR / "tb_train"))

    envs, dimV = make_envs(
        TASK_DIR=TASK_DIR,
        nenvs=nenvs,
        max_episode_steps=max_episode_steps,
        trainings=True,
        use_multi_reward=use_multi_reward,
        writer=sw,
    )

    def optm_maker(net: nn.Module):
        return optim.AdamW(
            net.parameters(),
            lr=lr,
            # momentum=0.1,
            weight_decay=1e-6,
        )

    def lr_scheduler_maker(optimizer: optim.Optimizer):
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=1
        )

    policy = make_policy(
        device=agent_device,
        dtype=th_float,
        envs=envs,
        logger=logr,
        use_multi_reward=use_multi_reward,
        dimV=dimV,
        optimizer_maker=optm_maker,
        lr_scheduler_maker=lr_scheduler_maker,
        target_update_freq=agent_sync_freq,
        grad_max=agent_grad_max,
    )
    if Path(pretrn_dir).exists():
        try:
            sd = torch.load(pretrn_dir, map_location=agent_device)
            policy.load_state_dict(sd)
            logr.info(f"policy<<{pretrn_dir}")
        except Exception as e:
            logr.info(f"Failed to load model: {e}")

    from codes.envs_np import NavHeadingEnv as Env_

    envs_core = cast(Env_, envs.unwrapped)
    aux_infos = {
        INFOKEY_REWS: (
            len(envs_core.reward_fns()),
            np_float,
            0.0,
        ),
    }

    buffer = make_buffer(
        TASK_DIR=TASK_DIR,
        envs=envs,
        nenvs=nenvs,
        buffer_size=buffer_size,
        max_steps=max_episode_steps,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        training=True,
        float_dtype=np_float,
        int_dtype=np.int64,
        aux_infos=aux_infos,
    )

    # def stop_fn(mean_rewards: float) -> bool:
    #     if env.spec:
    #         if not env.spec.reward_threshold:
    #             return False
    #         else:
    #             return mean_rewards >= env.spec.reward_threshold
    #     return False

    model_version = 0

    def save_best_fn(policy: BaseNNPolicy) -> None:
        nonlocal model_version
        model_version += 1
        fo = TASK_DIR / "weights" / f"policy_{model_version}.pth"
        fo.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(policy.state_dict(), fo)
            logr.info(f"policy>>{fo}")
        except Exception as e:
            logr.info(f"Failed to save policy: {e}\n{traceback.format_exc()}")

    # trainer = tianshou.trainer.OffpolicyTrainer(
    #     policy=policy,
    #     train_collector=collector,
    #     test_collector=test_collector,
    #     max_epoch=epoch,
    #     step_per_epoch=step_per_epoch,
    #     step_per_collect=step_per_collect,
    #     episode_per_test=test_num,
    #     batch_size=batch_size,
    #     update_per_step=1 / step_per_collect,
    #     train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    #     test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    #     save_best_fn=save_best_fn,
    #     stop_fn=stop_fn,
    #     logger=tianshou.utils.TensorboardLogger(
    #         SummaryWriter(runs_dir / "log/dqn")
    #     ),  # TensorBoard is supported!,
    # )
    try:
        # trainer.run()
        result = train_run(
            envs=envs,
            save_best_fn=save_best_fn,
            policy=policy,
            train_collector=buffer,
            max_epoch=max_epochs,
            ini_epoch=1,
            step_per_epoch=step_per_epoch,
            update_freq_epoch_step=update_freq_epoch_step,
            # episode_per_test=10,
            batch_size=batch_size,
            training=True,
            device=agent_device,
            dtype=th_float,
            writer=sw,
        )
        logr.info(f"Finished training in {result.timing.total_time} seconds")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logr.info(f"Exception occurred on train: {e}\n{traceback.format_exc()}")
        if _DEBUG:
            raise e
    # watch performance

    # try:
    #     eps_test = 0.05
    #     policy.set_eps(eps_test)
    #     collector = tianshou.data.Collector(policy, envs, exploration_noise=True)
    #     collector.collect(n_episode=100, render=1 / 35)
    # except KeyboardInterrupt:
    #     pass
    # except Exception as e:
    #     logr.info(f"Exception occurred on test: {e}")
    sw.close()


if __name__ == "__main__":
    main()
