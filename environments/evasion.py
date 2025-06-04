from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter
    from typing import Any, Sequence
import datetime
import torch
from typing import Literal
import numpy as np
import gymnasium
from gymnasium import spaces
from collections import OrderedDict
from pathlib import Path

from .models.aircraft import BaseAircraft, PointMassAircraft
from .models.missile import BaseMissile, PointMassMissile
from .utils.space import space2box, flatten, unflatten
from .utils.math import (
    quat_enu_ned,
    quat_rotate,
    quat_rotate_inv,
    euler_from_quat,
    Qx,
    quat_mul,
    ned2aer,
    vec_cosine,
)
from environments.utils.tacview_render import ObjectState, AircraftAttr, MissileAttr
from environments.reward_fns import (
    TimeRewardFn,
    LowAltitudeRewardFn,
    LowAirSpeedRewardFn,
)
from environments.reward_fns.evasion import *
from environments.termination_fns import *
from environments.termination_fns.evasion import *


class EvasionEnv(gymnasium.Env):
    metadata: dict[str, Any] = {"render_modes": ["tacview"], "render_fps": 25}

    def __init__(
        self,
        agent_step_size_ms: int,  # 决策步长
        sim_step_size_ms: int,  # 仿真步长
        position_min_limit: list[int],
        position_max_limit: list[int],
        writer: SummaryWriter,
        render_mode: str | None = None,
        render_dir: Path | None = None,
        num_envs: int = 1,
        device=torch.device("cpu"),
        mode: Literal["numpy", "pytorch"] = "numpy",
    ):
        np_float = np.float32
        super().__init__()
        assert (
            agent_step_size_ms > 0 and sim_step_size_ms > 0
        ), "仿真步长及决策步长必须大于0"
        assert (
            agent_step_size_ms // sim_step_size_ms > 0
            and agent_step_size_ms % sim_step_size_ms == 0
        ), "决策步长必须为仿真步长的整数倍"
        self.agent_step_size_ms = agent_step_size_ms
        self.sim_step_size_ms = sim_step_size_ms
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        assert mode in ["numpy", "pytorch"], "mode must be 'numpy' or 'pytorch'"
        self._out_as_np = mode == "numpy"
        self.position_min_limit = torch.tensor(position_min_limit, dtype=torch.int64)
        self.position_max_limit = torch.tensor(position_max_limit, dtype=torch.int64)
        self.writer = writer
        self.render_mode = render_mode
        if self.render_mode:
            assert render_dir is not None
            if not render_dir.exists():
                render_dir.mkdir(parents=True)
            self.render_dir = render_dir

        # 创建战斗机模型(逃逸者)
        self.aircraft = PointMassAircraft(
            acmi_name="J-20",
            acmi_color="Red",
            call_sign="agent1",
            position_e=torch.cat(
                [
                    torch.zeros(size=(self.num_envs, 2)),
                    -6000 * torch.ones(size=(self.num_envs, 1)),
                ],
                dim=-1,
            ),
            tas=600 * torch.ones(size=(self.num_envs, 1)),
            sim_step_size_ms=sim_step_size_ms,
            device=self.device,
        )

        # 创建导弹模型（追击者）
        random_vals = torch.rand(self.num_envs, 3)
        positions = self.position_min_limit + random_vals * (
            self.position_max_limit - self.position_min_limit
        )
        self.missile = PointMassMissile(
            call_sign="missile",
            acmi_color="Blue",
            position_g=positions,
            target=self.aircraft,
            sim_step_size_ms=sim_step_size_ms,
            device=self.device,
        )

        # =====define observation space=====#
        self.observation_space_dict = spaces.Dict()
        self.observation_space_dict["aircraft_position_g"] = spaces.Box(
            low=self.position_min_limit.numpy(),
            high=self.position_max_limit.numpy(),
            shape=(3,),
            dtype=np_float,
        )
        self.observation_space_dict["aircraft_velocity_g"] = spaces.Box(
            low=-1500, high=1500, shape=(3,), dtype=np_float
        )
        self.observation_space_dict["aircraft_tas"] = spaces.Box(
            low=0, high=2000, shape=(1,), dtype=np_float
        )
        self.observation_space_dict["aircraft_chi"] = spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np_float
        )
        self.observation_space_dict["aircraft_gamma"] = spaces.Box(
            low=-np.pi / 2, high=np.pi / 2, shape=(1,), dtype=np_float
        )
        self.observation_space_dict["aircraft_mu"] = spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np_float
        )
        self.observation_space_dict["aircraft_alpha"] = spaces.Box(
            low=-np.pi / 2, high=np.pi / 2, shape=(1,), dtype=np_float
        )

        self.observation_space_dict["missile_position_g"] = spaces.Box(
            low=self.position_min_limit.numpy(),
            high=self.position_max_limit.numpy(),
            shape=(3,),
            dtype=np_float,
        )
        self.observation_space_dict["missile_velocity_e"] = spaces.Box(
            low=-1500, high=1500, shape=(3,), dtype=np_float
        )
        self.observation_space = space2box(self.observation_space_dict, dtype=np_float)

        # =======define action space=======#
        self.action_space_dict = spaces.Dict()
        self.action_space_dict["thrust_cmd"] = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np_float
        )
        self.action_space_dict["alpha_cmd"] = spaces.Box(
            low=-np.pi / 4, high=np.pi / 4, shape=(1,), dtype=np_float
        )
        self.action_space_dict["mu_cmd"] = spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np_float
        )
        self.action_space = space2box(self.action_space_dict, dtype=np_float)

        # =====define reward functions=====#
        self._reward_fns: list[BaseRewardFn] = [
            # ApproachNavigationPointRewardFn(1e-4),
            # TimeRewardFn(1),
            AircraftShotdownRewardFn(100),
            AircraftSurvivalRewardFn(100),
            # LowAltitudeRewardFn(100),
            # LowAirSpeedRewardFn(100)
        ]

        # ===define termination functions===#
        self._termination_fns: list[BaseTerminationFn] = [
            LowAltitudeTerminationFn(min_altitude_m=100),
            LowAirSpeedTerminationFn(min_airspeed_mps=10),
            AircraftShotdownTerminationFn(),
            AircraftSurvivalTerminationFn(),
        ]

        # =======simulation variables=======#
        self.__sim_time_ms = torch.zeros(size=(self.num_envs, 1), device=self.device)

        # ==============render==============#
        self.__render_interval_ms = round((1000 / self.metadata["render_fps"]))
        self.__objects_states: list[ObjectState] = []
        self.__render_timestamp_ms = -float("inf")
        self.__render_count: int = 0

    @property
    def sim_time_ms(self) -> torch.Tensor:
        return self.__sim_time_ms

    @property
    def sim_time_s(self) -> torch.Tensor:
        return 0.001 * self.__sim_time_ms

    def _cast_out(self, data: torch.Tensor) -> torch.Tensor | np.ndarray:
        if self._out_as_np:
            data = data.cpu().numpy()
        return data


    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)

        # reset aircraft model
        pln = self.aircraft
        pln.reset(env_indices)
        pln.activate(env_indices)

        # reset missile model
        mis = self.missile
        mis.reset(env_indices)
        random_vals = torch.rand(len(env_indices), 3)
        positions = self.position_min_limit + random_vals * (
            self.position_max_limit - self.position_min_limit
        )
        mis.position_e[env_indices] = positions.to(device=self.device)

        aer = ned2aer(mis.target.position_e[env_indices] - mis.position_e[env_indices])
        mis.set_yaw(env_indices, aer[..., 0:1])
        mis.set_pitch(env_indices, aer[..., 1:2])

        mis.launch(env_indices)

        # ====reset simulation variables====#
        self.__sim_time_ms[env_indices] = 0.0

        # =====reset reward functions=====#
        for reward_fn in self._reward_fns:
            reward_fn.reset(self, env_indices)

        # ===reset termination functions===#
        # for termination_fn in self.__termination_fns:
        #     termination_fn.reset(self)

        if 0 in env_indices:
            # ==========reset render==========#
            if self.render_mode == "tacview":
                if self.__render_timestamp_ms > 0:
                    self.__objects_states.sort(key=lambda x: x.sim_time_s)
                    self.__render()

                file_path = self.render_dir / "{:07d}.acmi".format(
                    self.__render_count + 1
                )
                with open(file_path, "a") as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.2\n")
                    f.write(
                        "0,ReferenceTime={}Z\n".format(
                            datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                        )
                    )
            self.__objects_states.clear()
            self.__render_timestamp_ms = -float("inf")
            self.__render_count += 1

        obs_dict = self.__get_obs(env_indices)
        info = {}
        obs = flatten(self.observation_space_dict, obs_dict)

        return self._cast_out(obs), info

    def step(self, action: torch.Tensor, global_step: int):
        step_num = self.agent_step_size_ms // self.sim_step_size_ms

        for i in range(step_num):
            # record
            if self.render_mode is not None:
                if (
                    self.sim_time_ms[0].item() - self.__render_timestamp_ms
                    > self.__render_interval_ms
                ):
                    self.__render_timestamp_ms = self.sim_time_ms[0].item()
                    self.__record()

            # aircraft simulation step
            self.aircraft.run(action)

            # missile simulation step
            n_cmd = self._png(
                target_position_e=self.aircraft.position_e,
                target_velocity_e=self.aircraft.velocity_e,
            )
            self.missile.run(n_cmd)

            self.__sim_time_ms += self.sim_step_size_ms
        self.missile.try_miss()

        rew = self.__get_rew()
        truncated = torch.zeros(
            size=(self.num_envs, 1), dtype=torch.bool, device=self.device
        )
        terminated = self.__is_terminated()
        info = {}

        done = torch.logical_or(terminated, truncated)
        if done[0]:
            self.__record()

        if torch.any(done):
            indices = torch.where(done)[0]

            missed = self.missile.is_missed(indices)
            survival_rate = missed.sum().item() / len(indices)
            self.writer.add_scalar("survival_rate", survival_rate, global_step)

            miss_distance = self.missile.miss_distance[indices]
            self.writer.add_scalar(
                "miss_distance/mean", miss_distance.mean().item(), global_step
            )
            if len(indices) > 1:
                self.writer.add_scalar(
                    "miss_distance/std", miss_distance.std().item(), global_step
                )

            _obs_dict, _info = self.reset(indices)

        obs_dict = self.__get_obs()
        obs = flatten(self.observation_space_dict, obs_dict)

        return (
            self._cast_out(obs),
            self._cast_out(rew),
            self._cast_out(terminated),
            self._cast_out(truncated),
            info,
        )

    def render(self):
        if self.render_mode is not None:
            self.__objects_states.sort(key=lambda x: x.sim_time_s)
            self.__render()
            self.__objects_states.clear()

    def _png(
        self, target_position_e: torch.Tensor, target_velocity_e: torch.Tensor
    ) -> torch.Tensor:
        dp = target_position_e - self.missile.position_e  # LOS (...,3)
        dv = target_velocity_e - self.missile.velocity_e  # relative velocity (...,3)
        dem = dp.square().sum(dim=-1, keepdim=True).clip(min=1e-6)  # distance^2 (...,1)
        omega = torch.cross(dp, dv, dim=-1) / dem
        ad_g = self.missile._N * torch.cross(
            dv, omega, dim=-1
        )  # required acceleration in earth frame (...,3)
        a_f = quat_rotate(self.missile.Q_ew(), ad_g)

        return torch.clip(
            a_f[..., 1:] / self.missile._g,
            min=-self.missile._nyz_max,
            max=self.missile._nyz_max,
        )

    def __record(self):
        # aircraft
        euler = euler_from_quat(self.aircraft.q_kg[0])
        euler[0] = self.aircraft.mu[0]

        aircraft_state = ObjectState(
            sim_time_s=self.sim_time_s[0].item(),
            name=self.aircraft.uid,
            attr=AircraftAttr(
                Color=self.aircraft.acmi_color,
                TAS=self.aircraft.tas[0].item(),
            ),
            pos_ned=self.aircraft.position_e[0].clone().cpu(),
            rpy_rad=euler.cpu(),
        )
        self.__objects_states.append(aircraft_state)

        # missile
        mis = self.missile
        rpy_rad = mis.rpy_eb()

        missile_state = ObjectState(
            sim_time_s=self.sim_time_s[0].item(),
            name=mis.uid,
            attr=MissileAttr(
                Color=mis.acmi_color,
                TAS=mis.tas[0].item(),
            ),
            pos_ned=mis.position_e[0].clone().cpu(),
            rpy_rad=euler.cpu(),
        )
        self.__objects_states.append(missile_state)

    def __render(self):
        if self.render_mode == "tacview":
            file_path = self.render_dir / "{:07d}.acmi".format(self.__render_count)
            with open(file_path, "a") as f:
                sim_time_s = -1.0
                for object_state in self.__objects_states:
                    if object_state.sim_time_s > sim_time_s:
                        sim_time_s = object_state.sim_time_s
                        f.write("#{:.2f}\n".format(sim_time_s))

                    # write object position
                    f.write(
                        "{},T={}".format(
                            object_state.id,
                            "|".join(["{:7f}".format(v) for v in object_state.pos_lbh]),
                        )
                    )

                    # write object attitude
                    if object_state.rpy_deg:
                        f.write(
                            "|".join(
                                [""] + ["{:2f}".format(v) for v in object_state.rpy_deg]
                            )
                        )

                    # write object attribute
                    attr_dict = object_state.attr.model_dump(exclude_none=True)
                    f.write(",".join([""] + [f"{k}={v}" for k, v in attr_dict.items()]))
                    f.write("\n")

                    # write events
                    pass
        else:
            raise NotImplementedError

    def __get_obs(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> OrderedDict:
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_indices, Sequence):
            # check indices
            index_max = max(env_indices)
            index_min = min(env_indices)
            assert index_max < self.num_envs, index_min >= 0
            env_indices = torch.tensor(env_indices, device=self.device)

        elif isinstance(env_indices, torch.Tensor):
            # check indices
            assert len(env_indices.shape) == 1
            env_indices = env_indices.to(device=self.device)
            index_max = env_indices.max().item()
            index_min = env_indices.min().item()
            assert index_max < self.num_envs, index_min >= 0

        obs_dict = OrderedDict()
        obs_dict["aircraft_position_g"] = self.aircraft.position_e[
            env_indices
        ]  # 局部地轴系坐标
        obs_dict["aircraft_velocity_g"] = self.aircraft.velocity_e[
            env_indices
        ]  # 局部地轴系速度
        obs_dict["aircraft_tas"] = self.aircraft.tas[env_indices]  # 真空速
        euler = euler_from_quat(self.aircraft.q_kg)
        chi = euler[..., 2:3]
        gamma = euler[..., 1:2]
        obs_dict["aircraft_chi"] = chi[env_indices]
        obs_dict["aircraft_gamma"] = gamma[env_indices]
        obs_dict["aircraft_mu"] = self.aircraft.mu[env_indices]
        obs_dict["aircraft_alpha"] = self.aircraft.alpha[env_indices]
        obs_dict["missile_position_g"] = self.missile.position_e[env_indices]
        obs_dict["missile_velocity_e"] = self.missile.velocity_e[env_indices]

        return obs_dict

    def __get_rew(self) -> torch.Tensor:
        reward = torch.zeros(size=(self.num_envs, 1), device=self.device)
        for reward_fn in self._reward_fns:
            _reward = reward_fn(self)
            reward += _reward
        return reward

    def __is_terminated(self) -> torch.Tensor:
        terminated = torch.zeros(
            size=(self.num_envs, 1), dtype=torch.bool, device=self.device
        )
        for termination_fn in self._termination_fns:
            _terminated = termination_fn(self)
            terminated = torch.logical_or(terminated, _terminated)
        return terminated
