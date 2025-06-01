import datetime
import torch
from typing import Literal, cast
from functools import cached_property
import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Any, Sequence
from collections import OrderedDict
from pathlib import Path
from .models.aircraft import BaseAircraft, PointMassAircraft
from .utils.space import space2box, flatten, unflatten
from .utils.math import (
    quat_enu_ned,
    quat_rotate,
    quat_rotate_inv,
    euler_from_quat,
    Qx,
    quat_mul,
    ned2aer,
)
from .utils.tacview_render import ObjectState, AircraftAttr, WaypointAttr
from .reword_fns import *
from .termination_fns import *
from .proto4venv import TrueVecEnv


class NavigationEnv(TrueVecEnv):
    metadata: dict[str, Any] = {"render_modes": ["tacview"], "render_fps": 25}
    name = "NavigationEnv"

    def __init__(
        self,
        agent_step_size_ms: int,  # 决策步长(ms)
        sim_step_size_ms: int,  # 仿真步长(ms)
        navigation_points_total_num: int,  # 总导航点数量
        navigation_points_visible_num: int,  # 每一时刻对飞机可见的导航点数量
        position_min_limit: list[int],  # (x_\min,y_\min,z_\min)
        position_max_limit: list[int],  # (x_\max,y_\max,z_\max)
        render_mode: str | None = None,
        render_dir: Path | None = None,
        num_envs: int = 1,
        device=torch.device("cpu"),
        dtype=torch.float,
        np_float=np.float32,
        mode: Literal[
            "numpy", "pytorch"
        ] = "numpy",  # step输出obs格式, "numpy"->ndarray, "pytorch"->torch.Tensor
    ):
        super().__init__(num_envs=num_envs, device=device, dtype=dtype)
        assert (
            agent_step_size_ms > 0 and sim_step_size_ms > 0
        ), "仿真步长及决策步长必须大于0"
        assert (
            agent_step_size_ms // sim_step_size_ms > 0
            and agent_step_size_ms % sim_step_size_ms == 0
        ), "决策步长必须为仿真步长的整数倍"
        self.agent_step_size_ms = agent_step_size_ms
        self.sim_step_size_ms = sim_step_size_ms
        self.np_dtype = np_float
        self.mode = mode
        assert mode in ["numpy", "pytorch"], "mode must be 'numpy' or 'pytorch'"
        self._out_as_np = mode == "numpy"
        self.navigation_points_total_num = navigation_points_total_num
        self.navigation_points_visible_num = navigation_points_visible_num
        assert (
            navigation_points_visible_num >= 1
        ), "expect navigation_points_visible_num >= 1"
        self.position_min_limit = torch.tensor(
            position_min_limit,
            dtype=torch.int,
        ).ravel()  # (3,)
        self.position_max_limit = torch.tensor(
            position_max_limit,
            dtype=torch.int,
        ).ravel()  # (3,)

        self.render_mode = render_mode
        if self.render_mode:
            assert render_dir is not None
            if not render_dir.exists():
                render_dir.mkdir(parents=True)
            self.render_dir = render_dir

            # 计算渲染帧间隔
            self._render_interval_ms = round((1000 / self.metadata["render_fps"]))

        # 创建战斗机模型(逃逸者)
        self.aircraft = PointMassAircraft(
            model_name="agent",
            model_color="Red",
            position_e=torch.cat(
                [
                    torch.zeros(size=(self.num_envs, 2)),
                    -6000 * torch.ones(size=(self.num_envs, 1)),
                ],
                dim=-1,
            ),
            tas=340 * torch.ones(size=(self.num_envs, 1)),
            sim_step_size_ms=sim_step_size_ms,
            device=self.device,
        )

        # define observation space
        self._observation_space = spaces.Dict()

        self._observation_space["aircraft_position_g"] = spaces.Box(
            low=np.reshape(self.position_min_limit.numpy(), (-1,)).astype(np_float),
            high=np.reshape(self.position_max_limit.numpy(), (-1,)).astype(np_float),
            shape=(3,),
            dtype=np_float,
        )  # 局部地轴系位置
        self._observation_space["aircraft_velocity_g"] = spaces.Box(
            low=-1500, high=1500, shape=(3,), dtype=np_float
        )  # 局部地轴系速度
        self._observation_space["aircraft_tas"] = spaces.Box(
            low=0, high=2000, shape=(1,), dtype=np_float
        )  # 真空速
        self._observation_space["aircraft_chi"] = spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np_float
        )  # 速度系偏航角 rad [-π, π)
        self._observation_space["aircraft_gamma"] = spaces.Box(
            low=-np.pi / 2, high=np.pi / 2, shape=(1,), dtype=np_float
        )  # 速度系俯仰角 rad [-π/2, π/2]
        self._observation_space["aircraft_mu"] = spaces.Box(
            low=-np.pi, high=np.pi, shape=(1,), dtype=np_float
        )  # 滚转角 rad [-π, π)
        self._observation_space["aircraft_alpha"] = spaces.Box(
            low=-np.pi / 2, high=np.pi / 2, shape=(1,), dtype=np_float
        )  # 迎角 rad [-π/2, π/2]
        navigation_point = spaces.Dict()
        navigation_point["navigation_point_position_g"] = spaces.Box(
            low=np.reshape(self.position_min_limit.numpy(), (-1,)).astype(np_float),
            high=np.reshape(self.position_max_limit.numpy(), (-1,)).astype(np_float),
            shape=(3,),
            dtype=np_float,
        )
        self._observation_space["navigation_points"] = spaces.Tuple(
            [navigation_point] * self.navigation_points_visible_num
        )  # (滑动窗口)剩余可见&未达导航点

        # print(self._observation_space)

        # define action space
        self._action_space = spaces.Dict()
        self._action_space["thrust_cmd"] = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np_float
        )
        self._action_space["alpha_cmd"] = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np_float
        )
        self._action_space["mu_cmd"] = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np_float
        )

        # deffine reward functions
        self._reward_fns: list[BaseRewardFn] = [
            ReachNavigationPointRewardFn(min_distance_m=200, weight=0),
            ApproachNavigationPointRewardFn(weight=1),
        ]

        # define termination functions
        self._termination_fns: list[BaseTerminationFn] = [
            LowAltitudeTerminationFn(min_altitude_m=100),
            LowAirSpeedTerminationFn(min_airspeed_mps=10),
            ReachNavigationPointMaxNumTerminationFn(),
            TimeoutTerminationFn(1 * 60),
        ]

        # define simulation variables
        self.__sim_time_ms = torch.zeros(size=(self.num_envs, 1), device=self.device)
        self.__objects_states: list[ObjectState] = []
        self.__render_timestamp_ms = -float("inf")
        self.__render_count = 0

    @property
    def sim_time_ms(self) -> torch.Tensor:
        return self.__sim_time_ms

    @property
    def sim_time_s(self) -> torch.Tensor:
        return 0.001 * self.__sim_time_ms

    @cached_property
    def observation_space(self):
        return space2box(self._observation_space, dtype=self.np_dtype)

    @cached_property
    def action_space(self):
        return space2box(self._action_space, dtype=self.np_dtype)

    def generate_navigation_points(
        self,
        nenvs: int,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num = self.navigation_points_total_num
        npad = self.navigation_points_visible_num - 1  # 用最后一个导航点额外填充的数量
        rng = np.random.default_rng(seed)
        goals = np.concatenate(
            [
                rng.integers(
                    cast(int, self.position_min_limit[..., i].item()),
                    cast(int, self.position_max_limit[..., i].item()),
                    [nenvs, num + npad, 1],
                )
                for i in range(self.position_min_limit.shape[-1])
            ],
            axis=-1,
        )  # (...,N+npad,3)
        goals[..., -npad + 1 :, :] = goals[..., [-npad], :]  # 最后一个导航点填充

        navigation_points = torch.asarray(goals, dtype=self.dtype, device=self.device)
        navigation_point_index = torch.zeros(
            size=[*navigation_points.shape[:-2], 1, navigation_points.shape[-1]],
            dtype=torch.int64,
            device=self.device,
        )
        return navigation_points, navigation_point_index

    @torch.no_grad()
    def render_navigation_points(self):
        if self.render_mode:
            env_idx = 0  # 选择渲染的环境编号
            for i in range(self.navigation_points_visible_num):
                index = self.cur_nav_point_index[env_idx] + i  # (1,3)
                navigation_point = torch.gather(
                    self.navigation_points[env_idx], dim=1, index=index
                ).squeeze(1)

                index = index[0, 0].item()
                navigation_point = ObjectState(
                    sim_time_s=self.sim_time_s[0].item(),
                    name="{}th navigation point".format(index),
                    attr=WaypointAttr(name="{}th navigation point".format(index + 1)),
                    pos_ned=navigation_point[0],
                )
                self.__objects_states.append(navigation_point)

    def render_object_state(self, object_state: ObjectState):
        self.__objects_states.append(object_state)

    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)

        # reset aircraft model
        self.aircraft.reset(env_indices)
        self.aircraft.activate(env_indices)

        # reset navigation point
        # 穿梭机任务，按顺序经过所有导航点
        # generate navigation points  0: N, 1: E, 2: D, 3: 标志位
        navigation_points, navigation_point_index = self.generate_navigation_points(
            len(env_indices)
        )
        try:
            self.navigation_points[env_indices, ...] = navigation_points
            self.cur_nav_point_index[env_indices, ...] = navigation_point_index
        except AttributeError:
            self.navigation_points = navigation_points
            self.cur_nav_point_index = navigation_point_index
        self._update_navigation_point()

        # print("navigation points: " , self.navigation_points)
        # print("navigation point index: ", self.navigation_point_index)
        # 设置飞机角度为正对着导航点
        # selected_points = torch.gather(self.navigation_points, dim=1, index=self.navigation_point_index).squeeze(1)
        # aer = ned2aer(selected_points[env_indices]-self.aircraft.position_g[env_indices])
        # # print("aer: ", aer)
        # self.aircraft.set_q_kg(
        #     gamma = aer[..., 1:2],
        #     chi = aer[..., 0:1],
        #     env_indices = env_indices
        # )
        # euler = euler_from_quat(self.aircraft.q_kg[env_indices])
        # print("euler: ", euler)

        # reset simulation variables
        self.__sim_time_ms[env_indices] = 0.0

        self.__render_timestamp_ms = -float("inf")

        for reward_fn in self._reward_fns:
            reward_fn.reset(self, env_indices)

        obs_dict = self.__get_obs(env_indices)
        info = {}

        if 0 in env_indices:
            self.__objects_states.clear()
            self.render_navigation_points()

        obs = flatten(self._observation_space, obs_dict)
        obs_ = self._cast_out(obs)
        return obs_, info

    def _update_navigation_point(self):
        """缓存当前导航点信息"""
        self.cur_nav_pos = torch.gather(
            self.navigation_points,
            dim=-2,
            index=self.cur_nav_point_index,
        ).squeeze(
            -2
        )  # (nenvs,3), 当前需要到达的导航点位置

        self.cur_nav_LOS = (
            self.cur_nav_pos - self.aircraft.position_e
        )  #  飞机-当前导航点视线 (nenvs,3)
        self.cur_nav_dist = torch.norm(self.cur_nav_LOS, dim=-1, p=2).clip(
            1e-6
        )  # (nenvs,) 飞机-导航点距离
        self.cur_nav_LOS_azimuth, self.cur_nav_LOS_elevation, self.cur_nav_dist = (
            torch.split(ned2aer(self.cur_nav_LOS), [1, 1, 1], dim=-1)
        )

    def _cast_out(self, data: torch.Tensor) -> torch.Tensor | np.ndarray:
        if self._out_as_np:
            data = data.cpu().numpy()
        return data

    def step(self, action: torch.Tensor):
        step_num = self.agent_step_size_ms // self.sim_step_size_ms
        for i in range(step_num):
            self.__sim_time_ms += self.sim_step_size_ms
            if (
                self.render_mode
                and self.sim_time_ms[0] - self.__render_timestamp_ms
                > self._render_interval_ms
            ):
                self.__render_timestamp_ms = self.sim_time_ms[0].item()
                record_flag = True
            else:
                record_flag = False

            self.aircraft.run(action)
            if record_flag:
                euler = euler_from_quat(self.aircraft.q_kg)[0, ...]
                euler[0] = self.aircraft.mu[0, :]

                aircraft_state = ObjectState(
                    sim_time_s=0.001 * self.sim_time_ms[0].item(),
                    name=self.aircraft.model_name,
                    attr=AircraftAttr(
                        Color=self.aircraft.model_color,
                        TAS=self.aircraft.tas[0].item(),
                    ),
                    pos_ned=self.aircraft.position_e[0, ...].detach().cpu(),
                    rpy_rad=euler.detach().cpu(),
                )
                self.__objects_states.append(aircraft_state)

        obs_dict = self.__get_obs()

        rew = self.__get_rew()
        truncated = torch.zeros(
            size=(self.num_envs, 1), dtype=torch.bool, device=self.device
        )
        terminated = self.__is_terminated()
        done = terminated | truncated

        info = {}

        if torch.any(done):
            indices = torch.where(done)[0]

            # render
            if self.render_mode == "tacview" and self.render_dir:
                if done[0].item():
                    self.__render_count += 1
                    # if self.__render_count % 10 == 0:
                    # 对数据按照时间进行排序
                    self.__objects_states.sort(key=lambda x: x.sim_time_s)
                    file_path = self.render_dir / "{}.acmi".format(self.__render_count)
                    with open(file_path, "w") as f:
                        f.write("FileType=text/acmi/tacview\n")
                        f.write("FileVersion=2.2\n")
                        f.write(
                            "0,ReferenceTime={}Z\n".format(
                                datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                            )
                        )
                        cur_sim_time_s = -1.0
                        for object_state in self.__objects_states:
                            if object_state.sim_time_s > cur_sim_time_s:
                                cur_sim_time_s = object_state.sim_time_s
                                f.write("#{:.2f}\n".format(cur_sim_time_s))

                            # 写入目标位置
                            f.write(
                                "{},T={}".format(
                                    object_state.id,
                                    "|".join(
                                        [
                                            "{:7f}".format(v)
                                            for v in object_state.pos_lla
                                        ]
                                    ),
                                )
                            )

                            # 写入目标姿态
                            if object_state.rpy_deg:
                                f.write(
                                    "|".join(
                                        [""]
                                        + [
                                            "{:2f}".format(v)
                                            for v in object_state.rpy_deg
                                        ]
                                    )
                                )

                            # 写入目标属性
                            attr_dict = object_state.attr.model_dump(exclude_none=True)
                            f.write(
                                ",".join(
                                    [""] + [f"{k}={v}" for k, v in attr_dict.items()]
                                )
                            )
                            f.write("\n")

                            # 写入事件
                            pass

            # reset specific envs
            _obs_dict, _info = self.reset(indices)
            obs_dict = self.__get_obs()  # TODO: 这里可以改进为利用上前面_obs_dict信息的

        obs = flatten(self._observation_space, obs_dict)
        return (
            self._cast_out(obs),
            self._cast_out(rew),
            self._cast_out(terminated),
            self._cast_out(truncated),
            info,
        )

    def render(self):
        pass

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
        else:
            raise TypeError("unsupported type for env_indices", type(env_indices))

        obs_dict = OrderedDict()
        obs_dict["aircraft_position_g"] = self.aircraft.position_e[env_indices]
        obs_dict["aircraft_velocity_g"] = self.aircraft.velocity_e[env_indices]
        obs_dict["aircraft_tas"] = self.aircraft.tas[env_indices]
        euler = euler_from_quat(self.aircraft.q_kg)
        chi = euler[..., 2:3]
        gamma = euler[..., 1:2]
        obs_dict["aircraft_chi"] = chi[env_indices]
        obs_dict["aircraft_gamma"] = gamma[env_indices]
        obs_dict["aircraft_mu"] = self.aircraft.mu[env_indices]
        obs_dict["aircraft_alpha"] = self.aircraft.alpha[env_indices]

        navigation_points = []
        for i in range(self.navigation_points_visible_num):
            navigation_point_dict = OrderedDict()
            navigation_point = torch.gather(
                self.navigation_points, dim=1, index=self.cur_nav_point_index + i
            ).squeeze(
                1
            )  # (nenvs,3)
            navigation_point_dict["navigation_point_position_g"] = navigation_point[
                env_indices
            ]
            # aer = ned2aer(navigation_point-self.aircraft.position_g)
            # navigation_point_dict["navigation_point_az"] = aer[env_indices, 0:1]
            # navigation_point_dict["navigation_point_elev"] = aer[env_indices, 1:2]
            # navigation_point_dict["navigation_point_slant_range"] = aer[env_indices, 2:3]/torch.norm((self.position_max_limit.to(device=self.device)-self.position_min_limit.to(device=self.device)).to(dtype=torch.float32), p=2)

            navigation_points.append(navigation_point_dict)
        obs_dict["navigation_points"] = tuple(navigation_points)

        return obs_dict

    def __get_rew(self) -> torch.Tensor:
        reward = torch.zeros(
            size=(self.num_envs, 1), device=self.device, dtype=self.dtype
        )
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
            terminated = terminated | _terminated
        return terminated
