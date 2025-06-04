from __future__ import annotations
from copy import deepcopy
import logging
import traceback
from typing import TYPE_CHECKING

from .utils.log_ext import LogConfig

if TYPE_CHECKING:
    from .proto4venv import _EnvIndexType
    from numpy.typing import NDArray
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
from .models.base_model import BaseModel
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
    affcmb,
)
from .utils.tacview_render import ObjectState, AircraftAttr, WaypointAttr, get_obj_id
from .utils.tacview import ACMI_Types
from .reward_fns import *
from .termination_fns import *
from .proto4venv import TrueSyncVecEnv
from .utils.tacview import acmi_id

_LOGR = logging.getLogger(__name__)


class NavigationEnv(TrueSyncVecEnv):
    metadata: dict[str, Any] = {
        "render_modes": ["tacview"],
        "render_fps": 25,  # 渲染 ACMI 频率(仿真时间)
    }
    name = "NavigationEnv"

    def __init__(
        self,
        agent_step_size_ms: int,  # 决策步长(ms)
        sim_step_size_ms: int,  # 仿真步长(ms)
        position_min_limit: Sequence[float],  # (x_\min,y_\min,z_\min) 活动范围约束
        position_max_limit: Sequence[float],  # (x_\max,y_\max,z_\max) 活动范围约束
        waypoints_visible_num: int = 1,  # 每一时刻对飞机可见的导航点数量
        waypoints_total_num: int = 1,  # 总导航点数量
        waypoints_dR_ratio_min: float = 0.0,  # 相邻导航点间距最小值比例 in (0,1)
        waypoints_dR_ratio_max: float = 1.0,  # 相邻导航点间距最大值比例 in (0,1)
        waypoints_gen_range_ratio: float = 0.5,  # 导航点生成区半径比例 in (0,1)
        lat0: float = 39.9042,  # 坐标原点纬度, deg
        lon0: float = 116.4074,  # 坐标原点经度, deg
        alt0: float = 5000,  # 坐标原点高度, m
        render_mode: str | None = None,
        render_dir: Path | None = None,
        num_envs: int = 1,
        max_sim_ms: int = 600_000,  # 单局最大仿真时长(ms)
        device=torch.device("cpu"),
        dtype=torch.float,
        np_float=np.float32,
        out_torch=1,  # step输出obs格式, 0->ndarray, 1->torch.Tensor
        logconfig: LogConfig | None = None,  # 日志配置
    ):
        super().__init__(num_envs=num_envs, device=device, dtype=dtype)
        self._logr = _LOGR if logconfig is None else logconfig.make()
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
        self.mode = out_torch
        self.max_sim_time_ms = int(max_sim_ms)
        # assert out_numpy in ["numpy", "pytorch"], "mode must be 'numpy' or 'pytorch'"
        # self._out_as_np = out_numpy == "numpy"
        self._out_as_np = not out_torch
        self.waypoints_total_num = waypoints_total_num
        self.waypoints_visible_num = waypoints_visible_num  # horizon
        assert waypoints_visible_num >= 1, "expect navigation_points_visible_num >= 1"
        self.position_min_limit = torch.tensor(
            position_min_limit,
            dtype=torch.int64,
        ).ravel()  # (3,)
        self.position_max_limit = torch.tensor(
            position_max_limit,
            dtype=torch.int64,
        ).ravel()  # (3,)

        # 坐标系原点的地理位置
        self._origin_lat = lat0
        """坐标原点纬度, deg"""
        self._origin_lon = lon0
        """坐标原点经度, deg"""
        self._origin_alt = alt0
        """坐标原点高度, m"""

        self.render_mode = render_mode
        if self.render_mode:
            assert render_dir is not None
            if not render_dir.exists():
                render_dir.mkdir(parents=True)
            self.render_dir = render_dir

            # 计算渲染帧间隔
            self._render_interval_ms = round((1000 / self.metadata["render_fps"]))

        _0f = torch.zeros(size=(self.num_envs, 1), dtype=dtype, device=device)

        # 创建战斗机模型
        tas = 340 + _0f
        self.aircraft =pln= PointMassAircraft(
            id=0x10086,
            acmi_name="J-10",
            call_sign="agent",
            acmi_color="Red",
            position_e=torch.cat(
                [_0f, _0f, _0f],
                dim=-1,
            ),
            tas=tas,
            alt0=alt0,
            sim_step_size_ms=sim_step_size_ms,
            device=device,
            dtype=dtype,
        )
        pln.logr = self._logr

        # define observation space
        self._observation_space = spaces.Dict()

        self._observation_space["aircraft_position_g"] = spaces.Box(
            low=np.reshape(self.position_min_limit.numpy(), (-1,)).astype(np_float),
            high=np.reshape(self.position_max_limit.numpy(), (-1,)).astype(np_float),
            shape=(3,),
            dtype=np_float,
        )  # 局部地轴系位置
        rmax = 5000
        self._observation_space["aircraft_velocity_g"] = spaces.Box(
            low=np.asarray([-rmax, -rmax, -rmax], dtype=np_float),
            high=np.asarray([rmax, rmax, rmax], dtype=np_float),
            shape=(3,),
            dtype=np_float,
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
            [navigation_point] * self.waypoints_visible_num
        )  # (滑动窗口)剩余可见&未达导航点

        # print(self._observation_space)

        # define action space
        self._action_space = spaces.Dict()
        self._action_space["thrust_cmd"] = spaces.Box(
            low=0, high=1.0, shape=(1,), dtype=np_float
        )
        self._action_space["alpha_cmd"] = spaces.Box(
            low=0, high=1.0, shape=(1,), dtype=np_float
        )
        self._action_space["mu_cmd"] = spaces.Box(
            low=0, high=1.0, shape=(1,), dtype=np_float
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
        self.__objects_states: list[ObjectState] = []
        self.__render_timestamp_ms = -float("inf")
        self.__render_count = 0

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
        """产生一组路径点
        TODO: 增加连续性约束，例如相邻点距、曲率，在约束下快速生成

        Returns:
            navigation_points: shape (nenvs, N+n-1, 3)
            navigation_point_index: shape (nenvs, 1, 3)
        """
        device = self.device  # @generate_navigation_points
        dtype = self.dtype  # @generate_navigation_points
        num = self.waypoints_total_num
        npad = self.waypoints_visible_num - 1  # 用最后一个导航点额外填充的数量
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

        navigation_points = torch.asarray(goals, dtype=dtype, device=device)
        navigation_point_index = torch.zeros(
            size=[*navigation_points.shape[:-2], 1, navigation_points.shape[-1]],
            dtype=torch.int64,
            device=device,
        )
        return navigation_points, navigation_point_index

    def render_navigation_points(self):
        if self.render_mode:
            env_idx = 0  # 选择渲染的环境编号
            for i in range(self.waypoints_visible_num):
                index = self.cur_nav_point_index[env_idx] + i  # (1,3)
                navigation_point = torch.gather(
                    self.navigation_points[env_idx], dim=1, index=index
                ).squeeze(1)

                index = index[0, 0].item()
                navigation_point = ObjectState(
                    sim_time_s=self.sim_time_s[0].item(),
                    name="{}th navigation point".format(index),
                    attr=WaypointAttr(name="{}th navigation point".format(index + 1)),
                    lat0=self._origin_lat,
                    lon0=self._origin_lon,
                    h0=self._origin_alt,
                    pos_ned=navigation_point[0],
                )
                self.__objects_states.append(navigation_point)

    def render_object_state(self, object_state: ObjectState):
        self.__objects_states.append(object_state)

    @torch.no_grad()
    def reset(self, env_indices: _EnvIndexType = None, cast_out=True):
        env_indices = self.proc_indices(env_indices)

        pln = self.aircraft  # @reset
        # reset aircraft model
        pln.reset(env_indices)
        pln.activate(env_indices)

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
        # aer = ned2aer(selected_points[env_indices]-pln.position_g[env_indices])
        # # print("aer: ", aer)
        # pln.set_q_kg(
        #     gamma = aer[..., 1:2],
        #     chi = aer[..., 0:1],
        #     env_indices = env_indices
        # )
        # euler = euler_from_quat(pln.q_kg[env_indices])
        # print("euler: ", euler)

        # reset simulation variables
        self._sim_time_ms[env_indices] = 0
        self.sync_sim_time(env_indices)

        self.__render_timestamp_ms = -float("inf")

        for reward_fn in self._reward_fns:
            reward_fn.reset(self, env_indices)

        obs_dict = self.__get_obs(env_indices)
        info = {}

        if 0 in env_indices:
            self.__objects_states.clear()
            self.render_navigation_points()

        for tcf in self._termination_fns:
            tcf.reset(self)

        obs = flatten(self._observation_space, obs_dict)
        obs = self._cast_out(obs, cast_out)
        return obs, info

    def _update_navigation_point(self):
        """缓存当前导航点信息"""
        pln = self.aircraft  # @update_navigation_point
        self.cur_nav_pos = torch.gather(
            self.navigation_points,
            dim=-2,
            index=self.cur_nav_point_index,
        ).squeeze(
            -2
        )  # 当前需要到达的导航点坐标, shape (nenvs,3)

        self.cur_nav_LOS = (
            self.cur_nav_pos - pln.position_e()
        )  #  飞机-当前导航点视线 (nenvs,3)
        self.cur_nav_dist = torch.norm(self.cur_nav_LOS, dim=-1, p=2).clip(
            1e-6
        )  # (nenvs,) 飞机-导航点距离
        (
            self.cur_nav_LOS_azimuth,
            self.cur_nav_LOS_elevation,
            self.cur_nav_dist,
        ) = torch.split(ned2aer(self.cur_nav_LOS), [1, 1, 1], dim=-1)

    def _cast_out(self, data: torch.Tensor, accept=True) -> torch.Tensor | NDArray:
        if self._out_as_np and accept:
            data = data.cpu().numpy()
        return data

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        step_num = self.agent_step_size_ms // self.sim_step_size_ms
        pln = self.aircraft  # @step
        idx_rcd = 0  # 选择渲染的环境编号

        for i in range(step_num):
            self._sim_time_ms[...] += self.sim_step_size_ms
            self.sync_sim_time()
            if (
                self.render_mode
                and self.sim_time_ms[idx_rcd] - self.__render_timestamp_ms
                > self._render_interval_ms
            ):
                self.__render_timestamp_ms = self.sim_time_ms[idx_rcd].item()
                record_flag = True
            else:
                record_flag = False

            pln.run(action)
            if record_flag:
                rpy_eb = pln.rpy_eb(idx_rcd)

                aircraft_state = ObjectState(
                    sim_time_s=self.sim_time_s[idx_rcd].item(),
                    name=acmi_id(int(pln.id[idx_rcd].item())),
                    attr=AircraftAttr(
                        Color=pln.acmi_color,
                        TAS=f"{pln.tas(idx_rcd).item():.2f}",
                    ),
                    pos_ned=pln.position_e(idx_rcd).cpu(),
                    rpy_rad=rpy_eb.cpu(),
                    lat0=self._origin_lat,
                    lon0=self._origin_lon,
                    h0=self._origin_alt,
                )
                self.__objects_states.append(aircraft_state)

        obs_dict = self.__get_obs()

        rew = self.__get_rew()
        truncated = self.sim_time_ms >= self.max_sim_time_ms
        terminated = self.__is_terminated()
        done = terminated | truncated

        info = {}

        obs = flatten(self._observation_space, obs_dict).clone()
        obs_out = self._cast_out(obs)
        info[self.KEY_FINAL_OBS] = obs_out
        info[self.KEY_FINAL_INFO] = deepcopy(info)

        if torch.any(done):
            indices = torch.where(done)[0]

            # render
            if done[idx_rcd].item():
                self._render()

            # reset specific envs
            obs_, info_ = self.reset(indices, cast_out=False)

            # 合并信息
            obs_ = cast(torch.Tensor, obs_)
            obs[indices] = obs_
            info.update(info_)

            obs_out = self._cast_out(obs)

        return (
            obs_out,
            self._cast_out(rew),
            self._cast_out(terminated),
            self._cast_out(truncated),
            info,
        )

    def render(self):
        # 本函数不要动
        pass

    def _render(self):
        if self.render_mode == "tacview" and self.render_dir:
            self.__render_count += 1
            # if self.__render_count % 10 == 0:
            # 对数据按照时间进行排序
            self.__objects_states.sort(key=lambda x: x.sim_time_s)
            file_path = self.render_dir / "{}.acmi".format(self.__render_count)

            buf = [
                "FileType=text/acmi/tacview\n",
                "FileVersion=2.2\n",
                "0,ReferenceTime={}Z\n".format(
                    datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                ),
            ]
            cur_sim_time_s = -1.0
            for object_state in self.__objects_states:
                if object_state.sim_time_s > cur_sim_time_s:
                    cur_sim_time_s = object_state.sim_time_s
                    buf.append("#{:.2f}\n".format(cur_sim_time_s))

                attrs = [
                    "{:7f}".format(v) for v in object_state.pos_lbh
                ]  # long,lat,alt (deg,m)
                if object_state.rpy_deg:  # 姿态 roll,pitch,yaw (deg)
                    attrs.extend(["{:2f}".format(v) for v in object_state.rpy_deg])
                T = ["|".join(attrs)]

                attr_dict = object_state.attr.model_dump(exclude_none=True)
                T.extend([f"{k}={v}" for k, v in attr_dict.items()])  # 其他属性

                buf.append("{},T={}\n".format(object_state.id, ",".join(T)))

                # 写入事件
                pass
            buf.append("\n")
            with open(file_path, "w") as f:
                f.writelines(buf)

    def __get_obs(self, env_indices: _EnvIndexType = None) -> OrderedDict:
        env_indices = self.proc_indices(env_indices)
        pln = self.aircraft  # @get_obs

        obs_dict: OrderedDict[str, torch.Tensor | Any] = OrderedDict()
        obs_dict["aircraft_position_g"] = pln.position_e(env_indices)
        obs_dict["aircraft_velocity_g"] = pln.velocity_e(env_indices)
        obs_dict["aircraft_tas"] = pln.tas(env_indices)
        obs_dict["aircraft_chi"] = pln.chi(env_indices)
        obs_dict["aircraft_gamma"] = pln.gamma(env_indices)
        obs_dict["aircraft_mu"] = pln.mu(env_indices)
        obs_dict["aircraft_alpha"] = pln.alpha(env_indices)

        navigation_points = []
        for i in range(self.waypoints_visible_num):
            navigation_point_dict = OrderedDict()
            navigation_point = torch.gather(
                self.navigation_points, dim=1, index=self.cur_nav_point_index + i
            ).squeeze(
                1
            )  # (nenvs,3)
            navigation_point_dict["navigation_point_position_g"] = navigation_point[
                env_indices
            ]
            # aer = ned2aer(navigation_point-pln.position_g)
            # navigation_point_dict["navigation_point_az"] = aer[env_indices, 0:1]
            # navigation_point_dict["navigation_point_elev"] = aer[env_indices, 1:2]
            # navigation_point_dict["navigation_point_slant_range"] = aer[env_indices, 2:3]/torch.norm((self.position_max_limit.to(device=self.device)-self.position_min_limit.to(device=self.device)).to(dtype=torch.float32), p=2)

            navigation_points.append(navigation_point_dict)
        obs_dict["navigation_points"] = tuple(navigation_points)

        return obs_dict

    def __get_rew(self) -> torch.Tensor:
        reward = torch.zeros(
            size=(self.num_envs, 1), device=self.device, dtype=self.dtype  # @get_rew
        )
        plane = self.aircraft  # @get_rew
        for reward_fn in self._reward_fns:
            _reward = reward_fn(self, plane)
            reward += _reward
        return reward

    def __is_terminated(self) -> torch.Tensor:
        plane = self.aircraft  # @get_terminate
        terminated = torch.zeros(
            size=(self.num_envs, 1),
            dtype=torch.bool,
            device=self.device,  # @get_terminate
        )
        for termination_fn in self._termination_fns:
            try:
                _terminated = termination_fn(self, plane)
            except Exception as e:
                print(traceback.format_exc())
                raise e
            if _terminated[0]:
                _LOGR.info(
                    "Env[{}] terminated: {}".format(
                        0, termination_fn.__class__.__name__
                    )
                )
            terminated = terminated | _terminated
        return terminated
