from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import logging
import math
from os import PathLike
import traceback
from typing import TYPE_CHECKING

from pymap3d import ned2geodetic
import socket

from .utils.dig import DIG_clean, DIG_property
from .utils.log_ext import LogConfig


from datetime import datetime

# import torch
from typing import Literal, cast
from functools import cached_property
import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Any, Sequence
from collections import OrderedDict, defaultdict
from pathlib import Path
from .simulators.proto4model import BaseModel
from .utils.space import space2box, flatten, unflatten
from .utils.math_np import (
    B01toI,
    quat_enu_ned,
    quat_rotate,
    quat_rotate_inv,
    euler_from_quat,
    Qx,
    quat_mul,
    ned2aer,
    affcmb,
    norm_,
    calc_zem1,
    vec_cosine,
    Float_NDArr,
    BoolNDArr,
)
from .utils.tacview_render import ObjectState, AircraftAttr, WaypointAttr, get_obj_id
from .utils.tacview import ACMI_Types, TacviewRecorder
from .reward_fns import *
from .termination_fns import *
from .proto4venv import SyncVecEnv as VEnv
from .utils.tacview import format_id

if TYPE_CHECKING:
    from .proto4venv import EnvMaskType
    from numpy.typing import NDArray

PI = math.pi


def assert_finite(x: NDArray, name: str) -> None:
    tag = np.isfinite(x)
    assert tag.all(), (f"non-finite value(s) in {name}", np.where(~tag))


@dataclass
class RowWithTime:
    t_ms: int
    msg: str


class NavHeadingEnv(VEnv):
    r"""
    定点导航问题: 状态=导航点位置, 控制量=过载+滚转;
    约定:导航点在坐标原点
    """

    RENDER_MODE_LOCAL = "tacview_local"
    RENDER_MODE_REMOTE = "tacview_remote"

    metadata: dict[str, Any] = {
        "render_fps": 25,  # 渲染频率, 单位:1/仿真sec
        "render_modes": [
            RENDER_MODE_LOCAL,  # ACMI 文件
            RENDER_MODE_REMOTE,  # 实时遥测
        ],
    }

    _waypoints: np.ndarray
    """路径点全集, shape=(nenvs,N+P,d)"""
    _waypoint_index: np.ndarray
    """当前路径点索引, shape=(nenvs,1,d)"""

    def __init__(
        self,
        agent_step_size_ms: int = 100,  # 决策步长(ms)
        sim_step_size_ms: int = 20,  # 仿真步长(ms)
        max_sim_ms: int = 600_000,  # 单局最大仿真时长(ms)
        xmax: float = 10_000,  # X方向活动半径
        ymax: float | None = None,  # Y方向活动半径
        zmax: float = 5_000,  # 高度活动半径
        pos_e_nvec: Sequence[int] | int = 100,  # XYZ位置离散化个数 linsapce
        V0: float = 240.0,  # 初始速度, m/s
        Vmin: float = 100.0,  # 最小速度, m/s
        Vmax: float | None = None,  # 最大速度, m/s
        nx_max: float = 2.0,  # 最大X方向过载, unit:G
        nx_min: float = -0.5,  # 最小X方向过载, unit:G
        ny_max: float = 0.5,  # 最大Y方向过载, unit:G
        nz_max: float = 0.5,  # NED+Z方向最大过载, unit:G
        nz_min: float = -0.5,  # NED+Z方向最小过载, unit:G
        dmu_max: float = math.radians(360 / 6),  # 最大滚转角速度, unit:rad/s
        waypoints_visible_num: int = 10,  # 路径点可见数量
        waypoints_total_num: int = 1,  # 路径点总数量
        num_envs: int = 1,
        render_mode: str | None = None,
        render_dir: PathLike | None = None,
        render_port: int | None = None,
        lat0: float = 39.9042,  # 原点纬度, deg
        lon0: float = 116.4074,  # 原点经度, deg
        alt0: float = 5000,  # 原点高度, m
        use_gravity=False,
        version="1.0",
        easy_mode=False,
        **kwargs,
    ):
        self._version = version
        self._easy_mode = easy_mode

        super().__init__(
            num_envs=num_envs,
            sim_step_size_ms=sim_step_size_ms,
            max_sim_ms=max_sim_ms,
            **kwargs,
        )
        logr = self.logger
        self._use_gravity = use_gravity
        np_flt = self.dtype
        # time config
        assert agent_step_size_ms > 0 and sim_step_size_ms > 0, (
            "仿真步长必须大于0",
            sim_step_size_ms,
        )
        assert (
            agent_step_size_ms // sim_step_size_ms > 0
            and agent_step_size_ms % sim_step_size_ms == 0
        ), (
            "决策步长必须为仿真步长的正整数倍",
            agent_step_size_ms,
            "/",
            sim_step_size_ms,
        )
        self._agent_step_size_ms = agent_step_size_ms

        nenv = self.num_envs
        _0f1 = np.zeros(shape=(1,), dtype=np_flt)
        _0f3 = np.zeros(shape=(3,), dtype=np_flt)

        assert waypoints_visible_num > 0, (
            "waypoints_visible_num must be positive",
            waypoints_visible_num,
        )
        self._waypoints_visible_num = waypoints_visible_num
        assert waypoints_total_num > 0, (
            "waypoints_total_num must be positive",
            waypoints_total_num,
        )
        self._waypoints_total_num = waypoints_total_num  # 有效路径点总数

        # vel
        eps = 1e-3
        Vmin = Vmin or (V0 - eps)
        Vmax = Vmax or (V0 + eps)
        assert Vmin < V0 < Vmax, (
            "Vmin <= V0 <= Vmax",
            Vmin,
            V0,
            Vmax,
        )
        self._tas_range = spaces.Box(
            low=Vmin + _0f1,
            high=Vmax + _0f1,
            shape=(1,),
            dtype=np_flt,
        )
        self._vel_e_range = spaces.Box(
            low=Vmin + _0f3,
            high=Vmax + _0f3,
            shape=(3,),
            dtype=np_flt,
        )

        # position
        ymax = ymax or xmax
        assert xmax > 0, ("xmax must be positive", xmax)
        assert ymax > 0, ("ymax must be positive", ymax)
        assert zmax > 0, ("zmax must be positive", zmax)
        self._pos_e_range = spaces.Box(
            low=np.asarray([-xmax, -ymax, -zmax], dtype=np_flt),
            high=np.asarray([xmax, ymax, zmax], dtype=np_flt),
            shape=(3,),
            dtype=np_flt,
        )
        self._pos_e_nvec = (np.zeros(3, np.int64) + pos_e_nvec).reshape((3,))  # (3,)
        self._pos_e_disc_table: list[Float_NDArr] = [
            np.reshape(
                np.linspace(
                    self._pos_e_range.low[i],
                    self._pos_e_range.high[i],
                    self._pos_e_nvec[i],
                ),
                (1, 1, -1),
            )
            for i in range(3)
        ]
        """离散化位置的索引表"""

        self._region_diam = np.linalg.norm((xmax, ymax, zmax)) * 2
        # pose
        self._qew_range = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np_flt,
        )
        self._roll_range = spaces.Box(
            low=-PI,
            high=PI,
            shape=(1,),
            dtype=np_flt,
        )
        self._pitch_range = spaces.Box(
            low=-PI / 2,
            high=PI / 2,
            shape=(1,),
            dtype=np_flt,
        )
        self._yaw_range = spaces.Box(
            low=-PI,
            high=PI,
            shape=(1,),
            dtype=np_flt,
        )

        # control
        self._nx_range = spaces.Box(
            low=nx_min + _0f1,
            high=nx_max + _0f1,
            shape=(1,),
            dtype=np_flt,
        )
        self._ny_range = spaces.Box(
            low=-ny_max + _0f1,
            high=ny_max + _0f1,
            shape=(1,),
            dtype=np_flt,
        )
        self._nz_range = spaces.Box(
            low=nz_min + _0f1,
            high=nz_max + _0f1,
            shape=(1,),
            dtype=np_flt,
        )
        self._dmu_range = spaces.Box(
            low=-dmu_max + _0f1,
            high=dmu_max + _0f1,
            shape=(1,),
            dtype=np_flt,
        )

        # 坐标系原点的地理位置(用于tacview渲染)
        self.lat0 = lat0
        """坐标原点纬度, deg"""
        self.lon0 = lon0
        """坐标原点经度, deg"""
        self.alt0 = alt0
        """坐标原点高度, m"""

        version = self._version  # @make_game
        if version.startswith("1."):
            self._make_game_v1()
        elif version.startswith("2."):
            self._make_game_v2()
        else:
            raise ValueError(f"Unsupported version: {version}")

        self._rew_sum = {
            i: np.zeros(
                (nenv, 1),
                dtype=np_flt,
                # device=device
            )
            for i, rf in self._reward_fns.items()
        }
        """无折扣累计奖励分量, shape=(nenv,1)"""

        # render config
        self.render_mode = render_mode
        self._render_dir = Path(render_dir) if render_dir else None
        self._render_port = render_port
        self._render_dt_ms = round((1000 / self.metadata["render_fps"]))  # 渲染帧间隔
        self.__render_t0_ms = -math.inf
        self.__objects_states: list[ObjectState] = []
        self.__render_count = 0
        self._render_ienv = 0  # 记录数据的环境编号
        self._render_id = (self._render_ienv, -1)
        self._render_id_last = deepcopy(self._render_id)
        self._acmi_writer = TacviewRecorder()

        # 步数管理
        self._cur_episode = np.full(
            (nenv, 1),
            0,
            dtype=np.int64,
            # device=device
        )  # 从1起算
        self._cur_act_step = np.full(
            (nenv, 1),
            -1,
            dtype=np.int64,
            # device=device
        )  # 从0起算
        return

    OBSKEY_POS_E = "aircraft_position_e"
    OBSKEY_VEL_E = "aircraft_velocity_e"
    OBSKEY_TAS = "aircraft_tas"
    OBSKEY_CHI = "aircraft_chi"  # yaw
    OBSKEY_GAMMA = "aircraft_gamma"  # pitch
    OBSKEY_MU = "aircraft_mu"  # roll
    OBSKEY_ALPHA = "aircraft_alpha"
    OBSKEY_BETA = "aircraft_beta"
    OBSKEY_QEW = "aircraft_qeb"  # Q_{ew}
    OBSKEY_LOS = "target_los"

    OBSKEY_GOALS = "navigation_goals"

    ACTKEY_NX = "nx_cmd"
    ACTKEY_NY = "ny_cmd"
    ACTKEY_NZ = "nz_cmd"
    ACTKEY_DMU = "dmu_cmd"

    INFOKEY_OBS = "origin_obs"
    INFOKEY_RET = "cumulative_reward"
    INFOKEY_TERM_EVENT = "termination_events"

    def _make_game_v1(self):
        logr = self.logger
        np_ftype = self.dtype
        # device = self.device
        nenv = self.num_envs
        alt0 = self.alt0
        Vmin = self._tas_range.low.item()
        Vmax = self._tas_range.high.item()
        _0f1 = np.zeros(
            (nenv, 1),
            dtype=np_ftype,
            #  device=device,
        )
        tas = Vmax + _0f1
        hmin = alt0 - self._pos_e_range.high[2]
        hmax = alt0 - self._pos_e_range.low[2]

        from .simulators.aircraft.p6dof import P6DOFPlane as Plane

        # 创建飞机模型
        pln = Plane(
            group_shape=nenv,
            acmi_id=0x10086,
            acmi_name="J-10",
            call_sign="agent",
            acmi_color="Red",
            use_gravity=self._use_gravity,
            alt0=alt0,
            #  device=device,
            dtype=np_ftype,
            logger=self.logger,
        )
        pln.set_ic_pos_e(0, None)
        pln.set_ic_tas(tas, None)
        self.aircraft = pln

        # obs_space
        _obs_spc = OrderedDict()
        _obs_spc[self.OBSKEY_POS_E] = self._pos_e_range
        _obs_spc[self.OBSKEY_TAS] = self._tas_range
        _obs_spc[self.OBSKEY_QEW] = self._qew_range
        _obs_spc[self.OBSKEY_MU] = self._roll_range

        pos_span = self._pos_e_range.high - self._pos_e_range.low
        wp_obs_spc = spaces.Dict(
            [
                (
                    self.OBSKEY_LOS,
                    spaces.Box(
                        low=-pos_span,
                        high=pos_span,
                        shape=(3,),
                        dtype=np_ftype,
                    ),
                ),
            ]
        )
        _obs_spc[self.OBSKEY_GOALS] = spaces.Tuple(
            [wp_obs_spc] * self._waypoints_visible_num
        )
        self._observation_space = spaces.Dict(_obs_spc)

        # act_space
        _act_spc = OrderedDict()
        _act_spc[self.ACTKEY_NX] = self._nx_range
        _act_spc[self.ACTKEY_NY] = self._ny_range
        _act_spc[self.ACTKEY_NZ] = self._nz_range
        _act_spc[self.ACTKEY_DMU] = self._dmu_range
        self._action_space = spaces.Dict(_act_spc)

        diam = float(self._region_diam)
        tc_p_dmax = diam * 0.8  # 过远
        tc_p_dmin = min(100, diam * 1e-2)  # 抵达距离
        self._tc_p_dmin = tc_p_dmin
        self._tc_p_dmax = tc_p_dmax

        # !!!deffine reward functions
        self._reward_fns: dict[str, BaseRewardFn] = {
            str(i): rf
            for i, rf in enumerate(
                [
                    RF_GoalHeadingAngle(weight=100),
                    RF_GoalDistance(
                        use_dR=True, use_quadratic=True, weight=100 / (diam**2)
                    ),
                    RF_GoalDistance(
                        use_dR=False, use_quadratic=True, weight=100 / diam
                    ),
                    RF_GoalReach(min_distance_m=tc_p_dmin, weight=1000),
                    RF_Time(weight=-0.01),  # 代价
                ]
            )
        }

        # !!!define termination functions
        self._termination_fns: list[BaseTerminationFn] = [
            TC_LowAltitude(min_altitude_m=hmin),
            TC_LowTAS(min_tas_mps=Vmin),
            TC_Timeout(self.max_sim_time_ms * 1e-3),
            TC_AwayFromGoal(distance_threshold=tc_p_dmax),
            TC_ReachAllGoal(),
        ]

        logr.info(
            "reward functions:\n{}".format(
                "\n".join([f"{i}: {rf.repr}" for i, rf in self._reward_fns.items()])
            )
        )
        logr.info(
            "termination functions:\n{}"
            "".format(
                "\n".join(
                    [f"{i}: {tf.repr}" for i, tf in enumerate(self._termination_fns)]
                )
            )
        )

    def _make_game_v2(self):
        raise NotImplementedError

    def reward_fns(self) -> dict[str, BaseRewardFn]:
        return self._reward_fns

    @cached_property
    def observation_space(self):
        return space2box(self._observation_space, dtype=self.dtype)

    @cached_property
    def action_space(self):
        return space2box(self._action_space, dtype=self.dtype)

    def generate_waypoints(
        self,
        nenvs: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""产生一组路径点
        TODO: 增加连续性约束，例如相邻点距、曲率，在约束下快速生成
        Args:
            seed: 随机种子
        Returns:
            navigation_points: shape=(nenvs, N+horizon-1, 3).

            navigation_point_index: shape=(nenvs, 1, 3)
        """
        from .utils.respawn import generate_waypoints

        goals = generate_waypoints(
            group_shape=(nenvs,),
            num=self._waypoints_total_num,
            npad=self._waypoints_visible_num - 1,
            nvec=self._pos_e_nvec,
            disc_table=self._pos_e_disc_table,
            use_random=True,
            dtype=self.dtype,  # @generate_navigation_points
            disc_max=1,
            seed=seed,
        )  # (...,N+H,d)
        goals_index = np.zeros(
            [*goals.shape[:-2], 1, goals.shape[-1]],
            dtype=np.intp,
            #  device=device,
        )
        return goals, goals_index

    def render_waypoints(self, env_idx: int = 0):
        namefmt = "{}th waypoint"
        callsignfmt = "waypoint_{}"

        if self.render_mode in [self.RENDER_MODE_LOCAL, self.RENDER_MODE_REMOTE]:
            rcd = self._acmi_writer
            i_begin = int(self._waypoint_index[env_idx, 0, 0])  # .cpu().item()
            i_end = min(
                i_begin + self._waypoints_visible_num, self._waypoints_total_num
            )
            n = i_end - i_begin
            xyz = self._waypoints[env_idx, slice(i_begin, i_end), :]  # (T,3)
            lat, lon, alt = ned2geodetic(
                xyz[..., 0:1],
                xyz[..., 1:2],
                xyz[..., 2:3],
                self.lat0,
                self.lon0,
                self.alt0,
            )  # (T,1)
            names = [namefmt.format(i) for i in range(i_begin, i_end)]
            callsigns = [callsignfmt.format(i) for i in range(i_begin, i_end)]
            uids = [get_obj_id(name) for name in names]
            Type = ACMI_Types.Waypoint.value

            for i in range(n):
                rcd.add(
                    rcd.format_unit(
                        id=uids[i],
                        lat=lat[i, 0],
                        lon=lon[i, 0],
                        alt=alt[i, 0],
                        Name=names[i],
                        CallSign=callsigns[i],
                        Type=Type,
                        Next=uids[i + 1] if i < n - 1 else "",
                    )
                )

    def render_object_state(self, object_state: ObjectState):
        self.__objects_states.append(object_state)

    # @np.no_grad()
    def reset(self, env_mask: EnvMaskType | None = None):
        env_mask = self.proc_to_mask(env_mask)
        logr = self.logger
        ie_rcd = self._render_ienv
        DIG_clean(self)
        self._cur_episode[env_mask, 0] += 1
        self._cur_act_step[env_mask, 0] = -1

        rng = self.np_random

        pln = self.aircraft  # @reset
        # reset aircraft model
        pln.reset(env_mask)
        pln.activate(env_mask)

        nenvs_rst = int(np.sum(env_mask))
        if nenvs_rst:
            new_goals, new_goals_index = self.generate_waypoints(
                nenvs=nenvs_rst,
                seed=rng.integers(0, np.iinfo(np.int32).max),
            )
            try:
                self._waypoints[env_mask] = new_goals
                self._waypoint_index[env_mask] = new_goals_index
            except AttributeError:
                self._waypoints = new_goals
                self._waypoint_index = new_goals_index

            if self.DEBUG:
                if env_mask[ie_rcd]:
                    logr.debug(
                        "\n".join(
                            [
                                "waypoints: {}".format(self._waypoints[ie_rcd]),
                                "current wp index: {}".format(
                                    self._waypoint_index[ie_rcd, ..., 0]
                                ),
                            ]
                        )
                    )

            if self._easy_mode:
                # 设置飞机角度为正对着导航点
                # selected_goal = np.take_along_axis(
                #     self._waypoints[env_mask, :, :],
                #     indices=self._waypoint_index[env_mask, :, :],
                #     axis=-2,
                # )  # (...,1,d)
                # selected_goal = selected_goal.squeeze(axis=-2)  # (...,d)
                selected_goal = self.goal_position[env_mask, :]
                aer = ned2aer(selected_goal - pln.position_e()[env_mask, :])
                # print("aer: ", aer)
                azimuth = aer[..., 0:1]
                elevation = aer[..., 1:2]
                _eps = 1e-2
                pln.set_gamma(
                    elevation * affcmb(1 - _eps, 1 + _eps, rng.random(elevation.shape)),
                    env_mask,
                )
                pln.set_chi(
                    azimuth * affcmb(1 - _eps, 1 + _eps, rng.random(azimuth.shape)),
                    env_mask,
                )
                pln._ppgt_rpy_ew2Qew(env_mask)
                pln._propagate(env_mask)

        # reset simulation variables
        self._sim_time_ms[env_mask] = 0
        self.sync_sim_time(env_mask)

        self.__render_t0_ms = -float("inf")

        for rfname, rf in self._reward_fns.items():  # @reset
            rf.reset(self, env_mask)
            self._rew_sum[rfname][env_mask, :] = 0.0

        for tc in self._termination_fns:  # @reset
            tc.reset(self, env_mask)

        if env_mask[ie_rcd]:
            self.render_reset(ie_rcd)
            self.render_waypoints(ie_rcd)

        obs_dict = self.__get_obs(env_mask)
        info = {}
        info[self.INFOKEY_OBS] = obs_dict
        info[self.INFOKEY_RET] = self._rew_sum

        obs_vec = flatten(self._observation_space, obs_dict)
        return deepcopy(obs_vec), deepcopy(info)

    @DIG_property
    def goal_distance(self) -> Float_NDArr:
        """[DIG] 到目标点的剩余距离, shape=(N,1)"""
        return self.goal_aer[..., 2:3]

    @DIG_property
    def goal_cur_reached(self) -> BoolNDArr:
        """[DIG] 最近的导航点是否已抵达, shape=(N,1)"""
        return self.goal_distance < self._tc_p_dmin

    @DIG_property
    def goal_all_reached(self) -> BoolNDArr:
        """[DIG] 是否已抵达全部目标点, shape=(N,1)"""
        return self.goal_cur_reached & (
            self._waypoint_index[..., 0, 0:1] >= self._waypoints_total_num - 1
        )

    @DIG_property
    def goal_is_far_away(self) -> BoolNDArr:
        """[DIG] 是否距离过远, shape=(N,1)"""
        return self.goal_distance > self._tc_p_dmax

    @DIG_property
    def goal_position(self) -> Float_NDArr:
        """[DIG] 当前导航点坐标, shape=(N,3)"""
        goal = np.take_along_axis(
            self._waypoints,
            indices=self._waypoint_index,
            axis=-2,
        )  # (...,1,d)
        goal = goal.squeeze(axis=-2)  # (...,d)
        return goal

    @DIG_property
    def goal_LOS(self) -> Float_NDArr:
        """[DIG] 到当前导航点的视线, shape=(N,3)"""
        los = self.goal_position - self.aircraft.position_e()
        return los

    @DIG_property
    def goal_heading_angle(self) -> Float_NDArr:
        """[DIG] 目标视线的水平航向角, shape=(N,1)"""
        return self.goal_aer[..., 0:1]

    @DIG_property
    def goal_elevation_angle(self) -> Float_NDArr:
        """[DIG] 目标视线的俯仰角, shape=(N,1)"""
        return self.goal_aer[..., 1:2]

    @DIG_property
    def goal_aer(self) -> Float_NDArr:
        """[DIG] 目标视线的AER坐标, shape=(N,3)"""
        _aer = ned2aer(self.goal_LOS)
        return _aer

    @DIG_property
    def goal_AA(self) -> Float_NDArr:
        """[DIG] 目标进入角, in [0,pi],  shape=(N,1)"""
        _aa = np.arccos(
            vec_cosine(
                self.goal_LOS,
                np.zeros_like(self.goal_LOS),  # 静止目标
            )
        )
        return _aa

    @DIG_property
    def goal_ATA(self) -> Float_NDArr:
        """[DIG] 天线指向角, in [0,pi] shape=(N,1)"""
        _ata = np.arccos(
            vec_cosine(
                self.goal_LOS,
                self.aircraft.velocity_e(),
            )
        )
        return _ata

    @DIG_property
    def goal_zem(self) -> Float_NDArr:
        """对目标的零控脱靶量(一个决策周期内), shape=(N,1)"""
        _zem = calc_zem1(
            self.aircraft.position_e() - self.goal_position,
            self.aircraft.velocity_e(),
            tmax=self._agent_step_size_ms * 1e-3,
        )[0]
        return _zem

    # @np.no_grad()
    def step(self, action: np.ndarray):
        info = {}
        step_num = self._agent_step_size_ms // self._sim_step_size_ms
        pln = self.aircraft  # @step
        i_rcd = self._render_ienv  # 选择渲染的环境编号
        rcd = self._acmi_writer  # @step
        self._cur_act_step += 1

        mask = self.proc_to_mask(None)

        for i in range(step_num):
            DIG_clean(self)
            self._sim_time_ms += self._sim_step_size_ms
            self.sync_sim_time(None)

            pln.set_action(action, mask)
            pln.run(mask)

            # 抵达逻辑
            reached = self.goal_cur_reached  # (...,1)
            if np.any(reached):
                self._waypoint_index[reached, ...] = (
                    self._waypoint_index[reached, ...] + 1
                ).clip(0, self._waypoints_total_num - 1)
                DIG_clean(self)

            self.render()

        obs_dict = self.__get_obs(None)

        rew, rews_ = self.__get_rew()
        for rfname, rew_ in rews_.items():
            self._rew_sum[rfname] += rew_

        truncated = self.sim_time_ms >= self.max_sim_time_ms  # @step, (...,1)
        terminated, events = self.__is_terminated()
        # done = terminated | truncated  # (...,1)

        info[self.INFOKEY_OBS] = obs_dict
        info[self.INFOKEY_RET] = self._rew_sum
        info[self.INFOKEY_TERM_EVENT] = events

        obs_vec = flatten(self._observation_space, obs_dict)
        obs_vec = obs_vec.copy()

        if self.DEBUG:
            assert_finite(rew, "reward")

        self.render_events(events)

        obs_out = obs_vec
        return (
            deepcopy(obs_out),
            deepcopy(rew),
            deepcopy(terminated),
            deepcopy(truncated),
            deepcopy(info),
        )

    def render(self):
        i_rcd = self._render_ienv  # 选择渲染的环境编号
        rcd = self._acmi_writer  # @render
        render_t_ms = self.sim_time_ms[i_rcd, ...].item()
        render_mode = self.render_mode  # @render
        if (
            render_mode and render_t_ms - self.__render_t0_ms > self._render_dt_ms
        ):  # 进入新帧
            self.__render_t0_ms = render_t_ms
            rend_flag = True
            # 清空缓存
            msg = rcd.merge()
            if len(msg):
                if render_mode == self.RENDER_MODE_LOCAL:
                    rcd.write_local(msg)
                elif render_mode == self.RENDER_MODE_REMOTE:
                    rcd.write_remote(msg)

            rcd.add(rcd.format_timestamp(render_t_ms * 1e-3))
        else:
            rend_flag = False

        if rend_flag:
            pln = self.aircraft  # @render
            # 目前只渲染单机
            # n = 1 if len(pln.group_shape) == 1 else pln.group_shape[-1]
            rpy_eb = pln.rpy_ew()[i_rcd, :]  # (3,)
            rpy_deg = np.rad2deg(rpy_eb)
            pos_ned = pln.position_e()[i_rcd, :]  # .cpu().numpy()
            lat, lon, alt = ned2geodetic(
                pos_ned[0],
                pos_ned[1],
                pos_ned[2],
                self.lat0,
                self.lon0,
                self.alt0,
            )  # scalar
            rcd.add(
                rcd.format_unit(
                    id=pln.acmi_id[i_rcd, ...].item(),
                    lat=float(lat),
                    lon=float(lon),
                    alt=float(alt),
                    roll=float(rpy_deg[0]),
                    pitch=float(rpy_deg[1]),
                    yaw=float(rpy_deg[2]),
                    Name=pln.acmi_name[i_rcd, ...].item(),
                    Type=pln.acmi_type[i_rcd, ...].item(),
                    Color=pln.acmi_color[i_rcd, ...].item(),
                    CallSign=pln.call_sign[i_rcd, ...].item(),
                    TAS="{:.3f}".format(pln.tas()[i_rcd, ...].item()),
                )
            )

    def render_reset(self, env_index: int = 0):
        writer = self._acmi_writer
        _render_mode = self.render_mode
        if _render_mode == self.RENDER_MODE_LOCAL:
            assert self._render_dir is not None, (
                "render_dir must be set on render_mode",
                _render_mode,
            )
            file_path = self._render_dir / "{}_{}.acmi".format(
                env_index, self._cur_episode[[env_index]].item()
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            writer.reset_local(file_path, reference_time=datetime.now())
        elif _render_mode == self.RENDER_MODE_REMOTE:
            port = self._render_port
            assert port is not None, (
                "render_port must be set on render_mode",
                _render_mode,
            )
            ip = socket.gethostbyname(socket.gethostname())
            port = int(port)
            writer.reset_remote((ip, port), timeout=1.0, reference_time=datetime.now())

    def render_events(self, events: list):
        render_mode = self.render_mode  # @render_events
        if render_mode in [self.RENDER_MODE_LOCAL, self.RENDER_MODE_REMOTE]:
            rcd = self._acmi_writer  # @render_events
            for event in events:
                rcd.add(rcd.format_bookmark(event))
            msg = rcd.merge()
            if render_mode == self.RENDER_MODE_LOCAL:
                rcd.write_local(msg)
            elif render_mode == self.RENDER_MODE_REMOTE:
                rcd.write_remote(msg)

    def _render(self):
        _render_mode = self.render_mode  # @render
        if _render_mode == "tacview_local" and self._render_dir:
            self.__render_count += 1
            # if self.__render_count % 10 == 0:
            # 对数据按照时间进行排序
            self.__objects_states.sort(key=lambda x: x.sim_time_s)
            file_path = self._render_dir / "{}.acmi".format(self.__render_count)

            buf = [
                "FileType=text/acmi/tacview\n",
                "FileVersion=2.2\n",
                "0,ReferenceTime={}Z\n".format(
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
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
        elif _render_mode == "tacview_remote":
            raise NotImplementedError(f"'{_render_mode}' not implemented yet")

    def __get_obs(self, env_mask: EnvMaskType | None) -> dict[str, np.ndarray | Any]:
        env_mask = self.proc_to_mask(env_mask)
        pln = self.aircraft  # @get_obs

        obs_dict: dict[str, np.ndarray | Any] = dict()

        version = self._version  # @get_obs
        if version.startswith("1."):
            # 飞机本体状态
            pln_pos_e = pln.position_e()[env_mask, :]
            obs_dict[self.OBSKEY_POS_E] = pln_pos_e
            obs_dict[self.OBSKEY_TAS] = pln.tas()[env_mask, :]
            obs_dict[self.OBSKEY_QEW] = pln.Q_ew()[env_mask, :]
            obs_dict[self.OBSKEY_MU] = pln.mu()[env_mask, :]

            waypoints = []
            wps_all = self._waypoints[env_mask, :, :]
            wps_idx0 = self._waypoint_index[env_mask, :, :]
            for i in range(self._waypoints_visible_num):
                wp_pos = np.take_along_axis(
                    wps_all,
                    indices=wps_idx0 + i,
                    axis=-2,
                )  # (...,1,d)
                wp_pos = wp_pos.squeeze(axis=-2)  # (...,d)
                wp_los = wp_pos - pln_pos_e

                wp_sd = dict()
                wp_sd[self.OBSKEY_LOS] = wp_los

                # aer = ned2aer(navigation_point-pln.position_g)
                # navigation_point_dict["navigation_point_az"] = aer[env_indices, 0:1]
                # navigation_point_dict["navigation_point_elev"] = aer[env_indices, 1:2]
                # navigation_point_dict["navigation_point_slant_range"] = aer[env_indices, 2:3]/np.norm((self.position_max_limit.to(device=self.device)-self.position_min_limit.to(device=self.device)).to(dtype=np.float32), p=2)

                waypoints.append(wp_sd)
            obs_dict[self.OBSKEY_GOALS] = tuple(waypoints)
        elif version == "2.0":
            raise NotImplementedError("version 2.0 not implemented yet")
            self._make_game_v2
            pln = cast(P6DOFPlane, pln)
            obs_dict["los_b"] = quat_rotate_inv(
                pln.Q_ew(env_mask), self._DIG_goal_los[env_mask]
            )
            if pln._Vmin != pln._Vmax:
                obs_dict["tas"] = pln.tas(env_mask)

            if pln._nx_min != pln._nx_max:
                obs_dict["nx"] = pln._n_w[env_mask, [0]]
            if pln._ny_min != pln._ny_max:
                obs_dict["ny"] = pln._n_w[env_mask, [1]]
            if pln._nz_down_max != pln._nz_up_max:
                obs_dict["nz"] = -pln._n_w[env_mask, [2]]

            obs_dict["dmu"] = pln._dmu[env_mask, :]

        return obs_dict

    def __get_rew(self) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        reward = np.zeros(
            (self.num_envs, 1),
            # device=self.device,
            dtype=self.dtype,  # @get_rew
        )
        meta = {}
        plane = self.aircraft  # @get_rew
        for name, reward_fn in self._reward_fns.items():  # @get_rew
            _reward = reward_fn(self, plane)
            reward += _reward
            meta[name] = _reward
        return reward, meta

    def __is_terminated(self) -> tuple[np.ndarray, list]:
        logr = self.logger  # @get_terminate
        pln = self.aircraft  # @get_terminate
        terminated = np.zeros(
            (self.num_envs, 1),
            dtype=np.bool_,
            #  device=self.device,  # @get_terminate
        )
        evts = []
        ienv = self._render_ienv  # @get_terminate
        for tc in self._termination_fns:  # @get_terminate
            try:
                _terminated = tc(self, pln)
                terminated |= _terminated

                if _terminated[ienv]:
                    logr.debug(
                        "Env[{}]@Ep{} terminated: {}".format(
                            ienv,
                            self._cur_episode[ienv, ...].item(),
                            tc.__class__.__name__,
                        )
                    )
                    evts.append(tc.__class__.__name__)

            except Exception as e:
                print(traceback.format_exc())
                raise e
        return terminated, evts
