# 人类控制测试
from __future__ import annotations
from functools import partial
import logging
from matplotlib.patches import Ellipse
import pymap3d
from numpy.linalg import norm
from numpy.typing import NDArray

_DEBUG = True


def _setup():  # 确保项目根节点在 sys.path 中
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


ROOT = _setup()

from enum import Enum
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import threading
import time
from typing import Callable, Dict, List, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
from pynput import keyboard
from envs_np.simulators.decoy.base_decoy import BaseDecoy
from envs_np.simulators.proto4model import BaseModel
from envs_np.simulators.missile import BaseMissile
from envs_np.utils.tacview import TacviewRecorder, ACMI_Types, format_id
from envs_np.utils.math_np import calc_zem1, rpy2mat
from envs_np.utils import math_np as math_ext
from envs_np.utils.math_np import bkbn as np
from envs_np.simulators.aircraft.p6dof import P6DOFPlane as Plane_
from envs_np.simulators.missile.p6dof import (
    P6DOFMissile as Missile,
)
from envs_np.simulators.proto4model import BaseModel
from envs_np.simulators.decoy.dof0decoy import DOF0BallDecoy as Decoy
from envs_np.utils import log_ext
from envs_np.utils.time_ext import Timer_Pulse

ACT_IDX_NX = 0
ACT_IDX_NY = 1
ACT_IDX_NZ = 2
ACT_IDX_ROLL = 3
ACT_IDX_DECOY = 4
ACT_IDX_MSL = 5
ACT_KEYS = [
    ACT_IDX_NX,
    ACT_IDX_NZ,
    ACT_IDX_ROLL,
    ACT_IDX_DECOY,
    ACT_IDX_MSL,
    ACT_IDX_NY,
]


class ACMI_Color(Enum):
    Red = "Red"
    Blue = "Blue"
    Green = "Green"


@dataclass
class ACMI_Info:
    id: int
    Color: str = ACMI_Color.Red.value
    CallSign: str = ""
    Type: str = ""
    Name: str = ""
    Parent: str = ""

    @property
    def acmi_id(self) -> str:
        """ACMI Object ID"""
        u = format_id(self.id)
        u = "0x" + u
        return u


@dataclass
class PlaneAction:
    nx_cmd: float = 0
    ny_cmd: float = 0
    nz_cmd: float = 0
    droll_cmd: float = 0


@dataclass
class _Keymap:
    keyname: str

    def equals(self, key: "_Keymap"):
        return self.keyname == key.keyname

    @classmethod
    def fromKey(cls, key: Union[keyboard.Key, keyboard.KeyCode, str]):
        if isinstance(key, keyboard.Key):
            return cls(key.name)
        if isinstance(key, keyboard.KeyCode):
            if key.char is not None:
                return cls(key.char)
            if key.vk is not None:
                return cls(str(key.vk))
            raise NotImplementedError
        if isinstance(key, str):
            return cls(key)
        raise TypeError(f"Invalid key type: {type(key)}")

    def __str__(self):
        return self.keyname


# 键位绑定
HOTKEY_NX_P = _Keymap("t")
HOTKEY_NX_N = _Keymap("g")
HOTKEY_NY_P = _Keymap("h")
HOTKEY_NY_N = _Keymap("f")
HOTKEY_NZ_P = _Keymap("s")
HOTKEY_NZ_N = _Keymap("w")
HOTKEY_ROLL_P = _Keymap("d")
HOTKEY_ROLL_N = _Keymap("a")
HOTKEY_DMSL = _Keymap("x")
HOTKEY_DECOY = _Keymap("v")
HOTKEY_ESC = _Keymap.fromKey(keyboard.Key.esc)
HOTKEY_PAUSE = _Keymap("p")  # 暂停仿真
HOTKEY_RESET = _Keymap("r")
# Tacview 热键冲突警告:
# 方向键被用于播放条控制, ijkl 用于视角移动,
# +-qe 视角缩放,
# 空格暂停播放(只是Tacview本身暂停，但是不能同步到本程序,仿真可能依然推进)


def los_is_blocked_np(los: np.ndarray, rthres: np.ndarray):
    """
    只能判断被一个球完全阻挡,不能判断被开覆盖阻挡
    Args:
        los: 视线组 shape=(...,n,2|3)
        rthres: 视线终点的阻挡半径 float|shape=(...,n,1)
    Returns:
        block: 是否被遮挡 shape=(...,n,1)
    """
    assert isinstance(los, (np.ndarray))
    assert isinstance(rthres, (np.ndarray))
    assert len(los.shape) == len(rthres.shape), (
        "expect len(los.shape) == len(rthres.shape), got",
        len(los.shape),
        len(rthres.shape),
    )
    assert rthres.shape[-1] == 1, (
        "expect rthres.shape[-1] == 1, got",
        rthres.shape[-1],
    )
    _0f_d = np.zeros_like(los)
    R_ = norm(los, axis=-1, keepdims=True)  # (...,n,1)
    case1 = R_ > rthres  # (...,n,1)
    case1__j = np.expand_dims(case1, -3)  # (...,1,n,1)
    case1_i_ = np.expand_dims(case1, -2)  # (...,n,1,1)
    # 核心问题
    # max_\{\cos\angle(p,p_i)||p-p_j|\leq r_j\}\leq |p_i|/\sqrt{r_i^2+|p_i|^2}
    # s_j^2||p_i|/r_j|^2=s_j^2+L_j^2
    # L_j^2+d_{ij}^2=|p_i|^2
    # => s_j^2(|p_j|^2-r_j^2)=r_j^2*(p_i^2-d_{ij}^2)
    r_i_ = np.expand_dims(rthres, axis=-2)  # (...,n,1,1)
    R_i_ = np.expand_dims(R_, axis=-2)  # (...,n,1,1)
    r__j = np.expand_dims(rthres, axis=-3)  # (...,1,n,1)
    R__j = np.expand_dims(R_, axis=-3)  # (...,1,n,1)

    d_ij = calc_zem1(los, -los)[0]  # (...,n,n,1)
    # [...,i,j,:]:= p_i 到射线 p_j*[0,\infty) 的投影距离
    assert np.isfinite(d_ij).all(), "d_ij is not finite"
    s__j2 = (r__j**2 * ((R_i_) ** 2 - d_ij**2)) / (R__j**2 - r__j**2 + ~case1__j)
    rst1 = case1__j & (r_i_ > s__j2 + d_ij)  # (...,n,n,1)
    rst2 = (~case1_i_) & case1__j  # (...,n,n,1)

    bij = rst1 | rst2  # (...,n,n,1) i 遮挡 j
    for i in range(bij.shape[0]):  # 自己对自己不构成遮挡
        bij[..., i, i, 0] = False
    blk = np.any(bij, axis=-3)  # (...,n,1)
    blk = cast(NDArray[np.bool_], blk)
    return blk


def affcmb(a, b, t) -> np.ndarray:
    a = np.asarray(a)
    return a + (b - a) * t


EllipsisType = type[Ellipsis]


def index2mask(
    shape: Tuple[int, ...], *index: Union[int, slice, Sequence[int], EllipsisType]
) -> math_ext.BoolNDArr:
    mask = np.zeros(shape, dtype=np.bool_)
    mask[*index] = True
    return mask


def missile_filt_units(
    self_loc: np.ndarray,
    targ_loc: np.ndarray,
    targ_radius: float | np.ndarray,
):
    """
    可见目标过滤
    Args:
        self_loc: 自身位置 shape=(n,d)
        targ_loc: 目标位置 shape=(m,d)
        targ_radius: 目标体积半径 float| shape=(m,1)|(m,)
    """
    assert len(self_loc.shape) == 2, (
        "expect self_loc be 2D array, but got",
        self_loc.shape,
    )
    assert len(targ_loc.shape) == 2, (
        "expect targ_loc be 2D array, but got",
        targ_loc.shape,
    )
    n, d = self_loc.shape
    assert targ_loc.shape[-1] == d, (
        "expect targ_loc.shape[-1] == d, got",
        targ_loc.shape[-1],
    )
    m = targ_loc.shape[0]
    self_loc = self_loc.reshape(-1, 1, d)
    targ_loc = targ_loc.reshape(1, -1, d)
    targ_radius = np.ravel(targ_radius)  # (m|1,)
    assert len(targ_radius) in (1, m), (
        f"expect len(targ_radius) in (1,{m}), got",
        len(targ_radius),
    )
    targ_radius = np.reshape(targ_radius, (1, m, 1))  # (1,m|1,1)
    targ_radius = np.broadcast_to(targ_radius, (n, m, 1))  # (n,m,1)

    los = targ_loc - self_loc  # (n,m,3)
    los = los.reshape(-1, d)  # (n*m,3)
    br = targ_radius.reshape(-1, 1)  # (n*m,1)

    valid = np.logical_not(los_is_blocked_np(los, br))  # (...,n*m,1)
    valid = valid.reshape(n, m)
    return valid


def reuse_index(
    group: BaseModel,
):
    tag = group.is_inactive().reshape(-1, 1)  # (B,1)
    if tag.any():
        idxs = np.where(tag)[0]  # (b,)
        index = int(np.random.choice(idxs))
        return index
    else:
        return -1


def arrange_id(
    group: BaseModel,
):
    for i in range(1, group.group_shape[-1]):
        group.acmi_id[..., i, 0] = group.acmi_id[..., i - 1, 0] + 1


def missile_regen(
    grp: Missile,
    new_id: int,
    target_pos: np.ndarray,
    hmin: float,
    hmax: float,
    alt0: float,
    unit_tc: Callable[[BaseModel, int], List[Sequence[str]]],
    ntry_tol=1000,
    rng: np.random.Generator = np.random.default_rng(),
    parent_pos: np.ndarray | None = None,
    parent_rpy: np.ndarray | None = None,
    parent_vel: np.ndarray | None = None,
    echo=True,
    recorder: TacviewRecorder | None = None,
):
    index: int = reuse_index(grp)
    if index == -1:
        return index

    grp.acmi_id[index, 0] = new_id
    grp.call_sign[index, 0] = "M{}".format(format_id(new_id))

    # lat0, lon0, alt0 = target.blh0()
    np_float = grp.dtype
    det_rmax = grp._det_rmax
    dst_index = index2mask(grp.group_shape, index)
    # init
    if parent_pos is None:
        assert target_pos is not None, "target_pos must be provided on no parent"
        tgt_ned_tsr = np.asarray(
            target_pos,
            dtype=np_float,
        ).reshape(
            -1, grp.position_e().shape[-1]
        )  # (-1,3)
        tgt_z = tgt_ned_tsr[0, [-1]].item()
        tgt_alt = alt0 - tgt_z
        assert len(tgt_ned_tsr.shape) == 2
        msg_target = "target_pos={}".format(tgt_ned_tsr.tolist())
    else:
        msg_target = ""
    # try
    for ntry in range(ntry_tol):
        msg_parent = ""
        if parent_pos is None:  # 无载机位置
            assert target_pos is not None
            losz = affcmb(hmin, hmax, rng.random()) - tgt_alt
            rxy = min(float(affcmb(0.1, 0.5, rng.random()) * det_rmax), 5000)
            azi = affcmb(
                0,
                2 * np.pi,
                rng.random(),
            )
            losx = rxy * np.cos(azi)
            losy = rxy * np.sin(azi)
            elv = np.arctan2(-losz, rxy)
            rpy_np = np.asarray([0, elv, azi])  # (3,)
            los_e = np.asarray([losx, losy, losz])  # (3,)
            los_tsr = np.asarray(
                los_e,
                dtype=np_float,
            ).reshape(
                1, -1
            )  # (1,3)
            rpy_tsr = np.asarray(
                rpy_np,
                dtype=np_float,
            ).reshape(
                1, -1
            )  # (1,3)

            tas = np.asarray(
                affcmb(grp._Vmin, grp._Vmax, rng.random()),
                dtype=np_float,
            )
            assert not np.isnan(los_tsr).any()
            assert not np.isnan(tgt_ned_tsr).any()
            grp.set_ic_pos_e(tgt_ned_tsr - los_tsr, dst_index)
        else:  # 载机发射模式
            assert parent_pos is not None
            assert parent_rpy is not None
            assert parent_vel is not None
            parent_id_ = int(grp.acmi_parent.reshape(-1, 1)[index, [0]].item())
            msg_parent = f"parent={format_id(parent_id_)}"
            ego_pos_ = np.asarray(
                parent_pos,
                dtype=np_float,
            ).reshape(1, -1)
            ego_vel_ = np.asarray(
                parent_vel,
                dtype=np_float,
            ).reshape(1, -1)
            rpy_tsr = np.asarray(
                parent_rpy,
                dtype=np_float,
            ).reshape(1, -1)
            tas = np.clip(
                math_ext.norm(ego_vel_, 2, -1, keepdims=True), grp._Vmin, grp._Vmax
            )
            assert not np.isnan(ego_pos_).any()
            grp.set_ic_pos_e(ego_pos_, dst_index)

        assert not np.isnan(rpy_tsr).any()
        assert not np.isnan(tas).any()
        grp.set_ic_rpy_ew(rpy_tsr, dst_index)
        grp.set_ic_tas(tas, dst_index)

        grp.launch(dst_index)
        # assert norm(msl._Reb - Reb) < 1e-2, f"Reb={Reb} rpy={rpy}, Reb={msl._Reb }"
        tc_evt = unit_tc(grp, index)
        if len(tc_evt):
            grp.set_status(grp.STATUS_INACTIVE, dst_index)  # 复位
            continue
        break
    else:
        raise ValueError("max try reached@missile_reset")
    if echo and index >= 0:
        assert recorder is not None
        msg = ";".join(
            [
                row
                for row in [
                    "create missile {} @index={}".format(
                        format_id(int(grp.acmi_id.ravel()[index].item())),
                        index,
                    ),
                    "pos_e={}".format(
                        grp.position_e().reshape(-1, 3)[index, :].tolist()
                    ),
                    "vel_e={}".format(
                        grp.velocity_e().reshape(-1, 3)[index, :].tolist()
                    ),
                    msg_parent,
                    msg_target,
                ]
                if len(row)
            ]
        )
        # print(msg)
        grp.logger.info(msg)
        recorder.add(recorder.format_bookmark(msg))
    return index


def decoy_regen(
    grp: Decoy,
    new_id: int,
    parent_id: np.ndarray,
    parent_pos: np.ndarray,
    parent_vel: np.ndarray,
    unit_tc: Callable[[BaseModel, int], List[Sequence[str]]],
    ntry_tol=1000,
    rng: np.random.Generator = np.random.default_rng(),
    echo=True,
    recorder: TacviewRecorder | None = None,
):
    index: int = reuse_index(grp)
    if index == -1:
        return index
    grp.acmi_id[..., index, 0] = new_id
    grp.call_sign[..., index, 0] = f"D{format_id(new_id)}"
    grp.acmi_parent[..., index, 0] = parent_id
    # group.acmi_type[index] = m_info.Type
    # group.acmi_name[index] = m_info.Name
    # lat0, lon0, alt0 = target.blh0()
    np_float = grp.dtype
    device = grp.device
    prnt_pos_tsr = np.asarray(
        parent_pos,
        dtype=np_float,
    ).reshape(
        1, -1
    )  # (1,3)
    prnt_vel_tsr = np.asarray(
        parent_vel,
        dtype=np_float,
    ).reshape(
        1, -1
    )  # (1,3)
    prnt_id = int(parent_id.item())
    dst_index = index2mask(grp.group_shape, index)
    for ntry in range(ntry_tol):
        grp._pos_e[dst_index, :] = prnt_pos_tsr
        grp._vel_e[dst_index, :] = prnt_vel_tsr * (
            1
            - 0.1 * (2 * np.asarray(rng.random(prnt_vel_tsr.shape), dtype=np_float) - 1)
        )
        grp.reset(dst_index)
        grp.activate(dst_index)

        # assert norm(msl._Reb - Reb) < 1e-2, f"Reb={Reb} rpy={rpy}, Reb={msl._Reb }"
        tc_evt = unit_tc(grp, index)
        if len(tc_evt):
            grp.set_status(grp.STATUS_INACTIVE, dst_index)
            continue
        break
    else:
        raise ValueError("max try reached@decoy_reset")
    if echo:
        assert recorder is not None
        if index >= 0:
            msg = ";".join(
                row
                for row in [
                    "create decoy {} @index={}".format(
                        format_id(int(grp.acmi_id[..., index, 0].item())),
                        index,
                    ),
                    "pos_e={}".format(grp.position_e()[..., index, :].tolist()),
                    "vel_e={}".format(grp.velocity_e()[..., index, :].tolist()),
                    "parent={}".format(format_id(prnt_id)),
                ]
                if len(row)
            )
            recorder.add(recorder.format_bookmark(msg))
            grp.logger.info(msg)
    return index


def merge_info4observe(
    ego: BaseModel,
    groups: List[BaseModel],
    use_visual_block=False,
):
    m = ego.batch_size
    ego_alv = math_ext.unsqueeze(ego.is_alive(), -3)  # (...,1,m,1)
    ego_pos = math_ext.unsqueeze(ego.position_e(), -3)  # (...,1,m,d)
    ego_id = math_ext.unsqueeze(ego.acmi_id, -3)  # (...,1,m,1)
    ego_clr = math_ext.unsqueeze(ego.acmi_color, -3)  # (...,1,m,1)
    tmp_masks = []
    tmp_pos = []
    tmp_vel = []
    tmp_r = []
    tmp_idtt = []
    tmp_id = []
    for grp in groups:
        grp_ids = grp.acmi_id  # (...,n,1,)
        grp_alv = math_ext.unsqueeze(grp.is_alive(), -2)  # (...,n,1,1)
        grp_clr = np.expand_dims(grp.acmi_color, -2)  # (...,n,1,1)
        is_enm = np.asarray(
            ego_clr != grp_clr,
            dtype=np.bool_,
            # device=ego.device,
        )  # (...,n,m,1)
        not_self = np.asarray(
            np.expand_dims(grp_ids, -2) != ego_id,
            dtype=np.bool_,
            # device=ego.device,
        )  # (...,n,m,1)
        vld21 = (ego_alv & grp_alv) & is_enm & not_self  # (...,n,m,1)
        tmp_masks.append(vld21)
        tmp_pos.append(grp.position_e())  # (...,n,3)
        tmp_vel.append(grp.velocity_e())  # (...,n,3)
        if use_visual_block:
            tmp_r.append(math_ext.unsqueeze(grp.vis_radius(), -2))  # (...,n,1,1)
        tmp_idtt.append(not_self)
        tmp_id.append(grp_ids)

    mask = math_ext.cat(tmp_masks, axis=-3)  # (...,N,m,1)
    N = mask.shape[0]
    id2 = math_ext.cat(tmp_id, axis=-2)  # (...,N,1)
    pos2 = math_ext.cat(tmp_pos, axis=-2)  # (...,N,3)
    vel2 = math_ext.cat(tmp_vel, axis=-2)  # (...,N,3)
    if use_visual_block:
        visr2 = math_ext.cat(tmp_r, axis=-3)  # (...,N,1,1)
        not_self = math_ext.cat(tmp_idtt, axis=-3)  # (...N,m,1)
        visr12 = np.broadcast_to(visr2, mask.shape)  # (...,N,m,1)
        los12 = math_ext.unsqueeze(pos2, -2) - ego_pos  # (...,N,m,3)
        # los12 += not_self  # 距离掩码修正, 否则会在除0

        # blk = los_is_blocked_np(
        #     los12.flatten(-3, -2),
        #     visr12.flatten(-3, -2),
        # )  # (...,N*m,1)
        # blk = np.asarray(blk, dtype=np.bool, device=mask.device)  # (...,N*m,1)
        valid = math_ext.los_is_visible(
            math_ext.flatten(los12, -3, -2),
            math_ext.flatten(visr12, -3, -2),
            math_ext.flatten(mask, -3, -2),
        )[
            0
        ]  # (...,N*m,1)
        mask = math_ext.unflatten(valid, -2, (N, m))  # (...,N,m,1)

    mask = mask.squeeze(-1)  # (...,N,m)
    return (
        pos2,  # (...,N,d)
        vel2,  # (...,N,d)
        mask,  # (...,N,m)
        id2,  # (...,N,1)
    )


def render_groups(
    groups: List[BaseModel],
    recorder: TacviewRecorder,
):
    for grp in groups:
        N = grp.batch_size
        idxs_alv = np.where(grp.is_alive().reshape(N, 1))[0]
        if len(idxs_alv):
            p_ned = cast(
                np.ndarray, grp.position_e().reshape(N, -1)[idxs_alv, :]
            )  # (n,3)
            tas = cast(np.ndarray, grp.tas().reshape(N, -1)[idxs_alv, :])  # (n,1)
            blh0 = cast(np.ndarray, grp.blh0().reshape(N, -1)[idxs_alv, :])  # (n,3)
            assert np.isfinite(p_ned).all(), "nan or inf@p_ned"
            assert np.isfinite(tas).all(), "nan or inf@tas"
            assert np.isfinite(blh0).all(), "nan or inf@blh0"
            lat0, lon0, alt0 = np.split(blh0, 3, axis=-1)
            loc_n, loc_e, loc_d = np.split(p_ned, 3, axis=-1)
            lat, lon, alt = pymap3d.ned2geodetic(
                loc_n, loc_e, loc_d, lat0, lon0, alt0
            )  # (n,1)
            rpy = None
            if rpy is None:
                try:
                    rpy = grp.rpy_eb()
                    rpy = rpy.reshape(N, -1)[idxs_alv, :]  # (n,3)
                except Exception:
                    pass
            if rpy is None:
                try:
                    rpy = grp.rpy_ew()
                    rpy = rpy.reshape(N, -1)[idxs_alv, :]  # (n,3)
                except Exception:
                    pass
            if rpy is not None:
                rpy = cast(np.ndarray, rpy)
                rpy_deg = np.rad2deg(rpy)

            for i, i_in_grp in enumerate(cast(List[int], idxs_alv.tolist())):
                if rpy is None:
                    roll_i = pitch_i = yaw_i = None
                else:
                    roll_i, pitch_i, yaw_i = rpy_deg[i]
                recorder.add(
                    recorder.format_unit(
                        int(grp.acmi_id[i_in_grp, 0].item()),
                        lat=lat[i, 0],
                        lon=lon[i, 0],
                        alt=alt[i, 0],
                        roll=roll_i,
                        pitch=pitch_i,
                        yaw=yaw_i,
                        Name=str(grp.acmi_name.reshape(N, -1)[i_in_grp, 0]),
                        Color=str(grp.acmi_color.reshape(N, -1)[i_in_grp, 0]),
                        TAS="{:.1f}".format(tas[i, 0]),
                        Type=str(grp.acmi_type.reshape(N, -1)[i_in_grp, 0]),
                        CallSign=str(grp.call_sign.reshape(N, -1)[i_in_grp, 0]),
                        Parent=str(grp.acmi_parent.reshape(N, -1)[i_in_grp, 0]),
                    )
                )


def try_kill_groups(
    ego: BaseMissile,
    groups: List[BaseModel],
    recorder: TacviewRecorder,
):
    m = ego.acmi_id.shape[-2]
    alv1 = math_ext.unsqueeze(ego.is_dying(), -2)  # (...,m,1,1)
    ids1 = math_ext.unsqueeze(ego.acmi_id, -2)  # (...,m,1,1)
    p1 = math_ext.unsqueeze(ego.position_e(), -2)  # (...,m,1,d)
    kr = math_ext.unsqueeze(
        (np.zeros_like(ego.tas()) + ego.kill_radius), -2
    )  # (...,m,1,1)
    #
    envmidx = [0] * len(ego.acmi_id.shape[:-2])
    msgs: List[str] = []
    for i1 in range(m):
        if not alv1[*envmidx, i1, [0]].item():
            continue
        idh = format_id(int(ids1[*envmidx, i1, [0]].item()))
        md = ego.miss_distance[*envmidx, i1, [0]].item()
        msg = f"{idh} exploding, MD={md:.1f}"
        recorder.add(recorder.format_bookmark(msg))
        msgs.append(msg)
    #
    for grp in groups:
        n = grp.acmi_id.shape[-2]
        ids2 = math_ext.unsqueeze(grp.acmi_id, -3)  # (...,1,n,1)
        alv2 = math_ext.unsqueeze(grp.is_alive(), -3)  # (...,1,n,1)
        p2 = math_ext.unsqueeze(grp.position_e(), -3)  # (...,1,n,d)
        is_self = ids1 == ids2  # (...,m,n,1)
        dij = math_ext.norm_(p1 - p2, dim=-1, keepdim=True)  # (...,m,n,1)
        kill_evt = (~is_self) & (alv1 & alv2) & (dij < kr)  # (...,m,n,1) 有效击杀
        kill_tag_ = kill_evt.any(-3)  # (...,n,1)
        grp.status[...] = np.where(kill_tag_, grp.STATUS_DYING, grp.status)
        #
        for i1 in range(m):
            id1 = format_id(int(ego.acmi_id[*envmidx, i1, 0].item()))
            for i2 in range(n):
                if not kill_evt[*envmidx, i1, i2].item():
                    continue
                id2 = format_id(int(grp.acmi_id[*envmidx, i2, 0].item()))
                evt = "{} hit {}".format(id1, id2)
                msgs.append(evt)
                recorder.add(recorder.format_bookmark(evt))
    if len(msgs):
        print(*msgs, sep="\n")


def reset_id0(fri_pln_id_next: int = 0x0A0000):
    fri_dcy_id_next = fri_pln_id_next + 0x001000
    fri_msl_id_next = fri_pln_id_next + 0x002000
    enm_pln_id_next = fri_pln_id_next + 0x010000
    enm_dcy_id_next = enm_pln_id_next + 0x001000
    enm_msl_id_next = enm_pln_id_next + 0x002000
    return (
        fri_pln_id_next,
        fri_dcy_id_next,
        fri_msl_id_next,
        enm_pln_id_next,
        enm_dcy_id_next,
        enm_msl_id_next,
    )


def game_run(
    get_action: Callable[[], List[int]],  # ->[-1,1]
    f_stop: Callable[[], bool],
    f_trunc: Callable[[], bool] | None = None,
    f_is_paused: Callable[[], bool] | None = None,
    f_set_pause: Callable[[bool], None] | None = None,
    tacview_full=True,
    enm_msl_N=2,
    fri_msl_N=2,
    decoys_N=20,
    msl_simple_sens=False,  # 1->导弹开上帝视角, 0->导弹视野受限
):
    fri_pln_id_next = 0xA0000
    (
        fri_pln_id_next,
        fri_dcy_id_next,
        fri_msl_id_next,
        enm_pln_id_next,
        enm_dcy_id_next,
        enm_msl_id_next,
    ) = reset_id0(fri_pln_id_next)
    g = 9.80665  # m/s^2
    Vmin = 100.0
    # 飞机能力参数
    nx_max = 1.5
    nx_min = -0.5
    nz_dmax = 0.5
    nz_umax = 10.0
    dmu_max = (2 * math.pi) / 3  # 3s 转 360°
    pln_V_lb = Vmin
    pln_V_ub = 334 * 1.5

    use_float64 = True
    ftype = np.float64 if use_float64 else np.float32
    device = "cpu"
    if device != "cpu":
        print("CUDA 在小规模仿真里不够快!")

    # 干扰参数
    pln_objr = 10.0
    dec_objr = 30.0
    msl_objr = 0.5

    # 导弹能力参数
    msl_det_rmax = 50000
    msl_det_half_angle = 180
    msl_det_fov_deg = 2 * msl_det_half_angle
    msl_trk_fov_deg = 360
    msl_V0 = 600
    msl_Vmax = pln_V_ub * 2
    msl_Vmin = 200
    msl_nmax = (msl_Vmax * (math.tau / 4)) / g

    _FILE = Path(__file__).resolve()
    runs_dir = _FILE.parents[1] / "tmp"
    acmi_dir = runs_dir
    tel_addr = ("localhost", 21000)

    fri_color = ACMI_Color.Red.value
    enm_color = ACMI_Color.Blue.value

    simdt_ms = 40
    simdt_sec = simdt_ms * 1e-3

    rng = np.random.default_rng()

    lat0 = 30.0
    lon0 = 105.0
    alt0 = 0
    hmax = 25000
    hmin = 100
    simt_tol = 20 * 60  # 全局仿真时限
    msl_simt_tol = 30.0
    dcy_simt_tol = 10.0

    logr = log_ext.reset_logger(
        __name__ + "_game",
        level=log_ext.DEBUG,
        file_path=str(runs_dir / "log.log"),
        file_append=False,
    )
    pln_logr = log_ext.reset_logger(
        __name__ + "_pln",
        level=log_ext.DEBUG,
        file_path=str(runs_dir / "pln.log"),
        file_append=False,
    )
    enm_msl_logr = log_ext.reset_logger(
        logr.name + ("_msl"),
        level=log_ext.DEBUG,
        file_path=str(runs_dir / "msl.log"),
        file_append=False,
    )
    decoy_logr = log_ext.reset_logger(
        logr.name + ("_decoy"),
        level=log_ext.DEBUG,
        file_path=str(runs_dir / "decoy.log"),
        file_append=False,
    )
    pln_info = ACMI_Info(fri_pln_id_next, Color=fri_color, Name="J-20")
    pln_info.CallSign = f"F{pln_info.acmi_id}"

    use_plane = True
    if use_plane:
        agent = Plane_(
            # tas=240,
            # rpy_ew=0,
            sim_step_size_ms=simdt_ms,
            use_gravity=False,
            Vmin=pln_V_lb,
            Vmax=pln_V_ub,
            nx_max=nx_max,
            nx_min=nx_min,
            nz_up_max=nz_umax,
            nz_down_max=nz_dmax,
            dmu_max=dmu_max,
            g=g,
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            acmi_id=fri_pln_id_next,
            acmi_name=pln_info.Name,
            acmi_color=pln_info.Color,
            acmi_type=pln_info.Type,
            call_sign=pln_info.CallSign,
            vis_radius=pln_objr,
            dtype=ftype,
        )
    else:
        agent = Missile(
            group_shape=1,
            sim_step_size_ms=simdt_ms,
            acmi_id=fri_pln_id_next,
            Vmin=pln_V_lb,
            Vmax=pln_V_ub,
            dtype=ftype,
            acmi_color=pln_info.Color,
            acmi_name="AIM-9M",
            acmi_type=ACMI_Types.Missile.value,
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
        )
    agent.DEBUG = _DEBUG
    agent.set_ic_tas((pln_V_lb + (pln_V_ub - pln_V_lb) * 0.8), None)
    agent.set_ic_rpy_ew(0, None)
    agent.set_ic_pos_e(
        np.asarray([0, 0, -(hmax + hmin) * 0.5], dtype=ftype).reshape(1, 3),
        None,
    )
    agent.logger = pln_logr
    for iteam, msl_N in enumerate(
        [fri_msl_N, enm_msl_N],
    ):
        msl_color = fri_color if iteam == 0 else enm_color
        msls = Missile(
            sim_step_size_ms=simdt_ms,
            acmi_id=fri_msl_id_next if iteam == 0 else enm_msl_id_next,
            group_shape=msl_N,
            Vmin=msl_Vmin,
            Vmax=msl_Vmax,
            det_rmax=msl_det_rmax,
            det_fov_deg=msl_det_fov_deg,
            trk_fov_deg=msl_trk_fov_deg,
            dtype=ftype,
            acmi_color=msl_color,
            acmi_name="AIM-9M",
            acmi_type=ACMI_Types.Missile.value,
            acmi_parent=pln_info.id if iteam == 0 else -1,
            call_sign="",
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            vis_radius=msl_objr,
            debug=True,
        )
        msls_info: List[ACMI_Info] = []
        for j in range(msl_N):
            msk = index2mask(msls.group_shape, j)
            msls.set_ic_tas(msl_V0, msk)
            msls.set_ic_rpy_ew(0, msk)
            msls.set_ic_pos_e(agent._pos_e[0, :] + 1000, msk)
            msls.reset(msk)
            if iteam == 0:
                fri_msl_id_next += 1
                newid = fri_msl_id_next
            else:
                enm_msl_id_next += 1
                newid = enm_msl_id_next
            msls_info.append(
                ACMI_Info(
                    newid,
                    Color=msls.acmi_color[j, 0],
                    Name=msls.acmi_name[j, 0],
                    Type=msls.acmi_type[j, 0],
                    Parent=format_id(int(msls.acmi_parent[j, [0]].item())),
                )
            )
            msls.acmi_id[j, 0] = newid

        if iteam == 0:
            fri_msl = msls
            fri_msl.DEBUG = False
        else:
            enm_msl = msls
            enm_msl.DEBUG = _DEBUG

    enm_msl.logger = enm_msl_logr

    decoys = Decoy(
        sim_step_size_ms=simdt_ms,
        acmi_id=fri_dcy_id_next,
        group_shape=decoys_N,
        vis_radius=dec_objr,
        effect_duration=30.0,
        dtype=ftype,
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
        acmi_color=fri_color,
        acmi_name=ACMI_Types.FlareDecoy.name,
        acmi_type=ACMI_Types.FlareDecoy.value,
        acmi_parent=agent.acmi_id,
    )
    decoys.logger = decoy_logr
    decoys.DEBUG = _DEBUG
    for j in range(decoys_N):
        msk = index2mask(decoys.group_shape, j)
        decoys._pos_e[j, :] = agent._pos_e[0, :]
        decoys._vel_e[j, :] = agent._vel_e[0, :]
        decoys.reset(msk)

        decoys.acmi_name[j, 0] = "FlareDecoy"
        decoys.acmi_type[j, 0] = ACMI_Types.FlareDecoy.value

        fri_dcy_id_next += 1

    # 挂载限制(<0不限制)
    nlim_msl = -1
    nlim_dec = -1

    def unit_tc(
        unit: BaseModel,
        idx: int = 0,
        Vmin: float = Vmin,
        tmax: float = simt_tol,
    ) -> List[Sequence[str]]:
        rst = []
        if unit.is_alive().reshape(-1, 1)[idx, 0].item():
            alt = unit.altitude_m().reshape(-1, 1)[idx, 0].item()
            if alt <= hmin:
                rst.append(("fly too low", f"alt={alt:.1f}"))
            elif alt > hmax:
                rst.append(("fly too high", f"alt={alt:.1f}"))
            tas = unit.tas().reshape(-1, 1)[idx, 0].item()
            if tas <= Vmin:
                rst.append(("fly too slow", f"tas={tas:.1f}"))
            t = unit.sim_time_s().reshape(-1, 1)[idx, 0].item()
            if t > tmax:
                rst.append(("life time out", f"t={t:.1f}"))
        return rst

    recorder = TacviewRecorder()

    ss_reset = 0
    ss_step = 1
    ss_pause = 2
    ss = ss_reset
    action0 = PlaneAction()
    tmr_fresh = Timer_Pulse(simdt_sec)
    tmr_fps = Timer_Pulse()
    tmr_echo = Timer_Pulse(1.0)
    print(f"若实际FPS<={1/simdt_sec:.1f}, 则不能保证实时仿真")
    sim_epi = 0  # episode数
    sim_k = 0  # 仿真步数
    max_enm_missiles = enm_msl.batch_size  # 同时存在的最大数量

    def get_simt():
        return sim_k * simdt_sec

    unit_groups: List[BaseModel] = [
        agent,  # @group
        fri_msl,  # @group
        enm_msl,  # @group
        decoys,  # @group
    ]
    while True:
        if f_stop():
            break
        try:
            if ss == ss_reset:  # @reset
                print(  # @hint
                    f"{HOTKEY_ESC} 连按退出",
                    f"{HOTKEY_RESET} 连按重启",
                    f"{HOTKEY_NX_N} / {HOTKEY_NX_P} 切向过载",
                    f"{HOTKEY_NY_N} / {HOTKEY_NY_P} 侧向过载",
                    f"{HOTKEY_NZ_N} / {HOTKEY_NZ_P} 俯仰过载",
                    f"{HOTKEY_ROLL_N} / {HOTKEY_ROLL_P} 滚转角速度",
                    f"{HOTKEY_PAUSE} 暂停/继续",
                    f"{HOTKEY_DECOY} 诱饵弹",
                    f"{HOTKEY_DMSL} 拦截弹",
                    sep="\n",
                )

                acmitime = datetime.now()
                acmitime = datetime(acmitime.year, acmitime.month, acmitime.day)
                if tacview_full:
                    rst = recorder.reset_remote(
                        addr=tel_addr, reference_time=acmitime, timeout=5.0
                    )
                    if not rst:
                        print("reset_remote rfailed")
                        # _arg = input("skip connection? ([y]/n)")
                        # if _arg.lower() == "n":
                        #     continue
                        # else:
                        #     pass
                        continue
                acmi_fn = runs_dir / f"{sim_epi}.acmi"
                acmi_fn.parent.mkdir(exist_ok=True, parents=True)
                recorder.reset_local(acmi_fn, reference_time=acmitime)

                ncomsum_msl = 0
                ncomsum_dec = 0
                ndoge_msl = 0
                (
                    fri_pln_id_next,
                    fri_dcy_id_next,
                    fri_msl_id_next,
                    enm_pln_id_next,
                    enm_dcy_id_next,
                    enm_msl_id_next,
                ) = reset_id0(0xA0000)
                for grp in unit_groups:
                    arrange_id(grp)
                    grp.reset(None)
                    if grp is agent:
                        grp.activate(None)
                    grp.logger.debug(f"reset @Ep{sim_epi}")

                tmr_fresh.reset()
                tmr_fps.reset()
                sim_k = 0
                ss = ss_step
            elif ss == ss_step:  # @step
                if f_is_paused and f_is_paused():
                    ss = ss_pause
                    recorder.add(recorder.format_bookmark("PAUSE"))
                    msg = recorder.merge()
                    if msg:
                        recorder.write_local(msg)
                        if tacview_full:
                            rst = recorder.write_remote(msg)
                    continue
                if tmr_fresh.step() == 0:
                    time.sleep(tmr_fresh.t_to_next())
                    continue
                if sim_k == 1 and f_set_pause and tacview_full:
                    f_set_pause(True)  # 先暂停游戏切换一下视角
                tmr_fps.step()
                recorder.add(recorder.format_timestamp(get_simt()))

                act = np.clip(get_action(), -1, 1)

                emn_msl_idxs = np.where(enm_msl.is_alive().reshape(-1, 1))[0]
                # action = PlaneAction(
                #     nx=float(
                #         (act[ACT_IDX_NX] * nx_max)
                #         if act[ACT_IDX_NX] >= 0
                #         else (act[ACT_IDX_NX] * nx_min)
                #     ),
                #     nz=float(act[ACT_IDX_NZ] * nz_max),
                #     roll_speed=float(act[ACT_IDX_ROLL] * dmu_max),
                # )
                action = PlaneAction(
                    nx_cmd=np.sign(act[ACT_IDX_NX]),
                    ny_cmd=np.sign(act[ACT_IDX_NY]),
                    nz_cmd=np.sign(act[ACT_IDX_NZ]),
                    droll_cmd=np.sign(act[ACT_IDX_ROLL]),
                )
                # pln.set_action(
                #     nx=action.nx, ny=0, nz=action.nz, roll_speed=action.roll_speed
                # )

                if act[ACT_IDX_DECOY]:  # 诱饵弹
                    if decoys in unit_groups:
                        dec_idx = decoy_regen(
                            grp=decoys,
                            new_id=fri_dcy_id_next,
                            parent_id=agent.acmi_id,
                            parent_pos=agent.position_e(),
                            parent_vel=agent.velocity_e(),
                            unit_tc=unit_tc,  # @decoy
                            rng=rng,
                            recorder=recorder,
                        )
                        if dec_idx >= 0:
                            ncomsum_dec += 1
                            fri_dcy_id_next += 1
                        else:
                            msg = "no available decoy"
                            print(msg)
                    else:
                        print("no decoy group, please add it into 'unit_groups'")

                if act[ACT_IDX_MSL] and len(emn_msl_idxs):  # 拦截弹
                    if fri_msl in unit_groups:

                        aim_idx = int(rng.choice(emn_msl_idxs))
                        msl_idx = missile_regen(  # @拦截弹
                            grp=fri_msl,
                            target_pos=enm_msl.position_e(),
                            new_id=fri_msl_id_next,
                            unit_tc=partial(unit_tc, Vmin=msl_Vmin),  # @missile
                            rng=rng,
                            hmin=hmin,
                            hmax=hmax,
                            alt0=alt0,
                            parent_pos=agent.position_e(),
                            parent_vel=agent.velocity_e(),
                            parent_rpy=agent.rpy_ew(),
                            recorder=recorder,
                        )
                        if msl_idx >= 0:
                            ncomsum_msl += 1
                            fri_msl_id_next += 1
                        else:
                            msg = f"{pln_info.acmi_id} have no available missile"
                            print(msg)
                    else:
                        print("no fri_msl group, please add it into 'unit_groups'")

                if action != action0:
                    if isinstance(agent, Plane_):
                        act_final = agent.action_n2c(
                            np.asarray(
                                [
                                    action.nx_cmd,
                                    action.ny_cmd,
                                    action.nz_cmd,
                                    action.droll_cmd,
                                ],
                                dtype=ftype,
                            ).reshape(1, -1),
                            linear=False,
                        )
                        agent.set_action(
                            act_final,
                            None,
                        )
                    elif isinstance(agent, Missile):
                        agent.set_action(
                            np.asarray(
                                [
                                    # action.nx_cmd,
                                    action.ny_cmd,
                                    action.nz_cmd,
                                    action.droll_cmd,
                                ],
                                dtype=ftype,
                            ).reshape(1, -1)
                        )
                    action0 = action
                    msg = f"{pln_info.acmi_id}|new action=" + " ".join(
                        [f"{k}={v}" for k, v in action.__dict__.items()]
                    )
                    recorder.add(recorder.format_bookmark(msg))
                    print(msg)

                # set control
                for grp in unit_groups:
                    if not grp.is_alive().any():
                        continue
                    if isinstance(grp, Missile) and grp is not agent:
                        if msl_simple_sens:
                            enm = agent if grp is enm_msl else enm_msl
                            enm_alv = enm.is_alive()
                            if not enm_alv.any():
                                continue
                            idx = (np.where(enm_alv)[0])[0]
                            if grp.target_id[0, 0] != enm.acmi_id[idx, 0]:
                                logr.info(("missile lock target", enm.acmi_id[idx, 0]))
                            grp.set_target(
                                enm.position_e()[..., [idx], :],
                                enm.velocity_e()[..., [idx], :],
                                enm.acmi_id[..., [idx], :],
                                None,
                            )
                            grp.set_action(grp.png(grp.target_pos_e, grp.target_vel_e))
                        else:
                            enm_pos, enm_vel, enm_mask, enm_id = merge_info4observe(
                                grp, unit_groups
                            )

                            grp.observe(
                                np.asarray(
                                    enm_pos,
                                    dtype=ftype,
                                ),
                                np.asarray(
                                    enm_vel,
                                    dtype=ftype,
                                ),
                                np.asarray(
                                    enm_id,
                                    dtype=np.int64,
                                ),
                                np.asarray(
                                    enm_mask,
                                    dtype=np.bool_,
                                ),
                            )
                        if grp is enm_msl and sim_k % 5 == 0:
                            logr.debug(
                                "\n".join(
                                    (
                                        "on emn_msl observe",
                                        "pln_pos:{}".format(
                                            agent.position_e()[0, :].ravel().tolist()
                                        ),
                                        "emn_tgt_pos:{}".format(
                                            enm_msl.target_pos_e[0, :].ravel().tolist()
                                        ),
                                        "pln_vel:{}".format(
                                            agent.velocity_e()
                                            .reshape(-1, 1)[0, :]
                                            .ravel()
                                            .tolist()
                                        ),
                                        "emn_tgt_vel:{}".format(
                                            enm_msl.target_vel_e[0, :].ravel().tolist()
                                        ),
                                    )
                                )
                            )

                # run ODE
                for grp in unit_groups:
                    grp.run(None)

                plnV = agent.tas().reshape(-1, 1)[0, 0].item()
                pln_alt = agent.altitude_m().reshape(-1, 1)[0, 0].item()
                if tmr_echo.step():
                    msg = [
                        f"fps={tmr_fps.fps():.01f}",
                        f"plnV={plnV:.01f}",
                        f"alt={pln_alt:.01f}",
                    ]
                    msg = " ".join(msg)
                    msg = msg.ljust(40)
                    print(msg)

                trunc = False
                term = False
                if f_trunc and f_trunc():
                    trunc = True

                # logic terminate condition
                uids_to_del = []
                for grp in unit_groups:
                    if isinstance(grp, Missile):
                        grp.try_hit()
                        grp.try_miss()
                        boom = grp.is_alive() & ~grp.is_no_result()
                        boom = boom.squeeze(-1)
                        if boom.any():
                            logr.info(("boom!@", grp.acmi_id[boom, :].tolist()))
                            grp.set_status(grp.STATUS_DYING, boom)  # 爆炸
                            try_kill_groups(grp, unit_groups, recorder)  # 对群毁伤判定

                    grp_tmax = simt_tol
                    if grp is agent:
                        pass
                    elif isinstance(grp, BaseMissile):
                        grp_tmax = msl_simt_tol
                    elif isinstance(grp, BaseDecoy):
                        grp_tmax = dcy_simt_tol

                    for iu in range(grp.batch_size):
                        u_uidh = format_id(int(grp.acmi_id[iu, [0]].item()))
                        imsk = index2mask(grp.group_shape, iu)
                        if grp.is_alive().reshape(-1, 1)[iu, [0]].item():

                            tc_evt = unit_tc(grp, iu, tmax=grp_tmax)  # @其他终止条件
                            if len(tc_evt):
                                grp.set_status(grp.STATUS_DYING, imsk)
                                msg = [f"{u_uidh} dying"]
                                msg.extend(
                                    [" - ".join([u_uidh, *eitems]) for eitems in tc_evt]
                                )
                                msg = "|".join(msg)
                                print(msg)
                                recorder.add(recorder.format_bookmark(msg))

                        elif grp.is_dying().reshape(-1, 1)[iu, [0]].item():
                            grp.set_status(grp.STATUS_DEAD, imsk)
                            recorder.add(recorder.format_destroyed(u_uidh))
                            recorder.add(recorder.format_remove(u_uidh))
                            uids_to_del.append(u_uidh)

                # render acmi
                render_groups(unit_groups, recorder)

                # 死亡回收
                use_reuse = True
                for grp in unit_groups:
                    if grp is agent:
                        continue
                    if not use_reuse:
                        continue
                    is_dead = grp.is_dead().squeeze(-1)
                    if is_dead.any():
                        if grp is enm_msl:
                            ndoge_msl += int(is_dead.sum())

                        grp.set_status(grp.STATUS_INACTIVE, is_dead)

                # for uid in uids_to_del:  # clean deads
                #     if uid in groups:
                #         del groups[uid]
                if not agent.is_alive().reshape(-1, 1)[0, 0].item():
                    term = True

                # 重复产生新导弹
                if enm_msl in unit_groups:
                    n_enm_msl = int(enm_msl.is_alive().sum().item())
                    for _ in range(max_enm_missiles - n_enm_msl):
                        msl_idx = missile_regen(  # @敌方导弹
                            grp=enm_msl,
                            target_pos=agent.position_e()[0, :],
                            new_id=enm_msl_id_next,
                            unit_tc=partial(unit_tc, Vmin=msl_Vmin),  # @missile,
                            rng=rng,
                            hmin=hmin,
                            hmax=hmax,
                            alt0=alt0,
                            recorder=recorder,
                        )
                        if msl_idx >= 0:
                            enm_msl_id_next += 1

                sim_k += 1
                if get_simt() > simt_tol:
                    print("time out")
                    trunc = True
                if trunc or term:
                    ss = ss_reset
                    sim_epi += 1
                    game_win = agent.is_alive().reshape(-1, 1)[0, 0].item()
                    logr.info(
                        "\n".join(
                            [
                                "Ep{} end".format(sim_epi),
                                "alive time={:.1f}s".format(get_simt()),
                                "result={}".format(f"win" if game_win else "lose"),
                                "consumed missiles={}".format(ncomsum_msl),
                                "consumed decoys={}".format(ncomsum_dec),
                                "doge missiles={}".format(ndoge_msl),
                            ]
                        )
                    )

                msg = recorder.merge()
                if msg:
                    recorder.write_local(msg)
                    if tacview_full and recorder.is_connected():
                        rst = recorder.write_remote(msg)
            elif ss == ss_pause:
                if f_is_paused and not f_is_paused():
                    tmr_fps.reset()
                    tmr_fresh.reset()
                    ss = ss_step
                    continue
                time.sleep(0.01)
        except KeyboardInterrupt:
            break

    recorder.close()


def demo4LOSblock(
    n=30,
    seed=int(time.time()),
    Rbase=0.5,
    rRmin=1.0,
    rRmax=3.0,
    rrmin=0.2,
    rrmax=0.5,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    rng = np.random.default_rng(seed)

    theta = rng.uniform(0, 2 * np.pi, (n, 1))
    R_ = affcmb(rng.random((n, 1)), Rbase * rRmin, Rbase * rRmax)
    ps = np.concatenate(
        [
            R_ * np.cos(theta),
            R_ * np.sin(theta),
        ],
        axis=-1,
    )
    brs = affcmb(rng.random((n, 1)), Rbase * rrmin, Rbase * rrmax)

    t0 = time.time()
    rst = math_ext.los_is_visible(np.asarray(ps), np.asarray(brs))
    noblock = rst[0]  # (n,1)
    etc = rst[1]  # (n,1)
    blocked = ~noblock
    mix = np.ravel(blocked & etc)
    behind = np.ravel(blocked & ~etc)
    dt = max(time.time() - t0, 1e-3)
    print(f"calc_is_blocked {n} pts, {dt:.3f} s, {n/dt:.1f} pts/s")
    fig = plt.figure(f"视线遮挡演示,关闭本窗口后游戏开始")
    ax = fig.gca()
    for mask, clr, ls in [
        (behind, "gray", "--"),
        (mix, "orange", "--"),
        (noblock, "green", "-"),
    ]:
        idxs = np.where(mask)[0]
        for i in idxs:
            x = ps[i, 0]
            y = ps[i, 1]
            r_i = brs[i].item()
            circle = Circle((x, y), r_i, edgecolor=clr, facecolor="none")
            ellipse = Ellipse(
                (x, y),
                2 * r_i,
                2 * r_i,
                # angle=0,
                # edgecolor=clr,
                facecolor=clr,
                alpha=0.3,
            )
            ax.plot([0, x], [0, y], linestyle=ls, c=clr)
            ax.add_patch(circle)
            ax.add_patch(ellipse)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"seed={seed}, n={n}")

    plt.show(block=True)


def main():
    np.set_printoptions(precision=4, suppress=True)

    tacview_full = True  # 是否有高级版
    _is_stop_ = False
    _esc_ = 0  # 退出程序
    _reset_ = 0  # 重置游戏
    _trunc_ = False  # 单局中断
    _pause_ = False  # 暂停游戏
    esc_tol = 3
    reset_tol = 3
    a_vec = np.zeros(len(ACT_KEYS))

    def set_pause(pause: bool = True):
        nonlocal _pause_
        if pause != _pause_ and pause:
            print(f"暂停, P 继续")
        _pause_ = pause

    def is_stop():
        return _is_stop_

    def is_trunc():
        nonlocal _trunc_
        rst = _trunc_
        _trunc_ = False
        return rst

    def is_paused():
        return _pause_

    def get_action_vec():
        nonlocal a_vec
        a_vec_ = deepcopy(a_vec)
        a_vec[ACT_IDX_DECOY] = 0
        a_vec[ACT_IDX_MSL] = 0
        return a_vec_

    # 定义处理键盘事件的回调函数
    def on_press(key: Union[keyboard.KeyCode, keyboard.Key, None]):
        if key is None:  # ???
            return
        nonlocal a_vec
        key_ = _Keymap.fromKey(key)
        if HOTKEY_NX_P.equals(key_):
            a_vec[ACT_IDX_NX] = 1
        elif HOTKEY_NX_N.equals(key_):
            a_vec[ACT_IDX_NX] = -1
        elif HOTKEY_NY_P.equals(key_):
            a_vec[ACT_IDX_NY] = 1
        elif HOTKEY_NY_N.equals(key_):
            a_vec[ACT_IDX_NY] = -1
        elif HOTKEY_NZ_P.equals(key_):
            a_vec[ACT_IDX_NZ] = 1
        elif HOTKEY_NZ_N.equals(key_):
            a_vec[ACT_IDX_NZ] = -1
        elif HOTKEY_ROLL_P.equals(key_):
            a_vec[ACT_IDX_ROLL] = 1
        elif HOTKEY_ROLL_N.equals(key_):
            a_vec[ACT_IDX_ROLL] = -1

    def on_release(key: Union[keyboard.KeyCode, keyboard.Key, None]):
        # 想要停止监听则返回 False
        if key is None:  # ???
            return True
        nonlocal a_vec, _esc_, _trunc_, _reset_
        key_ = _Keymap.fromKey(key)
        if HOTKEY_ESC.equals(key_):
            _esc_ += 1
            if _esc_ == esc_tol:
                print()
                return False
            else:
                print(f"再连按 {esc_tol - _esc_} 次 esc 退出")
        elif HOTKEY_RESET.equals(key_):
            _reset_ += 1
            if _reset_ == reset_tol:
                _reset_ = 0
                print("重启...")
                _trunc_ = True
            else:
                print(f"再连按 {reset_tol - _reset_} 次 {HOTKEY_RESET.keyname} 重启")
        elif HOTKEY_DECOY.equals(key_):
            a_vec[ACT_IDX_DECOY] = 1
        elif HOTKEY_DMSL.equals(key_):
            a_vec[ACT_IDX_MSL] = 1
        elif HOTKEY_PAUSE.equals(key_):
            set_pause(not is_paused())
        elif HOTKEY_NX_N.equals(key_) or HOTKEY_NX_P.equals(key_):
            a_vec[ACT_IDX_NX] = 0
        elif HOTKEY_NY_N.equals(key_) or HOTKEY_NY_P.equals(key_):
            a_vec[ACT_IDX_NY] = 0
        elif HOTKEY_NZ_N.equals(key_) or HOTKEY_NZ_P.equals(key_):
            a_vec[ACT_IDX_NZ] = 0
        elif HOTKEY_ROLL_N.equals(key_) or HOTKEY_ROLL_P.equals(key_):
            a_vec[ACT_IDX_ROLL] = 0

        if not HOTKEY_ESC.equals(key_):
            _esc_ = 0
        if not HOTKEY_RESET.equals(key_):
            _reset_ = 0

    # game_run
    kwargs = dict(
        get_action=get_action_vec,
        f_stop=is_stop,
        f_trunc=is_trunc,
        f_is_paused=is_paused,
        f_set_pause=set_pause,
        tacview_full=tacview_full,
    )
    th1 = threading.Thread(target=game_run, kwargs=kwargs)
    th1.start()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)  # type: ignore
    listener.start()

    try:
        listener.join()
    except KeyboardInterrupt:
        listener.stop()
        pass
    _is_stop_ = True
    th1.join()
    print("退出")


if __name__ == "__main__":
    main()
