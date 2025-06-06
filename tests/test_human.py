# 人类控制测试
from __future__ import annotations
import logging
import pymap3d


def _setup():
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


ROOT = _setup()

from enum import Enum
import torch
from environments.utils import log_ext
from environments.models.base_model import BaseModel
from environments.models.aircraft import PDOF6Plane as Plane
from environments.utils.tacview import TacviewRecorder, ACMI_Types, acmi_id
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import threading
import time
from typing import Callable, Dict, List, Sequence, Tuple, Union, cast

import numpy as np
import torch
from pynput import keyboard
from environments.utils.math_np import calc_zem, rpy2mat
from environments.models.aircraft.pdof6plane import PDOF6Plane as Plane, BaseAircraft
from environments.models.missile.pdof6missile import (
    PDOF6Missile as Missile,
    BaseMissile,
)
from environments.models.base_model import BaseModel
from environments.models.decoy.base_decoy import BaseDecoy
from environments.models.decoy.dof0decoy import DOF0BallDecoy as Decoy
from environments.utils.time_ext import Timer_Pulse

ACT_IDX_NX = 0
ACT_IDX_NZ = 1
ACT_IDX_ROLL = 2
ACT_IDX_DECOY = 3
ACT_IDX_MSL = 4
ACT_KEYS = [ACT_IDX_NX, ACT_IDX_NZ, ACT_IDX_ROLL, ACT_IDX_DECOY, ACT_IDX_MSL]


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
        u = acmi_id(self.id)
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


HOTKEY_NX_N = _Keymap("[")
HOTKEY_NX_P = _Keymap("]")
HOTKEY_NZ_N = _Keymap("s")
HOTKEY_NZ_P = _Keymap("w")
HOTKEY_ROLL_N = _Keymap("a")
HOTKEY_ROLL_P = _Keymap("d")
HOTKEY_DMSL = _Keymap("g")
HOTKEY_DECOY = _Keymap("h")
HOTKEY_ESC = _Keymap.fromKey(keyboard.Key.esc)
HOTKEY_PAUSE = _Keymap("p")  # 暂停仿真
HOTKEY_RESET = _Keymap("r")
# 热键冲突警告: 方向键被用于播放条控制, ijkl 用于视角移动, +-qe 视角缩放, 空格暂停播放(但是仿真依然推进)


def los_is_blocked(los: np.ndarray, rthres: float | np.ndarray):
    """
    Args:
        los: 视线组 shape=(n,2|3)
        rthres: 视线终点的阻挡半径 float|shape=(n,1)
    Returns:
        block: 是否被遮挡 shape=(n,1)
    """
    assert isinstance(los, (np.ndarray))
    assert len(los.shape) == 2
    n = los.shape[0]
    _0f = np.zeros_like(los)
    dij = calc_zem(_0f, los, los, _0f, tmax=1)[0]  # (n,n,1)
    # _[i,j]= 线段 {los[i]*t|t\in [0,1]} 与点 los[j] 的最短距离

    rthres = np.reshape(rthres, (1, -1, 1))  # (n|1,)
    rthres = np.broadcast_to(rthres, (1, n, 1))
    bij = dij < rthres  # _[i,j]= los[i] 被 los[j] 遮挡
    for i in range(bij.shape[0]):  # 自己对自己不构成遮挡
        bij[i, i] = False
    blk = np.any(bij, axis=1)  # any(bij[i,j] for j)
    blk = np.ravel(blk)
    return blk


def uniform_s(low, high, rng: np.random.Generator):
    low, high = np.broadcast_arrays(low, high)
    w = rng.random(low.shape)
    x = low + (high - low) * w
    return x


def affcmb(x, a, b):
    return a + (b - np.asarray(a)) * x


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

    valid = ~los_is_blocked(los, br)  # (n*m,)
    valid = valid.reshape(n, m)
    return valid


def reuse_index(
    group: BaseModel,
):
    tag = ~group.is_alive()  # (B,1)
    if tag.any():
        idxs = torch.where(tag)[0]  # (b,)
        index = int(idxs[0].item())
        return index
    else:
        return -1


def arrange_id(
    group: BaseModel,
):
    for i in range(1, group.batch_size):
        group.id[i, 0] = group.id[i - 1, 0] + 1


def missile_reuse(
    group: Missile,
    infos: List[ACMI_Info],
    new_id: int,
    target_pos: np.ndarray | torch.Tensor,
    unit_tc: Callable[[BaseModel, int], List[Sequence[str]]],
    ntry_tol=1000,
    rng: np.random.Generator = np.random.default_rng(),
):
    index: int = reuse_index(group)
    if index == -1:
        print("no available missile")
        return index
    m_info = infos[index]
    m_info.id = new_id
    m_info.CallSign = f"M{m_info.acmi_id}"
    infos[index] = m_info
    group.id[index, 0] = new_id
    group.call_sign[index] = m_info.CallSign

    # lat0, lon0, alt0 = target.blh0()
    th_float = group.dtype
    device = group.device
    det_rmax = group._det_rmax
    pln_ned_tsr = torch.asarray(target_pos, dtype=th_float, device=device).reshape(
        1, -1
    )  # (1,1)
    assert len(pln_ned_tsr.shape) == 2

    dst_index = slice(index, index + 1)
    for ntry in range(ntry_tol):
        rpy_np = affcmb(rng.random(3), [-180, -10, -180], [180, 10, 180])  # (3,)
        rpy_np = np.deg2rad(rpy_np)
        Rew_np = rpy2mat(rpy_np)  # (3,3)
        R_ = float(rng.uniform(0.2, 0.3) * det_rmax)
        los = Rew_np @ [R_, 0, 0]  # (3,)
        los_tsr = torch.asarray(los, dtype=th_float, device=device).reshape(
            1, 3
        )  # (1,3)
        rpy_tsr = torch.asarray(rpy_np, dtype=th_float, device=device).reshape(
            1, 3
        )  # (1,3)
        tas = float(affcmb(rng.random(), group._Vmin, group._Vmax))
        group.set_ic_tas(tas, dst_index)
        group.set_ic_rpy_ew(rpy_tsr, dst_index)

        group._pos_e[dst_index, :] = pln_ned_tsr - los_tsr
        group.launch(dst_index)

        # assert norm(msl._Reb - Reb) < 1e-2, f"Reb={Reb} rpy={rpy}, Reb={msl._Reb }"

        tc_evt = unit_tc(group, index)
        if len(tc_evt):
            continue
        break
    else:
        raise ValueError("max try reached@missile_reset")
    print(
        f"missile {m_info.acmi_id} created@index={index}, "
        + "pos_e={}, vel_e={}".format(
            group.position_e(index).cpu().ravel().tolist(),
            group.velocity_e(index).cpu().ravel().tolist(),
        )
    )
    return index


def decoy_reuse(
    group: Decoy,
    infos: List[ACMI_Info],
    new_id: int,
    parent_pos: torch.Tensor,
    parent_vel: torch.Tensor,
    unit_tc: Callable[[BaseModel, int], List[Sequence[str]]],
    ntry_tol=1000,
    rng: np.random.Generator = np.random.default_rng(),
):
    index: int = reuse_index(group)
    if index == -1:
        print("no available missile")
        return index
    m_info = infos[index]
    m_info.id = new_id
    m_info.CallSign = f"D{m_info.acmi_id}"
    infos[index] = m_info
    group.id[index, 0] = new_id
    group.call_sign[index] = m_info.CallSign
    # lat0, lon0, alt0 = target.blh0()
    th_float = group.dtype
    device = group.device
    pln_pos_tsr = torch.asarray(parent_pos, dtype=th_float, device=device).reshape(
        1, -1
    )  # (1,3)
    pln_vel_tsr = torch.asarray(parent_vel, dtype=th_float, device=device).reshape(
        1, -1
    )  # (1,3)
    dst_index = slice(index, index + 1)
    for ntry in range(ntry_tol):
        group._pos_e[dst_index, :] = pln_pos_tsr
        group._vel_e[dst_index, :] = pln_vel_tsr * (
            1 + (2 * torch.rand_like(pln_vel_tsr) - 1) * 0.1
        )
        group.reset(index)

        # assert norm(msl._Reb - Reb) < 1e-2, f"Reb={Reb} rpy={rpy}, Reb={msl._Reb }"

        tc_evt = unit_tc(group, index)
        if len(tc_evt):
            continue
        break
    else:
        raise ValueError("max try reached@decoy_reset")
    print(
        f"decoy {m_info.acmi_id} created@index={index}, "
        + "pos_e={}, vel_e={}".format(
            group.position_e(index).cpu().ravel().tolist(),
            group.velocity_e(index).cpu().ravel().tolist(),
        )
    )
    return index


def merge_info4observe(
    ego: BaseModel,
    groups: List[BaseModel],
):
    m = ego.batch_size
    ego_id = ego.id.reshape(1, m, 1)
    ego_clr = ego.acmi_color.reshape(1, m, 1)
    ego_alv = ego.is_alive().reshape(1, m, 1)
    tmp_masks = []
    tmp_pos = []
    tmp_vel = []
    tmp_r = []
    tmp_idt = []
    for grp in groups:
        n = grp.batch_size
        grp_ids = grp.id.reshape(n, 1, 1)
        grp_clr = grp.acmi_color.reshape(n, 1, 1)
        is_enm = ego_clr != grp_clr  # (n,m,1)
        not_self = grp_ids != ego_id  # (n,m,1)
        grp_alv = grp.is_alive().reshape(n, 1, 1)
        tag21 = (ego_alv & grp_alv) & is_enm & not_self  # (n,m,1)
        tmp_masks.append(tag21)
        tmp_pos.append(grp.position_e())  # (n,3)
        tmp_vel.append(grp.velocity_e())  # (n,3)
        tmp_r.append(grp.vis_radius().reshape(n, 1, 1))  # (n,1)
        tmp_idt.append(not_self)

    mask = torch.cat(tmp_masks, dim=0)  # (N,m,1)
    N = mask.shape[0]
    pos2 = torch.cat(tmp_pos, dim=0)  # (N,3)
    vel2 = torch.cat(tmp_vel, dim=0)  # (N,3)
    visr2 = torch.cat(tmp_r, dim=0)  # (N,1,1)

    not_self = torch.cat(tmp_idt, dim=0)  # (N,m,1)
    visr12 = visr2.reshape(N, 1).broadcast_to(N, m)
    los12 = pos2.reshape(N, 1, -1) - ego.position_e().reshape(1, m, -1)  # (N,m,3)
    los12 += not_self  # 距离掩码修正, 否则会在除0

    blk = los_is_blocked(
        los12.reshape(N * m, -1).cpu().numpy(),
        visr12.reshape(N * m, 1).cpu().numpy(),
    )  # (m*N,)
    blk = torch.asarray(blk, dtype=torch.bool, device=mask.device)  # (m*N,)
    valid = (~blk).reshape(N, m, 1)
    mask = valid & mask  # (N,m,1)
    mask = mask.reshape(N, m)
    return (
        pos2,  # (N,3)
        vel2,  # (N,3)
        mask,  # (N,m)
    )


def render_groups(
    groups: List[BaseModel],
    recorder: TacviewRecorder,
):
    for grp in groups:
        n = grp.batch_size
        idxs = torch.where(grp.is_alive())[0]
        if len(idxs) == 0:
            continue
        p_ned = grp.position_e(idxs).cpu().numpy()
        tas = grp.tas(idxs).cpu().numpy()
        blh0 = grp.blh0(idxs).cpu().numpy()
        lat0, lon0, alt0 = np.split(blh0, 3, axis=-1)
        loc_n, loc_e, loc_d = np.split(p_ned, 3, axis=-1)
        lat, lon, alt = pymap3d.ned2geodetic(loc_n, loc_e, loc_d, lat0, lon0, alt0)
        rpy = None
        if rpy is None:
            try:
                rpy = grp.rpy_eb(idxs)
            except Exception:
                pass
        if rpy is None:
            try:
                rpy = grp.rpy_ew(idxs)
            except Exception:
                pass
        if rpy is not None:
            rpy = cast(np.ndarray, rpy.cpu().numpy())
            rpy_deg = np.rad2deg(rpy)
        for i in range(n):
            if rpy is None:
                roll_i = pitch_i = yaw_i = None
            else:
                roll_i, pitch_i, yaw_i = rpy_deg[i]
            recorder.add(
                recorder.format_unit(
                    int(grp.id[i, 0].item()),
                    lat=lat[i],
                    lon=lon[i],
                    alt=alt[i],
                    tas=tas[i],
                    roll=roll_i,
                    pitch=pitch_i,
                    yaw=yaw_i,
                    Name=grp.acmi_name[i],
                    Color=grp.acmi_color[i],
                    Type=grp.acmi_type[i],
                    CallSign=grp.call_sign[i],
                )
            )


def try_kill_groups(
    ego: BaseMissile,
    groups: List[BaseModel],
    recorder: TacviewRecorder,
):
    m = ego.batch_size
    tag1 = ego.is_dying().reshape(m, 1)
    ids1 = ego.id.reshape(m, 1)
    p1 = ego.position_e().reshape(m, 1, -1)
    kr = (torch.zeros_like(ego.tas()) + ego.kill_radius).reshape(m, 1)
    msgs: List[str] = []
    for i in range(m):
        if not tag1[i, 0].item():
            continue
        idh = recorder.format_id(int(ids1[i, 0].item()))
        md = ego.miss_distance[i, 0].item()
        msg = f"{idh} exploding MD={md:.1f}"
        recorder.add(recorder.event_bookmark(msg))
        msgs.append(msg)

    for grp in groups:
        n = grp.batch_size
        ids2 = grp.id.reshape(1, n)
        is_self = ids1 == ids2
        tag2 = grp.is_alive().reshape(1, n)
        p2 = grp.position_e().reshape(1, n, -1)
        dij = torch.norm(p1 - p2, dim=-1)  # (m,n)
        kill_evt = (~is_self) & (tag1 & tag2) & (dij < kr)  # (m,n)
        kill_tag_ = kill_evt.any(0)  # (n,)
        if kill_tag_.any():
            idxs = torch.where(kill_tag_)[0]
            grp.set_status(grp.STATUS_DYING, idxs)
            for i in range(m):
                id1 = recorder.format_id(int(ego.id[i, 0].item()))
                for j in range(n):
                    if not kill_evt[i, j].item():
                        continue
                    id2 = recorder.format_id(int(grp.id[j, 0].item()))
                    evt = "{} hit {}".format(id1, id2)
                    msgs.append(evt)
                    recorder.add(recorder.event_bookmark(evt))
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
):
    fri_msl_N = 2
    enm_msl_N = 2
    decoys_N = 3
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
    Vmin = 100
    # 飞机能力参数
    nx_max = 1.5
    nx_min = -0.5
    nz_dmax = 0.5
    nz_umax = 10.0
    dmu_max = (2 * math.pi) / 3  # 3s 转 360°
    pln_V_lb = Vmin
    pln_V_ub = 500

    th_float = torch.float32
    np_float = np.float32
    device = "cpu"

    # 干扰参数
    pln_objr = 10.0
    dec_objr = 30.0
    msl_objr = 0.5

    # 导弹能力参数
    msl_det_rmax = 50000
    msl_det_half_angle = 90
    msl_det_fov_deg = 2 * msl_det_half_angle
    msl_trk_fov_deg = 30
    msl_V0 = 300
    msl_Vmax = 1200
    msl_Vmin = Vmin
    msl_nmax = (msl_Vmax * (math.tau / 4)) / g

    _FILE = Path(__file__).resolve()
    runs_dir = _FILE.parent / "tmp"
    acmi_dir = runs_dir
    tel_addr = ("localhost", 21000)

    fri_color = ACMI_Color.Red.value
    enm_color = ACMI_Color.Blue.value

    simdt_sec = 1.0 / 24
    simdt_ms = int(simdt_sec * 1000)
    rng = np.random.default_rng()

    lat0 = 30.0
    lon0 = 105.0
    alt0 = 5000
    hmax = 25000
    hmin = 100
    simt_tol = 20 * 60

    logr = log_ext.reset_logger(
        __name__ + "_pln",
        level=logging.DEBUG,
        file_path=str(runs_dir / "log.log"),
        file_append=False,
    )
    enm_msl_logr = log_ext.reset_logger(
        logr.name + ("_msl"),
        level=logging.DEBUG,
        file_path=str(runs_dir / "msl.log"),
        file_append=False,
    )
    pln_info = ACMI_Info(fri_pln_id_next, Color=fri_color, Name="J-10")
    pln_info.CallSign = f"F{pln_info.acmi_id}"
    pln = Plane(
        tas=240,
        rpy_ew=0,
        position_e=torch.asarray(
            np.asarray([0, 0, -(hmax + hmin) * 0.5], dtype=np_float), dtype=th_float
        ),
        sim_step_size_ms=simdt_ms,
        id=fri_pln_id_next,
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
        acmi_name=pln_info.Name,
        acmi_color=pln_info.Color,
        acmi_type=pln_info.Type,
        acmi_parent="",
        call_sign=pln_info.CallSign,
        vis_radius=pln_objr,
    )
    pln.logr = logr
    for iteam, msl_N in enumerate(
        [fri_msl_N, enm_msl_N],
    ):
        msl_color = fri_color if iteam == 0 else enm_color
        msls = Missile(
            sim_step_size_ms=simdt_ms,
            id=fri_msl_id_next if iteam == 0 else enm_msl_id_next,
            batch_size=msl_N,
            Vmin=msl_Vmin,
            Vmax=msl_Vmax,
            det_rmax=msl_det_rmax,
            det_fov_deg=msl_det_fov_deg,
            trk_fov_deg=msl_trk_fov_deg,
            dtype=th_float,
            device=device,
            acmi_color=msl_color,
            acmi_name="AIM-9M",
            acmi_type=ACMI_Types.Missile.value,
            acmi_parent=pln_info.acmi_id if iteam == 0 else "",
            call_sign="",
            lat0=lat0,
            lon0=lon0,
            alt0=alt0,
            vis_radius=msl_objr,
        )
        msls_info: List[ACMI_Info] = []
        for j in range(msl_N):
            msls.set_ic_tas(msl_V0, j)
            msls.set_ic_rpy_ew(0, j)
            msls._pos_e[j, :] = pln._pos_e[0, :] + 1000
            msls.reset(j)
            msls.set_status(msls.STATUS_INACTIVATE, j)

            if iteam == 0:
                fri_msl_id_next += 1
                newid = fri_msl_id_next
            else:
                enm_msl_id_next += 1
                newid = enm_msl_id_next
            msls_info.append(
                ACMI_Info(
                    newid,
                    Color=msls.acmi_color[j],
                    Name=msls.acmi_name[j],
                    Type=msls.acmi_type[j],
                    Parent=msls.acmi_parent[j],
                )
            )
            msls.id[j, 0] = newid

        if iteam == 0:
            fri_msl = msls
            fri_msl_infos = msls_info
        else:
            enm_msl = msls
            enm_msl_infos = msls_info

    enm_msl.logr = enm_msl_logr

    decoys = Decoy(
        sim_step_size_ms=simdt_ms,
        id=fri_dcy_id_next,
        batch_size=decoys_N,
        vis_radius=dec_objr,
        effect_duration=20.0,
        dtype=th_float,
        device=device,
        lat0=lat0,
        lon0=lon0,
        alt0=alt0,
    )
    decoys_info: List[ACMI_Info] = []
    for j in range(decoys_N):
        decoys._pos_e[j, :] = pln._pos_e[0, :]
        decoys._vel_e[j, :] = pln._vel_e[0, :]
        decoys.reset(j)
        decoys.set_status(msls.STATUS_INACTIVATE, j)

        fri_dcy_id_next += 1
        decoys_info.append(
            ACMI_Info(
                fri_dcy_id_next,
                Color=fri_color,
                Name="Decoy",
                Type=ACMI_Types.FlareDecoy.value,
            )
        )

    # 挂载限制(<0不限制)
    nlim_msl = -1
    nlim_dec = -1

    def unit_tc(unit: BaseModel, idx: int = 0) -> List[Sequence[str]]:
        rst = []
        if unit.is_alive(idx).item():
            alt = unit.altitude_m(idx).item()
            if alt <= hmin:
                rst.append(("fly too low", f"alt={alt:.1f}"))
            elif alt > hmax:
                rst.append(("fly too high", f"alt={alt:.1f}"))
            tas = unit.tas(idx).item()
            if tas <= Vmin:
                rst.append(("fly too slow", f"tas={tas:.1f}"))
        return rst

    recorder = TacviewRecorder()

    ss_reset = 0
    ss_step = 1
    ss_pause = 2
    episode = 0
    ss = ss_reset
    action0 = PlaneAction()
    tmr_fresh = Timer_Pulse(simdt_sec)
    tmr_fps = Timer_Pulse()
    tmr_echo = Timer_Pulse(1.0)
    simk = 0
    max_enm_missiles = enm_msl.batch_size  # 同时存在的最大数量

    def get_simt():
        return simk * simdt_sec

    groups: List[BaseModel] = [
        pln,  # @group
        # fri_msl,# @group
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
                    f"{HOTKEY_RESET} 重启",
                    f"{HOTKEY_NZ_N} / {HOTKEY_NZ_P} 俯仰过载",
                    f"{HOTKEY_ROLL_N} / {HOTKEY_ROLL_P} 滚转角速度",
                    f"{HOTKEY_NX_N} / {HOTKEY_NX_P} 切向过载",
                    f"{HOTKEY_PAUSE} 暂停/继续",
                    f"{HOTKEY_DECOY} 诱饵弹",
                    f"{HOTKEY_DMSL} 拦截弹",
                    sep="\n",
                )

                acmitime = datetime.now()
                acmitime = datetime(acmitime.year, acmitime.month, acmitime.day)
                if tacview_full:
                    rst = recorder.reset_remote(
                        addr=tel_addr, reference_time=acmitime, timeout=10.0
                    )
                    if not rst:
                        print("reset_remote rfailed")
                        # _arg = input("skip connection? ([y]/n)")
                        # if _arg.lower() == "n":
                        #     continue
                        # else:
                        #     pass
                        continue
                acmi_fn = runs_dir / f"{episode}.acmi"
                acmi_fn.parent.mkdir(exist_ok=True, parents=True)
                recorder.reset_local(acmi_fn, reference_time=acmitime)

                (
                    fri_pln_id_next,
                    fri_dcy_id_next,
                    fri_msl_id_next,
                    enm_pln_id_next,
                    enm_dcy_id_next,
                    enm_msl_id_next,
                ) = reset_id0(0xA0000)
                for grp in groups:
                    arrange_id(grp)
                    grp.reset()
                    grp.set_status(
                        grp.STATUS_ALIVE
                        if isinstance(grp, Plane)
                        else grp.STATUS_INACTIVATE
                    )

                tmr_fresh.reset()
                tmr_fps.reset()
                simk = 0
                ss = ss_step
            elif ss == ss_step:
                if f_is_paused and f_is_paused():
                    ss = ss_pause
                    recorder.add(recorder.event_bookmark("PAUSE"))
                    msg = recorder.merge()
                    if msg:
                        recorder.write_local(msg)
                        if tacview_full:
                            rst = recorder.write_remote(msg)
                    continue
                if tmr_fresh.step() == 0:
                    time.sleep(tmr_fresh.t_to_next())
                    continue
                if simk == 1 and f_set_pause and tacview_full:
                    f_set_pause(True)  # 先暂停游戏切换一下视角
                tmr_fps.step()
                recorder.add(recorder.format_time(get_simt()))

                act = np.clip(get_action(), -1, 1)

                emn_msl_idxs = torch.where(enm_msl.is_alive())[0]

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
                    nz_cmd=np.sign(act[ACT_IDX_NZ]),
                    droll_cmd=np.sign(act[ACT_IDX_ROLL]),
                )
                # pln.set_action(
                #     nx=action.nx, ny=0, nz=action.nz, roll_speed=action.roll_speed
                # )

                if act[ACT_IDX_DECOY]:  # 诱饵弹
                    if decoys in groups:
                        dec_idx = decoy_reuse(
                            group=decoys,
                            infos=decoys_info,
                            new_id=fri_dcy_id_next,
                            parent_pos=pln.position_e(),
                            parent_vel=pln.velocity_e(),
                            unit_tc=unit_tc,
                            rng=rng,
                        )
                        if dec_idx >= 0:
                            fri_dcy_id_next += 1
                            msg = "{} launched decoy {}".format(
                                recorder.format_id(int(pln.id[0, 0].item())),
                                recorder.format_id(int(decoys.id[dec_idx, 0].item())),
                            )
                            recorder.add(recorder.event_bookmark(msg))
                        else:
                            msg = "no available decoy"
                        print(msg)
                    else:
                        print("no decoy group, please add it into groups")

                if act[ACT_IDX_MSL] and len(emn_msl_idxs):  # 拦截弹
                    if fri_msl in groups:
                        aim_idx = int(rng.choice(emn_msl_idxs.cpu().tolist()))
                        msl_idx = missile_reuse(
                            group=fri_msl,
                            infos=fri_msl_infos,
                            target_pos=enm_msl.position_e(aim_idx),
                            new_id=fri_msl_id_next,
                            unit_tc=unit_tc,
                            rng=rng,
                        )
                        if msl_idx >= 0:
                            fri_msl_id_next += 1
                            msg = "{} launched {} to {}".format(
                                recorder.format_id(int(pln.id[0, 0].item())),
                                recorder.format_id(int(fri_msl.id[msl_idx, 0].item())),
                                recorder.format_id(int(enm_msl.id[aim_idx, 0].item())),
                            )

                            msg = f"{pln_info.acmi_id} launched {msls_info[msl_idx].CallSign}"
                            recorder.add(recorder.event_bookmark(msg))
                        else:
                            msg = f"{pln_info.acmi_id} no available missile"
                        print(msg)
                    else:
                        print("no fri_msl group, please add it into groups")

                if action != action0:
                    pln.set_action(
                        np.asarray(
                            [action.nx_cmd, 0, action.nz_cmd, action.droll_cmd],
                            dtype=np_float,
                        ).reshape(1, -1)
                    )
                    action0 = action
                    msg = f"{pln_info.acmi_id}|new action=" + " ".join(
                        [f"{k}={v}" for k, v in action.__dict__.items()]
                    )
                    recorder.add(recorder.event_bookmark(msg))
                    print(msg)

                # set control
                for grp in groups:
                    if isinstance(grp, Missile):
                        pos, vel, mask = merge_info4observe(grp, groups)
                        grp.observe(
                            torch.asarray(pos, dtype=th_float, device=device),
                            torch.asarray(vel, dtype=th_float, device=device),
                            torch.asarray(mask, dtype=torch.bool, device=device),
                        )

                # step
                for grp in groups:
                    grp.run()

                plnV = pln.tas(0).item()
                pln_alt = pln.altitude_m(0).item()
                if tmr_echo.step():
                    msg = [
                        f"fps={tmr_fps.fps():.01f}",
                        f"plnV={plnV:.01f}",
                        "alt={:.0f}".format(pln_alt),
                    ]
                    msg = " ".join(msg)
                    msg = msg.ljust(40)
                    print(msg)

                # render acmi
                render_groups(groups, recorder)

                trunc = False
                term = False
                if f_trunc and f_trunc():
                    trunc = True

                # terminate condition
                uids_to_del = []
                for grp in groups:
                    if isinstance(grp, Missile):
                        grp.try_hit()
                        grp.try_miss()
                        boom = (grp._result != Missile.RESULT_NONE) & grp.is_alive()
                        if boom.any():
                            idxs_explode = torch.where(boom)[0]
                            grp.set_status(grp.STATUS_DYING, idxs_explode)  # 爆炸
                            try_kill_groups(grp, groups, recorder)  # 群体毁伤判定

                    for iu in range(grp.batch_size):
                        u_uidh = recorder.format_id(int(grp.id[iu, 0].item()))
                        if grp.is_alive(iu).item():
                            tc_evt = unit_tc(grp, iu)  # 其他终止条件
                            if len(tc_evt):
                                grp.set_status(grp.STATUS_DYING, iu)
                                msg = [f"{u_uidh} dying"]
                                msg.extend(
                                    [" - ".join([u_uidh, *eitems]) for eitems in tc_evt]
                                )
                                msg = "|".join(msg)
                                print(msg)
                                recorder.add(recorder.event_bookmark(msg))

                        elif grp.is_dying(iu).item():
                            grp.set_status(grp.STATUS_DEAD)
                            recorder.add(recorder.event_destroy(u_uidh))
                            recorder.add(recorder.event_remove(u_uidh))
                            uids_to_del.append(u_uidh)

                # for uid in uids_to_del:  # clean deads
                #     if uid in groups:
                #         del groups[uid]
                if not pln.is_alive():
                    term = True

                # 重复产生新导弹
                if enm_msl in groups:
                    n_enm_msl = int(enm_msl.is_alive().sum().item())
                    for iteam in range(max_enm_missiles - n_enm_msl):
                        msl_idx = missile_reuse(
                            group=enm_msl,
                            infos=enm_msl_infos,
                            target_pos=pln.position_e(0),
                            new_id=enm_msl_id_next,
                            unit_tc=unit_tc,
                            rng=rng,
                        )
                        if msl_idx >= 0:
                            enm_msl_id_next += 1
                            msg = f"spawn {enm_msl.call_sign[msl_idx]}"
                            recorder.add(recorder.event_bookmark(msg))

                simk += 1
                if get_simt() > simt_tol:
                    print("time out")
                    game_win = pln.is_alive(0).item()
                    if game_win:
                        print(f"win")
                    else:
                        print("lose")
                    trunc = True
                if trunc or term:
                    ss = ss_reset
                    episode += 1

                msg = recorder.merge()
                if msg:
                    recorder.write_local(msg)
                    if tacview_full:
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


def demo4LOSblock():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    n = 10
    seed = 3
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0, 2 * np.pi, (n,))
    sl = rng.uniform(0.5, 2.0, (n,))
    ps = np.stack(
        [
            sl * np.cos(theta),
            sl * np.sin(theta),
        ],
        axis=-1,
    )
    brs = rng.uniform(0.1, 0.3, (n,))

    t0 = time.time()
    yes = los_is_blocked(ps, brs)
    dt = max(time.time() - t0, 1e-3)
    print(f"calc_is_blocked {n} pts, {dt:.3f} s, {n/dt:.1f} pts/s")
    fig = plt.figure("演示,关闭本窗口后游戏开始")
    ax = fig.gca()
    for mask, clr, ls in [(yes, "r", "--"), (~yes, "g", "-")]:
        idxs = np.where(mask)[0]
        for i in idxs:
            x = ps[i, 0]
            y = ps[i, 1]
            ax.plot([0, x], [0, y], linestyle=ls, c=clr)
            circle = Circle((x, y), float(brs[i]), edgecolor=clr, facecolor="none")
            ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()


def main():
    demo4LOSblock()

    tacview_full = True  # 是否有高级版
    _is_stop_ = False
    _esc_ = 0  # 退出程序
    _trunc_ = False  # 单局中断
    _pause_ = False  # 暂停游戏
    esc_tol = 3
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
        a_vec_ = a_vec.copy()
        a_vec[ACT_IDX_DECOY] = 0
        a_vec[ACT_IDX_MSL] = 0
        return a_vec_

    # 定义处理键盘事件的回调函数
    def on_press(key: Union[keyboard.KeyCode, keyboard.Key, None]):
        if key is None:
            key
            return
        nonlocal a_vec
        key_ = _Keymap.fromKey(key)
        if HOTKEY_NX_P.equals(key_):
            a_vec[ACT_IDX_NX] = 1
        elif HOTKEY_NX_N.equals(key_):
            a_vec[ACT_IDX_NX] = -1
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
        if key is None:
            key
            return True
        nonlocal a_vec, _esc_, _trunc_
        key_ = _Keymap.fromKey(key)
        if HOTKEY_ESC.equals(key_):
            _esc_ += 1
            if _esc_ == esc_tol:
                print()
                return False
            else:
                print(f"再连按 {esc_tol - _esc_} 次 esc 退出")
        elif HOTKEY_RESET.equals(key_):
            print("重启...")
            _trunc_ = True
        elif HOTKEY_DECOY.equals(key_):
            a_vec[ACT_IDX_DECOY] = 1
        elif HOTKEY_DMSL.equals(key_):
            a_vec[ACT_IDX_MSL] = 1
        elif HOTKEY_PAUSE.equals(key_):
            set_pause(not is_paused())
        elif HOTKEY_NX_N.equals(key_) or HOTKEY_NX_P.equals(key_):
            a_vec[ACT_IDX_NX] = 0
        elif HOTKEY_NZ_N.equals(key_) or HOTKEY_NZ_P.equals(key_):
            a_vec[ACT_IDX_NZ] = 0
        elif HOTKEY_ROLL_N.equals(key_) or HOTKEY_ROLL_P.equals(key_):
            a_vec[ACT_IDX_ROLL] = 0

        if not HOTKEY_ESC.equals(key_):
            _esc_ = 0

    game_run
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
