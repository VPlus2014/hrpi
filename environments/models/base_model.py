from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING
import os
import logging
import numpy as np
import pymap3d
import torch
from abc import ABC, abstractmethod
from typing import Literal, Sequence, Union
from ..utils.math_torch import (
    quat_conj,
    quat_mul,
    quat_rotate,
    quat_rotate_inv,
    rpy2quat_inv,
    rpy2quat,
    uvw2alpha_beta,
)

LOGR = logging.getLogger(__name__)

_SupportedIndexType = Union[
    Sequence[int], torch.Tensor, int, slice, None, type(Ellipsis)
]


class BaseModel(ABC):
    STATUS_INACTIVATE = -1  # 未启动
    STATUS_ALIVE = 0  # 运行中
    STATUS_DYING = 1  # 暂停
    STATUS_DEAD = 2  # 结束

    logr = LOGR

    def __init__(
        self,
        id: torch.Tensor | int = 0,
        batch_size: int = 1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sim_step_size_ms: int = 1,
        use_gravity: bool = True,
        g: torch.Tensor | float = 9.8,  # 默认重力加速度 m/s^2
        use_eb=True,
        use_ew=True,
        use_wb=True,
        use_geodetic: bool = True,
        lat0: torch.Tensor | float = 0,
        lon0: torch.Tensor | float = 0,
        alt0: torch.Tensor | float = 0,
        use_mass=False,
        use_inertia=False,
        acmi_name: str = "",
        acmi_color: Literal["Red", "Blue"] | str = "Red",
        acmi_type: str = "",
        acmi_parent: str = "",
        call_sign: str = "",
        vis_radius: float = 1.0,
    ) -> None:
        """
        质点模型组 BaseModel
        Args:
            id (torch.Tensor | int): Tacview Object ID (整数形式).
            sim_step_size_ms (int, optional): 仿真步长, 单位:ms. Defaults to 1.
            batch_size (int): 组大小, Defaults to 1.
            device (torch.device, optional): 所在torch设备. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): torch浮点类型. Defaults to torch.float32.
            use_gravity (bool, optional): 是否启用重力(无则不支持计算重力). Defaults to True.
            g (float, optional): 重力加速度, 单位:m/s^2. Defaults to 9.8.
            use_eb (bool, optional): 是否启用地轴-体轴系状态. Defaults to True.
            use_ew (bool, optional): 是否启用地轴-风轴系状态. Defaults to True.
            use_wb (bool, optional): 是否启用风轴-体轴系状态. Defaults to True.
            use_geodetic (bool, optional): 是否使用地理坐标. Defaults to True.
            lat0 (torch.Tensor | float, optional): 坐标原点纬度, 单位:deg. float|shape: (B, 1)
            lon0 (torch.Tensor | float, optional): 坐标原点经度, 单位:deg. float|shape: (B, 1)
            alt0 (torch.Tensor | float, optional): 坐标原点高度, 单位:m. float|shape: (B, 1)
            acmi_color (Literal["Red", "Blue"] | str, optional): Tacview Color. Defaults to "Red".
            call_sign (str, optional): Tacview 呼号. Defaults to "".
            acmi_name (str, optional): Tacview 模型名(必须数据库中可检索否则无法正常渲染)
            acmi_type (str, optional): Tacview Object Type(符合ACMI标准). Defaults to "".
            acmi_parent (str, optional): Tacview 父对象 ID. Defaults to "".
            vis_radius (float, optional): 可视半径. Defaults to 1.0.
        """

        super().__init__()
        self._batch_size = batch_size
        self._device = device = torch.device(device)
        self._dtype = dtype
        _shape = [self.batch_size]
        _0f1 = torch.zeros(_shape + [1], device=device, dtype=dtype)
        self.id = torch.empty(_shape + [1], device=device, dtype=torch.int64)
        """Tacview Object ID, shape: (B,1)"""
        self.id.copy_(torch.asarray(id, device=device, dtype=torch.int64))

        self._sim_step_size_ms = sim_step_size_ms

        self.status = BaseModel.STATUS_INACTIVATE + torch.zeros(
            size=_shape + [1], dtype=torch.int64, device=device
        )  # shape: (B,1)
        self._sim_time_ms = torch.zeros(
            _shape + [1], dtype=torch.int64, device=device
        )  # (B,1)
        self.health_point = (
            torch.zeros(_shape + [1], device=device, dtype=dtype) + 100.0
        )
        """health point, shape: (B,1)"""
        self._vis_radius = torch.empty(_shape + [1], device=device, dtype=dtype)
        """可视半径, shape: (B,1)"""
        self._vis_radius.copy_(_0f1 + vis_radius)

        # simulation variables

        # cache variables
        self._pos_e = torch.empty(_shape + [3], device=device, dtype=dtype)
        """position in NED local frame, unit: m, shape: (B, 3)"""
        self._vel_e = torch.empty(_shape + [3], device=device, dtype=dtype)
        """velocity in NED local frame, unit: m/s, shape: (B, 3)"""

        # 常用缓存
        self._0f = _0 = _0f1
        self._1f = _1 = torch.ones_like(_0)
        self._e1f = torch.cat([_1, _0, _0], -1)
        self._e2f = torch.cat([_0, _1, _0], -1)
        self._e3f = torch.cat([_0, _0, _1], -1)

        self._g = torch.empty(_shape + [1], device=device, dtype=dtype)
        self._g.copy_(_0f1 + g)
        self.use_gravity = use_gravity
        """是否启用NED地轴系重力加速度向量缓存"""
        if use_gravity:
            """重力加速度, 单位: m/s^2"""
            self._g_e = torch.cat([_0, _0, self._g], -1)
            """重力加速度NED地轴系坐标, shape: (B, 3)"""

        # 本体飞控状态
        #
        self._tas = torch.zeros(
            _shape + [1], device=device, dtype=dtype
        )  # true air speed 真空速, unit: m/s, shape: (B,1)

        self._use_eb = use_eb
        if use_eb:
            self._vel_b = torch.zeros(
                _shape + [3], device=device, dtype=dtype
            )  # 惯性速度体轴坐标 (U,V,W) shape: (B,3)
            self._Q_eb = torch.zeros(
                _shape + [4], device=device, dtype=dtype
            )  # 地轴/体轴 四元数 shape: (B,4)
            self._rpy_eb = torch.zeros(
                _shape + [3], device=device, dtype=dtype
            )  # 地轴/体轴 欧拉角 (roll, pitch, yaw) shape:(B,3)
            self._omega_b = torch.zeros(
                _shape + [3], device=device, dtype=dtype
            )  # 体轴系下的旋转角速度 (P,Q,R) shape: (B,3)
        if use_inertia:
            assert use_eb, "use_inertia must be used with use_eb"
            self._I_b = torch.empty(_shape + [3, 3], device=device, dtype=dtype)
            self._I_b_inv = torch.empty(_shape + [3, 3], device=device, dtype=dtype)

        self._use_ew = use_ew
        if use_ew:
            self._rpy_ew = torch.zeros(
                _shape + [3], device=device, dtype=dtype
            )  # 地轴/风轴 欧拉角 (mu, gamma, chi) shape:(B,3)
            self._Q_ew = torch.zeros(
                _shape + [4], device=device, dtype=dtype
            )  # 地轴/风轴 四元数 shape: (B,4)
            self._vel_w = torch.zeros(
                _shape + [3], device=device, dtype=dtype
            )  # 惯性速度风轴分量 (TAS,0,0) shape: (B,3)

        self._use_wb = use_wb
        if use_wb:
            self._rpy_wb = torch.zeros(
                _shape + [3], device=device, dtype=dtype
            )  # 风轴/体轴 欧拉角 (0, alpha, -beta) shape:(B,3)
            self._Q_wb = torch.zeros(
                _shape + [4], device=device, dtype=dtype
            )  # 风轴/体轴 四元数 shape: (B,4)

        self._use_geodetic = use_geodetic
        if use_geodetic:
            self._blh0 = torch.empty(_shape + [3], device=device, dtype=dtype)
            """坐标原点(纬度,经度,高度), 单位:(deg,m), shape: (B, 3)"""
            self._blh0[..., 0:1] = lat0 + _0f1
            self._blh0[..., 1:2] = lon0 + _0f1
            self._blh0[..., 2:3] = alt0 + _0f1

            self._blh = torch.empty(_shape + [3], device=device, dtype=dtype)

            # self._lat.copy_(self._lat0)
            # self._lon.copy_(self._lon0)
            # self._alt.copy_(self._alt0)
            # pos_e->(lat, lon, alt)
            self._ppgt_ned2blh()

        self._use_mass = use_mass
        if use_mass:
            self._mass = torch.empty(_shape + [1], device=device, dtype=dtype)

        # if len(kwargs):
        #     msg = (
        #         self.__class__.__name__,
        #         f"received {len(kwargs)} unkown keyword arguments",
        #         list(kwargs.keys()),
        #     )
        #     LOGR.warning(msg)
        self._is_reset = False

        self.acmi_color = np.empty((batch_size,), dtype=object)
        """Tacview 颜色, shape: (B,)"""
        self.acmi_name = np.empty((batch_size,), dtype=object)
        """Tacview 模型名称, shape: (B,)"""
        self.acmi_type = np.empty((batch_size,), dtype=object)
        """Tacview 对象类型, shape: (B,)"""
        self.acmi_parent = np.empty((batch_size,), dtype=object)
        """Tacview 父对象 ID, shape: (B,)"""
        self.call_sign = np.empty((batch_size,), dtype=object)
        """Tacview 呼号, shape: (B,)"""
        for i in range(batch_size):
            self.acmi_color[i] = acmi_color
            self.acmi_name[i] = acmi_name
            self.acmi_type[i] = acmi_type
            self.acmi_parent[i] = acmi_parent
            self.call_sign[i] = call_sign

    @cached_property
    def batch_size(self) -> int:
        """组容量"""
        return self._batch_size

    @property
    def device(self) -> torch.device:
        """torch device"""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """torch dtype"""
        return self._dtype

    def proc_batch_index(self, batch_index: _SupportedIndexType = None):
        """对批索引做预处理"""
        if batch_index is None or batch_index is Ellipsis:
            idxs = slice(None)
        elif isinstance(batch_index, (slice, torch.Tensor)):
            idxs = batch_index
        elif isinstance(batch_index, int):
            idxs = [batch_index]
        else:
            idxs = torch.asarray(batch_index, device=self.device, dtype=torch.int64)
        # 不做检查
        return idxs

    @property
    def sim_step_size_ms(self) -> int:
        """仿真步长, 单位:ms"""
        return self._sim_step_size_ms

    @property
    def sim_step_size_s(self) -> float:
        """仿真步长, 单位:s"""
        return self._sim_step_size_ms * 1e-3

    def vis_radius(self, index: _SupportedIndexType = None):
        """实际遮蔽半径, 单位:m, shape: (B,1)"""
        index = self.proc_batch_index(index)
        return self._vis_radius[index] * self.is_alive(index)

    def position_e(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """position in NED local frame, unit: m, shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._pos_e[batch_index, :]

    def blh(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """当前(纬度,经度,高度), 单位:(deg,m), shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._blh[batch_index, :]

    def blh0(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """坐标原点(纬度,经度,高度), 单位:(deg,m), shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._blh0[batch_index, :]

    def longitude_deg(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """longitude, unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._blh[batch_index, 1:2]

    def latitude_deg(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """latitude, unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._blh[batch_index, 0:1]

    def altitude_m(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """海拔 altitude, unit: m, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._blh[batch_index, 2:3]

    def g_e(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """NED地轴系重力加速度向量"""
        batch_index = self.proc_batch_index(batch_index)
        return self._g_e[batch_index, :]

    def sim_time_ms(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """model simulation time, unit: ms, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._sim_time_ms[batch_index, :]

    def sim_time_s(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """model simulation time, unit: s, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self.sim_time_ms(batch_index) * 1e-3

    @abstractmethod
    def reset(self, batch_index: _SupportedIndexType = None):
        """状态复位"""
        batch_index = self.proc_batch_index(batch_index)

        self._sim_time_ms[batch_index] = 0.0
        self.set_status(BaseModel.STATUS_INACTIVATE, batch_index)

        self._is_reset = True

    def set_status(
        self, status: int | torch.Tensor, dst_idx: _SupportedIndexType = None
    ):
        """设置仿真生命状态"""
        dst_idx = self.proc_batch_index(dst_idx)
        self.status[dst_idx] = status

    @abstractmethod
    def run(self, batch_index: _SupportedIndexType = None):
        """仿真推进"""
        assert self._is_reset, "Model must be reset before running"
        self.update_time(batch_index)

    def update_time(self, batch_index: _SupportedIndexType = None):
        """更新仿真时间"""
        batch_index = self.proc_batch_index(batch_index)
        self._sim_time_ms[batch_index] += self._sim_step_size_ms

    def is_alive(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """判断是否存活"""
        batch_index = self.proc_batch_index(batch_index)
        return self.status[batch_index] == self.STATUS_ALIVE

    def is_dying(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """判断是否即将死亡"""
        batch_index = self.proc_batch_index(batch_index)
        return self.status[batch_index] == self.STATUS_DYING

    def is_dead(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """判断是否死亡"""
        batch_index = self.proc_batch_index(batch_index)
        return self.status[batch_index] == self.STATUS_DEAD

    def Q_eb(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """地轴系/体轴系四元数"""
        batch_index = self.proc_batch_index(batch_index)
        return self._Q_eb[batch_index, :]

    def Q_wb(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """风轴系/体轴系四元数"""
        batch_index = self.proc_batch_index(batch_index)
        return self._Q_wb[batch_index, :]

    def Q_ew(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """地轴系/风轴系四元数"""
        batch_index = self.proc_batch_index(batch_index)
        return self._Q_ew[batch_index, :]

    def tas(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """true air speed, unit: m/s, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._tas[batch_index, :]

    def velocity_b(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """惯性速度 NED体轴系坐标 (U,V,W), unit: m/s, shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._vel_b[batch_index, :]

    def velocity_e(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """惯性速度 NED地轴系坐标 (V_N, V_E, V_D), unit: m/s, shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._vel_e[batch_index, :]

    def velocity_w(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """惯性速度 NED风轴系坐标 (TAS,0,0), unit: m/s, shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._vel_w[batch_index, :]

    def rpy_eb(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """体轴系 (roll, pitch, yaw) unit: rad, shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_eb[batch_index, :]

    def rpy_ew(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """风轴系 (mu, gamma, chi) unit: rad, shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_ew[batch_index, :]

    def activate(self, batch_index: _SupportedIndexType = None):
        batch_index = self.proc_batch_index(batch_index)
        self.status[batch_index] = BaseModel.STATUS_ALIVE

    def mass(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """质量, unit: kg, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._mass[batch_index, :]

    def inertia(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """体轴惯性矩阵, unit: kg*m^2, shape: (B, 3, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._I_b[batch_index, :, :]

    # propagation modules
    def _ppgt_z2alt(self, batch_index: _SupportedIndexType = None):
        """高度->海拔"""
        batch_index = self.proc_batch_index(batch_index)
        self._blh[batch_index, 2:3] = (
            self._blh0[batch_index, 2:3] - self._pos_e[batch_index, 2:3]
        )

    def _ppgt_ned2blh(self, batch_index: _SupportedIndexType = None):
        """NED地轴系坐标->纬经高"""
        index = self.proc_batch_index(batch_index)
        p_np = self._pos_e[index, :].cpu().numpy()
        n, e, d = np.split(p_np, 3, axis=-1)
        lat0, lon0, alt0 = np.split(self._blh0[index, :].cpu().numpy(), 3, axis=-1)
        lat, lon, alt = pymap3d.ned2geodetic(n, e, d, lat0, lon0, alt0)
        blh = np.concatenate([lat, lon, alt], axis=-1)
        self._blh[index, :] = torch.asarray(blh, device=self.device, dtype=self.dtype)

    def _ppgt_rpy_eb2Qeb(self, batch_index: _SupportedIndexType = None):
        """地轴/体轴 欧拉角->四元数"""
        batch_index = self.proc_batch_index(batch_index)
        self._Q_eb[batch_index, :] = rpy2quat(self._rpy_eb[batch_index, :])

    def _ppgt_rpy_wb2Qwb(self, batch_index: _SupportedIndexType = None):
        """体轴/风轴 欧拉角->四元数"""
        batch_index = self.proc_batch_index(batch_index)
        self._Q_wb[batch_index, :] = rpy2quat(self._rpy_wb[batch_index, :])

    def _ppgt_rpy_ew2Qew(self, batch_index: _SupportedIndexType = None):
        """地轴/风轴 欧拉角->四元数"""
        batch_index = self.proc_batch_index(batch_index)
        self._Q_ew[batch_index, :] = rpy2quat(self._rpy_ew[batch_index, :])

    def _ppgt_Qeb2rpy_eb(self, batch_index: _SupportedIndexType = None):
        """地轴/体轴 四元数->欧拉角"""
        batch_index = self.proc_batch_index(batch_index)
        self._rpy_eb[batch_index, :] = rpy2quat_inv(
            self._Q_eb[batch_index, :], self._rpy_eb[batch_index, 0:1]
        )

    def _ppgt_Qwb2rpy_wb(self, batch_index: _SupportedIndexType = None):
        """风轴/体轴 四元数->欧拉角"""
        batch_index = self.proc_batch_index(batch_index)
        self._rpy_wb[batch_index, :] = rpy2quat_inv(
            self._Q_wb[batch_index, :], self._rpy_wb[batch_index, 0:1]
        )
        self._rpy_wb[batch_index, 0:1] = 0  # 风轴到体轴不定义滚转,恒为0

    def _ppgt_Qew2rpy_ew(self, batch_index: _SupportedIndexType = None):
        """地轴/风轴 四元数->欧拉角"""
        batch_index = self.proc_batch_index(batch_index)
        self._rpy_ew[batch_index, :] = rpy2quat_inv(
            self._Q_ew[batch_index, :], self._rpy_ew[batch_index, 0:1]
        )

    def _ppgt_Vb2Ve(self, batch_index: _SupportedIndexType = None):
        """惯性速度 体轴系->地轴系"""
        batch_index = self.proc_batch_index(batch_index)
        self._vel_e.copy_(quat_rotate(self._Q_eb, self._vel_b))

    def _ppgt_Ve2Vb(self, batch_index: _SupportedIndexType = None):
        """惯性速度 地轴系->体轴系"""
        batch_index = self.proc_batch_index(batch_index)
        self._vel_b[batch_index, :] = quat_rotate_inv(
            self._Q_eb[batch_index, :], self._vel_e[batch_index, :]
        )

    def _ppgt_Ve2tas(self, batch_index: _SupportedIndexType = None):
        """地轴系惯性速度->真空速tas"""
        batch_index = self.proc_batch_index(batch_index)
        self._tas[batch_index, :] = torch.norm(
            self._vel_e[batch_index, :], p=2, dim=-1, keepdim=True
        )

    def _ppgt_Vb2tas(self, batch_index: _SupportedIndexType = None):
        """体轴系惯性速度->真空速tas"""
        batch_index = self.proc_batch_index(batch_index)
        self._tas[batch_index, :] = torch.norm(
            self._vel_b[batch_index, :], p=2, dim=-1, keepdim=True
        )

    def _ppgt_Vw2tas(self, batch_index: _SupportedIndexType = None):
        """风轴系惯性速度->真空速tas"""
        batch_index = self.proc_batch_index(batch_index)
        self._tas[batch_index, :] = self._vel_w[batch_index, 0:1]

    def _ppgt_tas2Vw(self, batch_index: _SupportedIndexType = None):
        """真空速->风轴系惯性速度"""
        batch_index = self.proc_batch_index(batch_index)
        self._vel_w[batch_index, 0:1] = self._tas[batch_index, :]
        self._vel_w[batch_index, 1:3] = 0.0

    def _ppgt_Vw2Vb(self, batch_index: _SupportedIndexType = None):
        """真空速->体轴系惯性速度"""
        batch_index = self.proc_batch_index(batch_index)
        self._vel_b[batch_index, :] = quat_rotate_inv(
            self._Q_wb[batch_index, :], self._vel_w[batch_index, :]
        )

    def _ppgt_Vw2Ve(self, batch_index: _SupportedIndexType = None):
        """真空速->地轴系惯性速度"""
        batch_index = self.proc_batch_index(batch_index)
        self._vel_e[batch_index, :] = quat_rotate(
            self._Q_ew[batch_index, :], self._vel_w[batch_index, :]
        )

    def _ppgt_Vb2rpy_wb(self, batch_index: _SupportedIndexType = None):
        """体轴系惯性速度-> 体轴/风轴 欧拉角(迎角,侧滑角)"""
        batch_index = self.proc_batch_index(batch_index)
        alpha, beta = uvw2alpha_beta(self._vel_b[batch_index, :])
        self.set_alpha(alpha, batch_index)
        self.set_beta(beta, batch_index)

    def _ppgt_QebQwb2Qew(self, batch_index: _SupportedIndexType = None):
        """地轴/体轴 & 风轴/体轴 -> 地轴/风轴 四元数"""
        batch_index = self.proc_batch_index(batch_index)
        self._Q_ew[batch_index, :] = quat_mul(
            self._Q_eb[batch_index, :], quat_conj(self._Q_wb[batch_index, :])
        )

    def _ppgt_QewQwb_to_Qeb(self, batch_index: _SupportedIndexType = None):
        """地轴/风轴 & 风轴/体轴 -> 地轴/体轴 四元数"""
        batch_index = self.proc_batch_index(batch_index)
        self._Q_eb[batch_index, :] = quat_mul(
            self._Q_ew[batch_index, :], self._Q_wb[batch_index, :]
        )

    def roll(self, batch_index: _SupportedIndexType = None):
        """体轴滚转角 in (-pi,pi], unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_eb[batch_index, 0:1]

    def pitch(self, batch_index: _SupportedIndexType = None):
        """体轴俯仰角 in [-pi/2,pi/2], unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_eb[batch_index, 1:2]

    def yaw(self, batch_index: _SupportedIndexType = None):
        """体轴偏航角 in (-pi,pi], unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_eb[batch_index, 2:3]

    def alpha(self, batch_index: _SupportedIndexType = None):
        """迎角 in [-pi/2,pi/2], unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_wb[batch_index, 1:2]

    def beta(self, batch_index: _SupportedIndexType = None):
        """侧滑角 in [-pi,pi], unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return -(self._rpy_wb[batch_index, 2:3])

    def mu(self, batch_index: _SupportedIndexType = None):
        """速度系滚转角 in (-pi,pi], unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_ew[batch_index, 0:1]

    def gamma(self, batch_index: _SupportedIndexType = None):
        """速度系俯仰角 in [-pi/2,pi/2], unit: rad, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_ew[batch_index, 1:2]

    def chi(self, batch_index: _SupportedIndexType = None):
        """速度系偏航角 in (-pi,pi], shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._rpy_ew[batch_index, 2:3]

    def set_roll(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """赋值 体轴滚转角(无级联操作) 等价于 roll[dst_index,:]=value"""
        dst_index = self.proc_batch_index(dst_index)
        self._rpy_eb[dst_index, 0:1] = value

    def set_pitch(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """赋值 体轴俯仰角(无级联操作)  等价于 pitch[dst_index,:]=value"""
        dst_index = self.proc_batch_index(dst_index)
        self._rpy_eb[dst_index, 1:2] = value

    def set_yaw(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """赋值 体轴偏航角(无级联操作) 等价于 yaw[dst_index,:]=value"""
        dst_index = self.proc_batch_index(dst_index)
        self._rpy_eb[dst_index, 2:3] = value

    def set_alpha(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """赋值 迎角(无级联操作) 等价于 alpha[dst_index,:]=value"""
        dst_index = self.proc_batch_index(dst_index)
        self._rpy_wb[dst_index, 1:2] = value

    def set_beta(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """赋值 侧滑角(无级联操作) 等价于 beta[dst_index,:]=value"""
        dst_index = self.proc_batch_index(dst_index)
        self._rpy_wb[dst_index, 2:3] = -value

    def set_mu(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """赋值 航迹滚转角(无级联操作) 等价于 mu[dst_index,:]=value"""
        dst_index = self.proc_batch_index(dst_index)
        self._rpy_ew[dst_index, 0:1] = value

    def set_gamma(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """赋值 航迹俯仰角(无级联操作) 等价于 gamma[dst_index,:]=value"""
        dst_index = self.proc_batch_index(dst_index)
        self._rpy_ew[dst_index, 1:2] = value

    def set_chi(self, value: torch.Tensor, dst_index: _SupportedIndexType = None):
        """赋值 航迹偏航角(无级联操作) 等价于 chi[dst_index,:]=value"""
        dst_index = self.proc_batch_index(dst_index)
        self._rpy_ew[dst_index, 2:3] = value
