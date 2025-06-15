from __future__ import annotations
from typing import TYPE_CHECKING
import os
import logging
import numpy as np
import pymap3d
import torch
from abc import ABC, abstractmethod
from typing import Literal, Sequence, Union
from ..utils.math_pt import (
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
    Sequence[int], torch.Tensor, torch.LongTensor, int, slice, type(Ellipsis)
]
_ShapeLike = Union[Sequence[int], torch.Size]
_SliceNone = slice(None)
DeviceLike = Union[str, torch.device]


class BaseModel(ABC):
    STATUS_INACTIVATE = -1  # 未启动
    STATUS_ALIVE = 0  # 运行中
    STATUS_DYING = 1  # 即将结束
    STATUS_DEAD = 2  # 结束

    logr = LOGR
    DEBUG: bool = False

    def __init__(
        self,
        group_shape: Sequence[int] | int = 1,
        device: DeviceLike = "cpu",
        dtype: torch.dtype = torch.float64,
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
        vis_radius: float = 1.0,
        acmi_id: torch.Tensor | int = 0,
        acmi_name: str | Sequence[str] = "",
        acmi_color: str | Sequence[str] = "Red",
        acmi_type: str | Sequence[str] = "",
        call_sign: str | Sequence[str] = "",
        acmi_parent: torch.Tensor | int = 0,
    ) -> None:
        """
        质点模型组 BaseModel
        约定:
            所有非标量数据都是矩阵;
        Args:
            sim_step_size_ms (int, optional): 仿真步长, 单位:ms. Defaults to 1.
            group_shape (int|Sequence[int], optional): 组容量N/形状(N1,...,Nn), Defaults to 1.
            device (torch.device, optional): 所在torch设备. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): torch浮点类型. Defaults to torch.float32.
            use_gravity (bool, optional): 是否启用重力(无则不支持计算重力). Defaults to True.
            g (float, optional): 重力加速度, 单位:m/s^2. Defaults to 9.8.
            use_eb (bool, optional): 是否启用地轴-体轴系状态. Defaults to True.
            use_ew (bool, optional): 是否启用地轴-风轴系状态. Defaults to True.
            use_wb (bool, optional): 是否启用风轴-体轴系状态. Defaults to True.
            use_geodetic (bool, optional): 是否使用地理坐标. Defaults to True.
            lat0 (torch.Tensor | float, optional): 坐标原点纬度, 单位:deg. float|shape: (...,N,1)
            lon0 (torch.Tensor | float, optional): 坐标原点经度, 单位:deg. float|shape: (...,N,1)
            alt0 (torch.Tensor | float, optional): 坐标原点高度, 单位:m. float|shape: (...,N,1)
            acmi_id (torch.Tensor | int, optional): Tacview Object ID (整数形式).
            acmi_color (Literal["Red", "Blue"] | str, optional): Tacview Color. Defaults to "Red".
            acmi_name (str, Sequence[str], optional): Tacview 模型名(必须数据库中可检索否则无法正常渲染)
            acmi_type (str, Sequence[str], optional): Tacview Object Type(符合ACMI标准). Defaults to "".
            acmi_parent (torch.Tensor | int, optional): Tacview 父对象 ID. Defaults to "".
            call_sign (str, Sequence[str], optional): Tacview 呼号. Defaults to "".
            vis_radius (float, optional): 可视半径. Defaults to 1.0.
        """
        super().__init__()
        self._device = _device = torch.device(device)
        self._dtype = dtype
        assert dtype in (
            torch.float32,
            torch.float64,
        ), "dtype only support torch float32/float64"
        if isinstance(group_shape, int):
            group_shape = (group_shape,)
        assert len(group_shape) > 0, ("group_shape must be not empty", group_shape)
        self._group_shape = _shape_head = (*group_shape,)
        self._batch_size = int(np.prod(_shape_head))
        _0f1 = torch.zeros(_shape_head + (1,), device=_device, dtype=dtype)

        self._MASK4VEC = torch.ones(
            (1,) * (len(_shape_head) + 1), device=_device, dtype=torch.bool
        )

        self.status = BaseModel.STATUS_INACTIVATE + torch.zeros(
            size=_shape_head + (1,), dtype=torch.int64, device=_device
        )
        """仿真运行状态, shape: (...,N,1)"""
        self._sim_step_size_ms = sim_step_size_ms
        """仿真步长, unit: ms; int"""
        self._sim_time_ms = torch.zeros(
            _shape_head + (1,), dtype=torch.int64, device=_device
        )
        """仿真时钟 unit: ms, shape: (...,N,1)"""
        self.health_point = (
            torch.zeros(_shape_head + (1,), device=_device, dtype=dtype) + 100.0
        )
        """health point, shape: (...,N,1)"""
        self._vis_radius = torch.empty(_shape_head + (1,), device=_device, dtype=dtype)
        """可视半径, shape: (...,N,1)"""
        self._vis_radius.copy_(_0f1 + vis_radius)

        # simulation variables

        # cache variables
        self._pos_e = torch.empty(_shape_head + (3,), device=_device, dtype=dtype)
        """position in NED local frame, unit: m, shape: (...,N,3)"""
        self._vel_e = torch.empty(_shape_head + (3,), device=_device, dtype=dtype)
        """velocity in NED local frame, unit: m/s, shape: (...,N,3)"""

        # 常用缓存
        self._0F = _0 = _0f1  # (...,N,1)
        self._1F = _1 = torch.ones_like(_0)  # (...,N,1)
        self._E1F = torch.cat([_1, _0, _0], -1)
        self._E2F = torch.cat([_0, _1, _0], -1)
        self._E3F = torch.cat([_0, _0, _1], -1)

        self._g = torch.empty(_shape_head + (1,), device=_device, dtype=dtype)
        self._g.copy_(_0f1 + g)
        self.use_gravity = use_gravity
        """是否启用NED地轴系重力加速度向量缓存, shape: (...,N,1)"""
        if use_gravity:
            """重力加速度, 单位: m/s^2"""
            self._g_e = torch.cat([_0, _0, self._g], -1)
            """重力加速度NED地轴系坐标, shape: (...,N,3)"""

        # 本体飞控状态
        #
        self._tas = torch.zeros(_shape_head + (1,), device=_device, dtype=dtype)
        """true air speed 真空速, unit: m/s, shape: (...,N,1)"""

        self._use_eb = use_eb
        if use_eb:
            self._vel_b = torch.zeros(_shape_head + (3,), device=_device, dtype=dtype)
            """惯性速度体轴坐标 (U,V,W) shape: (...,N,3)"""
            self._Q_eb = torch.zeros(_shape_head + (4,), device=_device, dtype=dtype)
            """地轴/体轴 四元数 shape: (...,N,4)"""
            self._rpy_eb = torch.zeros(_shape_head + (3,), device=_device, dtype=dtype)
            """地轴/体轴 欧拉角 (roll, pitch, yaw) shape:(...,N,3)"""
            self._omega_b = torch.zeros(_shape_head + (3,), device=_device, dtype=dtype)
            """体轴系下的旋转角速度 (P,Q,R) shape: (...,N,3)"""
        if use_inertia:
            assert use_eb, "use_inertia must be used with use_eb"
            self._I_b = torch.empty(_shape_head + (3, 3), device=_device, dtype=dtype)
            """体轴惯性矩 shape: (...,N,3,3)"""
            self._I_b_inv = torch.empty(
                _shape_head + (3, 3), device=_device, dtype=dtype
            )

        self._use_ew = use_ew
        if use_ew:
            self._rpy_ew = torch.zeros(_shape_head + (3,), device=_device, dtype=dtype)
            """地轴/风轴 欧拉角 (mu, gamma, chi) shape:(...,N,3)"""
            self._Q_ew = torch.zeros(_shape_head + (4,), device=_device, dtype=dtype)
            """地轴/风轴 四元数 shape: (...,N,4)"""
            self._vel_w = torch.zeros(_shape_head + (3,), device=_device, dtype=dtype)
            """惯性速度风轴分量 (TAS,0,0) shape: (...,N,3)"""

        self._use_wb = use_wb
        if use_wb:
            self._rpy_wb = torch.zeros(_shape_head + (3,), device=_device, dtype=dtype)
            """风轴/体轴 欧拉角 (0, alpha, -beta) shape:(...,N,3)"""
            self._Q_wb = torch.zeros(_shape_head + (4,), device=_device, dtype=dtype)
            """风轴/体轴 四元数 shape: (...,N,4)"""

        self._use_geodetic = use_geodetic
        if use_geodetic:
            self._blh0 = torch.empty(_shape_head + (3,), device=_device, dtype=dtype)
            """坐标原点(纬度,经度,高度), 单位:(deg,m), shape: (...,N,3)"""
            self._blh0[..., 0:1] = lat0 + _0f1
            self._blh0[..., 1:2] = lon0 + _0f1
            self._blh0[..., 2:3] = alt0 + _0f1

            self._blh = torch.empty(_shape_head + (3,), device=_device, dtype=dtype)
            """当前 (纬度,经度,高度), 单位:(deg,m), shape: (...,N,3)"""

            # self._lat.copy_(self._lat0)
            # self._lon.copy_(self._lon0)
            # self._alt.copy_(self._alt0)
            # pos_e->(lat, lon, alt)
            # self._ppgt_ned2blh(None)

        self._use_mass = use_mass
        if use_mass:
            self._mass = torch.empty(_shape_head + (1,), device=_device, dtype=dtype)

        # if len(kwargs):
        #     msg = (
        #         self.__class__.__name__,
        #         f"received {len(kwargs)} unkown keyword arguments",
        #         list(kwargs.keys()),
        #     )
        #     LOGR.warning(msg)

        self.acmi_id = torch.empty(
            _shape_head + (1,), device=_device, dtype=torch.int64
        )
        """Tacview Object ID, shape: (...,N,1) Tensor[int64]"""
        self.acmi_id.copy_(torch.asarray(acmi_id, device=_device, dtype=torch.int64))

        self.acmi_color = np.empty(_shape_head + (1,), dtype=object)
        """Tacview 颜色, shape: (...,N,1) NDArray[str]"""
        self.acmi_color[..., 0] = acmi_color

        self.acmi_name = np.empty(_shape_head + (1,), dtype=object)
        """Tacview 模型名称, shape: (...,N,1) NDArray[str]"""
        self.acmi_name[..., 0] = acmi_name

        self.acmi_type = np.empty(_shape_head + (1,), dtype=object)
        """Tacview 对象类型, shape: (...,N,1) NDArray[str]"""
        self.acmi_type[..., 0] = acmi_type

        self.acmi_parent = torch.empty(
            _shape_head + (1,), device=_device, dtype=torch.int64
        )
        """Tacview 父对象 ID, shape: (...,N,1) Tensor[int64]"""
        self.acmi_parent.copy_(
            torch.asarray(acmi_parent, device=_device, dtype=torch.int64)
        )

        self.call_sign = np.empty(_shape_head + (1,), dtype=object)
        """Tacview 呼号, shape: (...,N,1) NDArray[str]"""
        self.call_sign[..., 0] = call_sign

    @property
    def batch_size(self) -> int:
        """组容量(拉平以后)"""
        return self._batch_size

    @property
    def group_shape(self):
        """组形状"""
        return self._group_shape

    @property
    def device(self) -> torch.device:
        """torch device"""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """torch dtype"""
        return self._dtype

    def proc_index(self, index: _SupportedIndexType | None):
        """对索引做预处理"""
        if index is None or index == Ellipsis:
            idxs = slice(None)
        elif isinstance(index, (slice, torch.Tensor)):
            idxs = index
        elif isinstance(index, int):
            idxs = [index]
        elif isinstance(index, Sequence):
            idxs = index
        else:
            idxs = torch.asarray(index, device=self.device, dtype=torch.int64)
        # 不做检查
        return idxs

    def proc_mask(self, mask: torch.Tensor | None = None):
        if mask is None or mask is Ellipsis or mask == slice(None):
            msk = self._MASK4VEC
        elif isinstance(mask, torch.Tensor):
            assert mask.shape == self._group_shape + (1, 1), "mask shape error"
            assert mask.dtype == torch.bool, "mask dtype error"
            msk = mask.to(self.device)
        else:
            raise TypeError("mask must be a tensor or None")
        return msk

    @property
    def sim_step_size_ms(self) -> int:
        """仿真步长, 单位:ms"""
        return self._sim_step_size_ms

    @property
    def sim_step_size_s(self) -> float:
        """仿真步长, 单位:s"""
        return self._sim_step_size_ms * 1e-3

    def vis_radius(self):
        """遮蔽半径, 单位:m, shape: (...,N,1)"""
        return self._vis_radius

    def position_e(self) -> torch.Tensor:
        """position in NED local frame, unit: m, shape: (...,N,3)"""
        return self._pos_e

    def blh(self) -> torch.Tensor:
        """当前(纬度,经度,高度), 单位:(deg,m), shape: (...,N,3)"""
        return self._blh

    def blh0(self) -> torch.Tensor:
        """坐标原点(纬度,经度,高度), 单位:(deg,m), shape: (...,N,3)"""
        return self._blh0[..., :]

    def longitude_deg(self) -> torch.Tensor:
        """longitude, unit: rad, shape: (...,N,1)"""
        return self._blh[..., 1:2]

    def latitude_deg(self) -> torch.Tensor:
        """latitude, unit: rad, shape: (...,N,1)"""
        return self._blh[..., 0:1]

    def altitude_m(self) -> torch.Tensor:
        """海拔 altitude, unit: m, shape: (...,N,1)"""
        return self._blh[..., 2:3]

    def g_e(self) -> torch.Tensor:
        """NED地轴系重力加速度向量, unit: m/s^2, shape: (...,N,3)"""
        return self._g_e

    def sim_time_ms(self) -> torch.Tensor:
        """model simulation time, unit: ms, shape: (...,N,1)"""
        return self._sim_time_ms

    def sim_time_s(
        self,
    ) -> torch.Tensor:
        """model simulation time, unit: s, shape: (...,N,1)"""
        return self.sim_time_ms() * 1e-3

    @abstractmethod
    def reset(self, index: _SupportedIndexType | None):
        """状态复位"""
        index = self.proc_index(index)

        self._sim_time_ms[index] = 0.0
        self.set_status(self.STATUS_INACTIVATE, index)

    def set_status(
        self, status: int | torch.Tensor, dst_idx: _SupportedIndexType | None
    ):
        """设置仿真运行状态"""
        dst_idx = self.proc_index(dst_idx)
        self.status[dst_idx, :] = status

    def activate(self, batch_index: _SupportedIndexType | None):
        """运行状态->激活"""
        batch_index = self.proc_index(batch_index)
        self.status[batch_index, :] = self.STATUS_ALIVE

    @abstractmethod
    def run(self, batch_index: _SupportedIndexType | None):
        """仿真推进"""
        self.update_sim_time(batch_index)

    def update_sim_time(self, batch_index: _SupportedIndexType | None):
        """更新仿真时间"""
        batch_index = self.proc_index(batch_index)
        self._sim_time_ms[batch_index] += self._sim_step_size_ms

    def is_ready(self) -> torch.Tensor:
        """判断sims是否就绪, shape=(...,N,1)"""
        return self.status == self.STATUS_INACTIVATE

    def is_alive(self) -> torch.Tensor:
        """判断sims是否运行, shape=(...,N,1)"""
        return self.status == self.STATUS_ALIVE

    def is_dying(self) -> torch.Tensor:
        """判断sims是否即将死亡, shape=(...,N,1)"""
        return self.status == self.STATUS_DYING

    def is_dead(self) -> torch.Tensor:
        """判断sims是否死亡, shape=(...,N,1)"""
        return self.status == self.STATUS_DEAD

    def Q_eb(self) -> torch.Tensor:
        """地轴系/体轴系四元数"""
        return self._Q_eb

    def Q_wb(self) -> torch.Tensor:
        """风轴系/体轴系四元数"""
        return self._Q_wb

    def Q_ew(self) -> torch.Tensor:
        """地轴系/风轴系四元数"""
        return self._Q_ew

    def tas(self) -> torch.Tensor:
        """true air speed, unit: m/s, shape: (...,N,1)"""
        return self._tas

    def velocity_b(self) -> torch.Tensor:
        """惯性速度 NED体轴系坐标 (U,V,W), unit: m/s, shape: (...,N,3)"""
        return self._vel_b

    def velocity_e(self) -> torch.Tensor:
        """惯性速度 NED地轴系坐标 (V_N, V_E, V_D), unit: m/s, shape: (...,N,3)"""
        return self._vel_e

    def velocity_w(self) -> torch.Tensor:
        """惯性速度 NED风轴系坐标 (TAS,0,0), unit: m/s, shape: (...,N,3)"""

        return self._vel_w

    def rpy_eb(self) -> torch.Tensor:
        """体轴系 (roll, pitch, yaw) unit: rad, shape: (...,N,3)"""
        return self._rpy_eb

    def rpy_ew(self) -> torch.Tensor:
        """风轴系 (mu, gamma, chi) unit: rad, shape: (...,N,3)"""
        return self._rpy_ew

    def mass(self) -> torch.Tensor:
        """质量, unit: kg, shape: (...,N,1)"""
        return self._mass

    def inertia(self) -> torch.Tensor:
        """体轴惯性矩阵, unit: kg*m^2, shape: (...,N,3,3)"""
        return self._I_b

    # propagation modules
    def _ppgt_z2alt(self, batch_index: _SupportedIndexType | None):
        """Z坐标->海拔"""
        batch_index = self.proc_index(batch_index)
        self._blh[batch_index, 2:3] = (
            self._blh0[batch_index, 2:3] - self._pos_e[batch_index, 2:3]
        )

    def _ppgt_alt2z(self, batch_index: _SupportedIndexType | None):
        """海拔->Z坐标"""
        batch_index = self.proc_index(batch_index)
        self._pos_e[batch_index, 2:3] = (
            self._blh0[batch_index, 2:3] - self._blh[batch_index, 2:3]
        )

    def _ppgt_ned2blh(self, batch_index: _SupportedIndexType | None):
        """NED地轴系坐标->纬经高"""
        index = self.proc_index(batch_index)
        p_np = self._pos_e[index, :].cpu().numpy()
        n, e, d = np.split(p_np, 3, axis=-1)
        lat0, lon0, alt0 = np.split(self._blh0[index, :].cpu().numpy(), 3, axis=-1)
        lat, lon, alt = pymap3d.ned2geodetic(n, e, d, lat0, lon0, alt0)
        blh = np.concatenate([lat, lon, alt], axis=-1)
        self._blh[index, :] = torch.asarray(blh, device=self.device, dtype=self.dtype)

    def _ppgt_rpy_eb2Qeb(self, batch_index: _SupportedIndexType | None):
        """地轴/体轴 欧拉角->四元数"""
        batch_index = self.proc_index(batch_index)
        self._Q_eb[batch_index, :] = rpy2quat(self._rpy_eb[batch_index, :])

    def _ppgt_rpy_wb2Qwb(self, batch_index: _SupportedIndexType | None):
        """体轴/风轴 欧拉角->四元数"""
        batch_index = self.proc_index(batch_index)
        self._Q_wb[batch_index, :] = rpy2quat(self._rpy_wb[batch_index, :])

    def _ppgt_rpy_ew2Qew(self, batch_index: _SupportedIndexType | None):
        """地轴/风轴 欧拉角->四元数"""
        batch_index = self.proc_index(batch_index)
        self._Q_ew[batch_index, :] = rpy2quat(self._rpy_ew[batch_index, :])

    def _ppgt_Qeb2rpy_eb(self, batch_index: _SupportedIndexType | None):
        """地轴/体轴 四元数->欧拉角"""
        batch_index = self.proc_index(batch_index)
        self._rpy_eb[batch_index, :] = rpy2quat_inv(
            self._Q_eb[batch_index, :], self._rpy_eb[batch_index, 0:1]
        )

    def _ppgt_Qwb2rpy_wb(self, batch_index: _SupportedIndexType | None):
        """风轴/体轴 四元数->欧拉角"""
        batch_index = self.proc_index(batch_index)
        self._rpy_wb[batch_index, :] = rpy2quat_inv(
            self._Q_wb[batch_index, :], self._rpy_wb[batch_index, 0:1]
        )
        self._rpy_wb[batch_index, 0:1] = 0  # 风轴到体轴不定义滚转,恒为0

    def _ppgt_Qew2rpy_ew(self, batch_index: _SupportedIndexType | None):
        """地轴/风轴 四元数->欧拉角"""
        batch_index = self.proc_index(batch_index)
        self._rpy_ew[batch_index, :] = rpy2quat_inv(
            self._Q_ew[batch_index, :], self._rpy_ew[batch_index, 0:1]
        )

    def _ppgt_Vb2Ve(self, batch_index: _SupportedIndexType | None):
        """惯性速度 体轴系->地轴系"""
        batch_index = self.proc_index(batch_index)
        self._vel_e.copy_(quat_rotate(self._Q_eb, self._vel_b))

    def _ppgt_Ve2Vb(self, batch_index: _SupportedIndexType | None):
        """惯性速度 地轴系->体轴系"""
        batch_index = self.proc_index(batch_index)
        self._vel_b[batch_index, :] = quat_rotate_inv(
            self._Q_eb[batch_index, :], self._vel_e[batch_index, :]
        )

    def _ppgt_Ve2tas(self, batch_index: _SupportedIndexType | None):
        """地轴系惯性速度->真空速tas"""
        batch_index = self.proc_index(batch_index)
        self._tas[batch_index, :] = torch.norm(
            self._vel_e[batch_index, :], p=2, dim=-1, keepdim=True
        )

    def _ppgt_Vb2tas(self, batch_index: _SupportedIndexType | None):
        """体轴系惯性速度->真空速tas"""
        batch_index = self.proc_index(batch_index)
        self._tas[batch_index, :] = torch.norm(
            self._vel_b[batch_index, :], p=2, dim=-1, keepdim=True
        )

    def _ppgt_Vw2tas(self, batch_index: _SupportedIndexType | None):
        """风轴系惯性速度->真空速tas"""
        batch_index = self.proc_index(batch_index)
        self._tas[batch_index, :] = self._vel_w[batch_index, 0:1]

    def _ppgt_tas2Vw(self, batch_index: _SupportedIndexType | None):
        """真空速->风轴系惯性速度"""
        batch_index = self.proc_index(batch_index)
        self._vel_w[batch_index, 0:1] = self._tas[batch_index, :]
        self._vel_w[batch_index, 1:3] = 0.0

    def _ppgt_Vw2Vb(self, batch_index: _SupportedIndexType | None):
        """真空速->体轴系惯性速度"""
        batch_index = self.proc_index(batch_index)
        self._vel_b[batch_index, :] = quat_rotate_inv(
            self._Q_wb[batch_index, :], self._vel_w[batch_index, :]
        )

    def _ppgt_Vw2Ve(self, batch_index: _SupportedIndexType | None):
        """真空速->地轴系惯性速度"""
        batch_index = self.proc_index(batch_index)
        self._vel_e[batch_index, :] = quat_rotate(
            self._Q_ew[batch_index, :], self._vel_w[batch_index, :]
        )

    def _ppgt_Vb2rpy_wb(self, batch_index: _SupportedIndexType | None):
        """体轴系惯性速度-> 体轴/风轴 欧拉角(迎角,侧滑角)"""
        batch_index = self.proc_index(batch_index)
        alpha, beta = uvw2alpha_beta(self._vel_b[batch_index, :])
        self.set_alpha(alpha, batch_index)
        self.set_beta(beta, batch_index)

    def _ppgt_QebQwb2Qew(self, batch_index: _SupportedIndexType | None):
        """地轴/体轴 & 风轴/体轴 -> 地轴/风轴 四元数"""
        batch_index = self.proc_index(batch_index)
        self._Q_ew[batch_index, :] = quat_mul(
            self._Q_eb[batch_index, :], quat_conj(self._Q_wb[batch_index, :])
        )

    def _ppgt_QewQwb_to_Qeb(self, batch_index: _SupportedIndexType | None):
        """地轴/风轴 & 风轴/体轴 -> 地轴/体轴 四元数"""
        batch_index = self.proc_index(batch_index)
        self._Q_eb[batch_index, :] = quat_mul(
            self._Q_ew[batch_index, :], self._Q_wb[batch_index, :]
        )

    def roll(self, batch_index: _SupportedIndexType | None):
        """体轴滚转角 in (-pi,pi], unit: rad, shape: (...,N,1)"""
        batch_index = self.proc_index(batch_index)
        return self._rpy_eb[batch_index, 0:1]

    def pitch(self, batch_index: _SupportedIndexType | None):
        """体轴俯仰角 in [-pi/2,pi/2], unit: rad, shape: (...,N,1)"""
        batch_index = self.proc_index(batch_index)
        return self._rpy_eb[batch_index, 1:2]

    def yaw(self, batch_index: _SupportedIndexType | None):
        """体轴偏航角 in (-pi,pi], unit: rad, shape: (...,N,1)"""
        batch_index = self.proc_index(batch_index)
        return self._rpy_eb[batch_index, 2:3]

    def alpha(self, batch_index: _SupportedIndexType | None):
        """迎角 in [-pi/2,pi/2], unit: rad, shape: (...,N,1)"""
        batch_index = self.proc_index(batch_index)
        return self._rpy_wb[batch_index, 1:2]

    def beta(self, batch_index: _SupportedIndexType | None):
        """侧滑角 in [-pi,pi], unit: rad, shape: (...,N,1)"""
        batch_index = self.proc_index(batch_index)
        return -(self._rpy_wb[batch_index, 2:3])

    def mu(self):
        """速度系滚转角 in (-pi,pi], unit: rad, shape: (...,N,1)"""
        return self._rpy_ew[..., 0:1]

    def gamma(self):
        """速度系俯仰角 in [-pi/2,pi/2], unit: rad, shape: (...,N,1)"""
        return self._rpy_ew[..., 1:2]

    def chi(self):
        """速度系偏航角 in (-pi,pi], shape: (...,N,1)"""
        return self._rpy_ew[..., 2:3]

    def set_roll(self, value: torch.Tensor, dst_index: _SupportedIndexType | None):
        """赋值 体轴滚转角(无级联操作) 等价于 roll[dst_index,:]=value"""
        dst_index = self.proc_index(dst_index)
        self._rpy_eb[dst_index, 0:1] = value

    def set_pitch(self, value: torch.Tensor, dst_index: _SupportedIndexType | None):
        """赋值 体轴俯仰角(无级联操作)  等价于 pitch[dst_index,:]=value"""
        dst_index = self.proc_index(dst_index)
        self._rpy_eb[dst_index, 1:2] = value

    def set_yaw(self, value: torch.Tensor, dst_index: _SupportedIndexType | None):
        """赋值 体轴偏航角(无级联操作) 等价于 yaw[dst_index,:]=value"""
        dst_index = self.proc_index(dst_index)
        self._rpy_eb[dst_index, 2:3] = value

    def set_alpha(self, value: torch.Tensor, dst_index: _SupportedIndexType | None):
        """赋值 迎角(无级联操作) 等价于 alpha[dst_index,:]=value"""
        dst_index = self.proc_index(dst_index)
        self._rpy_wb[dst_index, 1:2] = value

    def set_beta(self, value: torch.Tensor, dst_index: _SupportedIndexType | None):
        """赋值 侧滑角(无级联操作) 等价于 beta[dst_index,:]=value"""
        dst_index = self.proc_index(dst_index)
        self._rpy_wb[dst_index, 2:3] = -value

    def set_mu(self, value: torch.Tensor, dst_index: _SupportedIndexType | None):
        """赋值 航迹滚转角(无级联操作) 等价于 mu[dst_index,:]=value"""
        dst_index = self.proc_index(dst_index)
        self._rpy_ew[dst_index, 0:1] = value

    def set_gamma(self, value: torch.Tensor, dst_index: _SupportedIndexType | None):
        """赋值 航迹俯仰角(无级联操作) 等价于 gamma[dst_index,:]=value"""
        dst_index = self.proc_index(dst_index)
        self._rpy_ew[dst_index, 1:2] = value

    def set_chi(self, value: torch.Tensor, dst_index: _SupportedIndexType | None):
        """赋值 航迹偏航角(无级联操作) 等价于 chi[dst_index,:]=value"""
        dst_index = self.proc_index(dst_index)
        self._rpy_ew[dst_index, 2:3] = value
