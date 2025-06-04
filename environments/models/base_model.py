from __future__ import annotations
import hashlib
from types import EllipsisType
from typing import TYPE_CHECKING
import logging
import torch
from abc import ABC, abstractmethod
from typing import Literal, Sequence, Union

from ..utils.math import (
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
    STATUS_DEAD = 1  # 结束

    def __init__(
        self,
        id: torch.Tensor | int,
        position_e: torch.Tensor,
        alt0: torch.Tensor | float,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sim_step_size_ms: int = 1,
        call_sign: str = "",
        acmi_name: str = "",
        acmi_color: Literal["Red", "Blue"] | str = "Red",
        acmi_type: str = "",
    ) -> None:
        """模型基类 BaseModel

        Args:
            id (torch.Tensor | int): Tacview Object ID (整数形式).
            position_e (torch.Tensor): NED地轴系下的初始位置, 单位:m, shape: (n, 3)
            alt0 (torch.Tensor | float, optional): 坐标原点高度, 单位:m.
            sim_step_size_ms (int, optional): 仿真步长, 单位:ms. Defaults to 1.
            device (torch.device, optional): 所在torch设备. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): torch浮点类型. Defaults to torch.float32.
            acmi_color (Literal["Red", "Blue"] | str, optional): Tacview Color. Defaults to "Red".
            call_sign (str, optional): Tacview 呼号. Defaults to "".
            acmi_name (str, optional): Tacview 模型名(必须数据库中可检索否则无法正常渲染)
            acmi_type (str, optional): Tacview Object Type(符合ACMI标准). Defaults to "".
        """

        super().__init__()
        self.acmi_color = acmi_color
        self.acmi_name = acmi_name
        self.acmi_type = acmi_type
        self.call_sign = call_sign

        self._sim_step_size_ms = sim_step_size_ms
        self._device = device = torch.device(device)
        self._dtype = dtype

        # simulation paramters
        self._rho = 1.29  # atmosphere density, unit: kp/m^3
        self._g = 9.8  # acceleration of gravity, unit: m/s^2

        # simulation variables
        # initial conditions
        assert len(position_e.shape) == 2 and position_e.shape[-1] == 3
        self._ic_pos_e = position_e.to(device=device, dtype=dtype).clone()
        """model initial position in NED local frame, unit: m, shape: (B, 3)"""
        _shape = [self.batchsize]
        _0f1 = torch.zeros(_shape + [1], device=device, dtype=dtype)

        self.id = torch.empty(_shape + [1], device=device, dtype=torch.int64)
        """Tacview Object ID, shape: (B,1)"""
        self.id.copy_(torch.asarray(id, device=device, dtype=torch.int64))

        # cache variables
        self._pos_e = torch.empty(_shape + [3], device=device, dtype=dtype)
        """position in NED local frame, unit: m, shape: (B, 3)"""
        self._vel_e = torch.empty(_shape + [3], device=device, dtype=dtype)
        """velocity in NED local frame, unit: m/s, shape: (B, 3)"""
        self._alt0 = _0f1 + alt0
        """altitude, unit: m, shape: (B, 1)"""

        self._g_e = self._g * torch.cat(
            [
                torch.zeros((*_shape, 2), device=device, dtype=dtype),
                torch.ones(_shape + [1], device=device, dtype=dtype),
            ],
            dim=-1,
        )
        """重力加速度NED地轴系坐标"""

        self.status = BaseModel.STATUS_INACTIVATE + torch.zeros(
            size=_shape + [1], dtype=torch.int64, device=device
        )  # shape: (B,1)
        self._sim_time_ms = torch.zeros(
            _shape + [1], dtype=torch.int64, device=device
        )  # (B,1)
        # if len(kwargs):
        #     msg = (
        #         self.__class__.__name__,
        #         f"received {len(kwargs)} unkown keyword arguments",
        #         list(kwargs.keys()),
        #     )
        #     LOGR.warning(msg)

    @property
    def batchsize(self) -> int:
        """批容量"""
        return self._ic_pos_e.shape[0]

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
            idxs = Ellipsis
        elif isinstance(batch_index, (slice, torch.Tensor)):
            idxs = batch_index
        else:
            idxs = torch.asarray(batch_index, device=self.device, dtype=torch.int64)
        # 不做检查
        return idxs

    def position_e(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """position in NED local frame, unit: m, shape: (B, 3)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._pos_e[batch_index, :]

    def velocity_e(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """velocity in NED local frame, unit: m/s, shape: (B, 3)"""
        return self._vel_e[batch_index, :]

    def altitude_m(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """altitude, unit: m, shape: (B, 1)"""
        batch_index = self.proc_batch_index(batch_index)
        return self._alt0[batch_index, :] - self._pos_e[batch_index, 2:3]

    def longitude_deg(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """longitude, unit: rad, shape: (B, 1)"""
        raise NotImplementedError

    def latitude_deg(self, batch_index: _SupportedIndexType = None) -> torch.Tensor:
        """latitude, unit: rad, shape: (B, 1)"""
        raise NotImplementedError

    @property
    def g_e(self) -> torch.Tensor:
        """NED地轴系重力加速度向量"""
        return self._g_e

    @property
    def sim_time_ms(self) -> torch.Tensor:
        """model simulation time, unit: ms, shape: (B, 1)"""
        return self._sim_time_ms

    @property
    def sim_time_s(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """model simulation time, unit: s, shape: (B, 1)"""
        return self.sim_time_ms * 1e-3

    def reset(self, env_indices: _SupportedIndexType = None):
        """重置 位置, 仿真时间, 仿真生命状态"""
        env_indices = self.proc_batch_index(env_indices)

        self.status[env_indices] = BaseModel.STATUS_INACTIVATE
        self._sim_time_ms[env_indices] = 0.0

        self._pos_e[env_indices] = self._ic_pos_e[env_indices]

    def run(self):
        self._sim_time_ms += self._sim_step_size_ms

    def is_alive(self, env_indices: _SupportedIndexType = None) -> torch.Tensor:
        """判断是否存活"""
        env_indices = self.proc_batch_index(env_indices)
        return self.status[env_indices] == self.__class__.STATUS_ALIVE

    @property
    def sim_step_size_ms(self) -> int:
        """仿真步长, 单位: ms"""
        return self._sim_step_size_ms

    @property
    def sim_step_size_s(self) -> float:
        """仿真步长, 单位: s"""
        return self._sim_step_size_ms * 1e-3


class BaseFV(BaseModel):

    def __init__(
        self,
        use_eb=True,
        use_ew=True,
        use_wb=True,
        **kwargs,
    ) -> None:
        """飞行器基类 BaseFV

        Args:
            use_eb (bool, optional): 是否启用地轴-体轴系状态. Defaults to True.
            use_ew (bool, optional): 是否启用地轴-风轴系状态. Defaults to True.
            use_wb (bool, optional): 是否启用风轴-体轴系状态. Defaults to True.
            **kwargs: 其他参数, 参见 BaseModel.__init__
        """
        super().__init__(**kwargs)
        device = self.device
        dtype = self.dtype
        _shape = [self.batchsize]

        #
        self.health_point = (
            torch.zeros(_shape + [1], device=device, dtype=dtype) + 100.0
        )  # health point, shape: (B,1)
        #
        # simulation variables
        # 本体飞控状态
        self._tas = torch.zeros(
            _shape + [1], device=device, dtype=dtype
        )  # true air speed 真空速, unit: m/s, shape: (B,1)
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

        if use_wb:
            self._rpy_wb = torch.zeros(
                _shape + [3], device=device, dtype=dtype
            )  # 风轴/体轴 欧拉角 (0, alpha, -beta) shape:(B,3)
            self._Q_wb = torch.zeros(
                _shape + [4], device=device, dtype=dtype
            )  # 风轴/体轴 四元数 shape: (B,4)

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
        self.status[batch_index] = BaseFV.STATUS_ALIVE

    # propagation modules

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
        self._vel_w[batch_index, 1:3] = 0

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
