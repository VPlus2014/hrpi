from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import torch
from abc import ABC, abstractmethod
from typing import Literal
from collections.abc import Sequence
from ..utils.math import (
    quat_mul,
    quat_rotate,
    quat_rotate_inv,
    rpy2quat_inv,
    rpy2quat,
    xyz2aer,
)

LOGR = logging.getLogger(__name__)


class BaseModel(ABC):
    STATUS_INACTIVATE = -1  # 未启动
    STATUS_ALIVE = 0  # 运行中
    STATUS_DEAD = 1  # 结束

    def __init__(
        self,
        position_e: torch.Tensor,
        model_name: str,
        model_color: Literal["Red", "Blue"] | str = "Red",
        model_type: Literal["Aircraft", "Missile"] | str = "Aircraft",
        device=torch.device("cpu"),
        dtype=torch.float32,
        sim_step_size_ms: int = 1,
        **kwargs,
    ) -> None:
        """模型基类 BaseModel

        Args:
            position_e (torch.Tensor): NED地轴系下的初始位置, 单位:m, shape: (n, 3)
            model_name (str, optional): 模型名称. Defaults to "BaseModel".
            model_color (Literal["Red", "Blue"] | str, optional): Tacview Color. Defaults to "Red".
            model_type (Literal["Aircraft", "Missile"], optional): Tacview Type. Defaults to "Aircraft".
            device (torch.device, optional): 所在torch设备. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): torch浮点类型. Defaults to torch.float32.
            sim_step_size_ms (int, optional): 仿真步长, 单位:ms. Defaults to 1.
        """

        super().__init__()
        self.model_name = model_name
        self.model_color = model_color
        self._sim_step_size_ms = sim_step_size_ms
        self.model_type = model_type
        self._device = device = torch.device(device)
        self._dtype = dtype

        # simulation paramters
        self._rho = 1.29  # atmosphere density, unit: kp/m^3
        self._g = 9.8  # acceleration of gravity, unit: m/s^2

        # simulation variables
        # initial conditions
        assert len(position_e.shape) == 2 and position_e.shape[-1] == 3
        self._ic_pos_e = position_e.to(
            device=device, dtype=dtype
        )  # model initial position in NED local frame, unit: m, shape: (B, 3)
        batchsize = self.batchsize

        # cache variables
        self._pos_e = torch.empty(
            (batchsize, 3), device=device, dtype=dtype
        )  # position in NED local frame, unit: m, shape: (B, 3)
        self._vel_e = torch.empty(
            (batchsize, 3), device=device, dtype=dtype
        )  # velocity in NED local frame, unit: m/s, shape: (B, 3)

        self._g_e = self._g * torch.cat(
            [
                torch.zeros((batchsize, 2), device=device, dtype=dtype),
                torch.ones((batchsize, 1), device=device, dtype=dtype),
            ],
            dim=-1,
        )  # NED 重力加速度向量

        self.status = BaseModel.STATUS_INACTIVATE * torch.ones(
            size=(self.batchsize, 1), dtype=torch.int64, device=device
        )  # shape: (B,1)
        self._sim_time_ms = torch.zeros(
            (self.batchsize, 1), dtype=torch.int64, device=device
        )  # (B,1)
        if len(kwargs):
            LOGR.info(
                f"{self.__class__.__name__}.__init__() received {len(kwargs)} unkown keyword arguments, which are ignored."
            )

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

    def proc_indices(
        self, indices: Sequence[int] | torch.Tensor | None = None, check=False
    ):
        """对索引做预处理"""
        if indices is None:
            indices = torch.arange(self.batchsize, device=self.device)
        elif isinstance(indices, torch.Tensor):
            pass
        else:
            indices = torch.asarray(indices, device=self.device, dtype=torch.int64)
        # assert isinstance(env_indices, torch.Tensor)
        if check:
            imax = indices.max().item()
            assert (
                imax < self.batchsize
            ), f"env_indices {indices} out of range [0, {self.batchsize})"
        return indices

    @property
    def position_e(self) -> torch.Tensor:
        """
        position in NED local frame, unit: m, shape: (n, 3)
        """
        return self._pos_e

    @property
    def altitude_m(self) -> torch.Tensor:
        """海拔高度, unit: m"""
        return -1 * self.position_e[..., -1:]

    @property
    @abstractmethod
    def velocity_e(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        velocity in NED local frame, unit: m/s"""
        return self._vel_e

    @property
    def g_e(self) -> torch.Tensor:
        """NED地轴系重力加速度向量"""
        return self._g_e

    @property
    def sim_time_ms(self) -> torch.Tensor:
        """model simulation time, unit: ms, shape: (..., 1)"""
        return self._sim_time_ms

    @property
    def sim_time_s(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """model simulation time, unit: s, shape: (..., 1)"""
        return self.sim_time_ms * 1e-3

    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        """重置 位置, 仿真时间, 仿真生命状态"""
        env_indices = self.proc_indices(env_indices)

        self.status[env_indices] = BaseModel.STATUS_INACTIVATE
        self._sim_time_ms[env_indices] = 0.0

        self._pos_e[env_indices] = self._ic_pos_e[env_indices]

    def run(self):
        self._sim_time_ms += self._sim_step_size_ms

    def is_alive(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """判断是否存活"""
        env_indices = self.proc_indices(env_indices)
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
        model_type="Aircraft",
        use_body_frame=False,
        use_wind_frame=True,
        **kwargs,
    ) -> None:
        """飞行器基类 BaseFV

        Args:
            model_type (str, optional): Tacview model type. Defaults to "Aircraft".
            use_body_frame (bool, optional): 是否启用体轴系状态. Defaults to False.
            use_wind_frame (bool, optional): 是否启用风轴系/速度系状态. Defaults to True.
            **kwargs: 其他参数, 参见 BaseModel.__init__
        """
        super().__init__(model_type=model_type, **kwargs)
        device = self.device
        dtype = self.dtype
        nenvs = self.batchsize

        #
        self.health_point = (
            torch.zeros((self.batchsize, 1), device=device, dtype=dtype) + 100.0
        )  # health point, shape: (B,1)
        #
        # simulation variables
        # 本体飞控状态
        self._tas = torch.zeros(
            (nenvs, 1), device=device, dtype=dtype
        )  # true air speed 真空速, unit: m/s, shape: (B,1)
        if use_body_frame:
            self._vel_b = torch.zeros(
                (nenvs, 3), device=device, dtype=dtype
            )  # 体轴系速度坐标 (U,V,W) shape: (B,3)
            self._Q_eb = torch.zeros(
                (nenvs, 4), device=device, dtype=dtype
            )  # 地轴/体轴 四元数 shape: (B,4)
            self._Q_ba = torch.zeros(
                (nenvs, 4), device=device, dtype=dtype
            )  # 体轴/风轴 四元数 shape: (B,4)
            self._rpy_eb = torch.zeros(
                (nenvs, 3), device=device, dtype=dtype
            )  # 地轴/体轴 欧拉角 (roll, pitch, yaw) shape:(B,3)
            self._rpy_ba = torch.zeros(
                (nenvs, 3), device=device, dtype=dtype
            )  # 体轴/风轴 欧拉角 (0, alpha, beta) shape:(B,3)

            self._omega_b = torch.zeros(
                (nenvs, 3), device=device, dtype=dtype
            )  # 体轴系下的旋转角速度 (P,Q,R) shape: (B,3)

        if use_wind_frame:
            self._Q_ea = torch.zeros(
                (nenvs, 4), device=device, dtype=dtype
            )  # 地轴/风轴 四元数 shape: (B,4)
            self._rpy_ea = torch.zeros(
                (nenvs, 3), device=device, dtype=dtype
            )  # 地轴/风轴 欧拉角 (mu, gamma, chi) shape:(B,3)
            self._vel_a = torch.zeros(
                (nenvs, 3), device=device, dtype=dtype
            )  # 风轴系速度坐标 (Vn,Ve,Vd) shape: (B,3)

    @property
    def Q_eb(self) -> torch.Tensor:
        """地轴系/体轴系四元数"""
        return self._Q_eb

    @property
    def Q_ba(self) -> torch.Tensor:
        """体轴系/风轴系四元数"""
        return self._Q_ba

    @property
    def Q_ea(self) -> torch.Tensor:
        """地轴系/风轴系四元数"""
        return self._Q_ea

    @property
    def tas(self) -> torch.Tensor:
        """true air speed, unit: m/s, shape: (B, 1)"""
        return self._tas

    @property
    def velocity_b(self) -> torch.Tensor:
        """惯性速度的NED体轴系分量(U,V,W), unit: m/s, shape: (B, 3)"""
        return self._vel_b

    @property
    def velocity_e(self) -> torch.Tensor:
        """惯性速度的NED地轴系分量(Vn, Ve, Vd), unit: m/s, shape: (B, 3)"""
        return self._vel_e

    @property
    def velocity_a(self) -> torch.Tensor:
        """惯性速度的NED航迹系分量(Vn, Ve, Vd), unit: m/s, shape: (B, 3)"""
        return self._vel_a

    def activate(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)
        self.status[env_indices] = BaseFV.STATUS_ALIVE

    # propagation modules

    def _ppgt_rpy_eb2Qeb(self):
        """地轴/体轴 欧拉角->四元数"""
        self._Q_eb.copy_(rpy2quat(self._rpy_eb))

    def _ppgt_rpy_ba2Qba(self):
        """体轴/风轴 欧拉角->四元数"""
        self._Q_ba.copy_(rpy2quat(self._rpy_ba))

    def _ppgt_Qeb2rpy_eb(self):
        """地轴/体轴 四元数->欧拉角"""
        self._rpy_eb.copy_(rpy2quat_inv(self._Q_eb, self._rpy_eb[..., 0:1]))

    def _ppgt_Qba2rpy_ba(self):
        """体轴/风轴 四元数->欧拉角"""
        self._rpy_ba.copy_(rpy2quat_inv(self._Q_ba))  # 风轴到体轴不定义滚转(恒为0)

    def _ppgt_Qea2rpy_ea(self):
        """地轴/风轴 四元数->欧拉角"""
        self._rpy_ea.copy_(rpy2quat_inv(self._Q_ea, self._rpy_ea[..., 0:1]))

    def _ppgt_vb2ve(self):
        """惯性速度 体轴系->地轴系"""
        self._vel_e.copy_(quat_rotate(self._Q_eb, self._vel_b))

    def _ppgt_ve2vb(self):
        """惯性速度 地轴系->体轴系"""
        self._vel_b.copy_(quat_rotate_inv(self._Q_eb, self._vel_e))

    def _ppgt_ve2va(self):
        """惯性速度 地轴系->航迹系"""
        self._vel_a.copy_(quat_rotate_inv(self._Q_ea, self._vel_e))

    def _ppgt_va2ve(self):
        """惯性速度 航迹系->地轴系"""
        self._vel_e.copy_(quat_rotate(self._Q_ea, self._vel_a))

    def _ppgt_vb2tas(self):
        """体轴系惯性速度->真空速tas"""
        torch.norm(self._vel_b, p=2, dim=-1, keepdim=True, out=self._tas)

    def _ppgt_vb2rpy_ba(self):
        """体轴系惯性速度-> 体轴/风轴 欧拉角(迎角,侧滑角)"""
        ypR = xyz2aer(self._vel_b)
        beta, alpha, _ = ypR.split([1, 1, 1], -1)
        self.set_beta(beta)
        self.set_alpha(alpha)

    def _ppgt_vg2tas(self):
        """地轴系惯性速度->真空速tas"""
        torch.norm(self._vel_e, p=2, dim=-1, keepdim=True, out=self._tas)

    def _ppgt_Qea(self):
        """重算 地轴/风轴 四元数"""
        self._Q_ea.copy_(quat_mul(self._Q_eb, self._Q_ba))

    def _ppgt_tas2va(self):
        """将真空速转换为体轴系坐标"""
        self._vel_a[..., 0:1] = self._tas
        # self._vel_a[..., 1:3] = 0

    def _ppgt_va2tas(self):
        """速度系惯性速度->真空速"""
        self._tas.copy_(self._vel_a[..., 0:1])

    @property
    def roll(self):
        """体轴滚转角, unit: rad, shape: (B, 1)"""
        return self._rpy_eb[..., 0:1]

    @property
    def pitch(self):
        """体轴俯仰角, unit: rad, shape: (B, 1)"""
        return self._rpy_eb[..., 1:2]

    @property
    def yaw(self):
        """体轴偏航角, unit: rad, shape: (B, 1)"""
        return self._rpy_eb[..., 2:3]

    @property
    def alpha(self):
        """迎角, unit: rad, shape: (B, 1)"""
        return self._rpy_ba[..., 1:2]

    @property
    def beta(self):
        """侧滑角, unit: rad, shape: (B, 1)"""
        return self._rpy_ba[..., 2:3]

    @property
    def mu(self):
        """速度系滚转角, unit: rad, shape: (B, 1)"""
        return self._rpy_ea[..., 0:1]

    @property
    def gamma(self):
        """速度系俯仰角, unit: rad, shape: (B, 1)"""
        return self._rpy_ea[..., 1:2]

    @property
    def chi(self):
        """速度系偏航角, unit: rad, shape: (B, 1)"""
        return self._rpy_ea[..., 2:3]

    def set_roll(self, roll: torch.Tensor, dst_index=None):
        dst_index = self.proc_indices(dst_index)
        self._rpy_eb[dst_index, 0:1] = roll

    def set_pitch(self, pitch: torch.Tensor, dst_index=None):
        dst_index = self.proc_indices(dst_index)
        self._rpy_eb[dst_index, 1:2] = pitch

    def set_yaw(self, yaw: torch.Tensor, dst_index=None):
        dst_index = self.proc_indices(dst_index)
        self._rpy_eb[dst_index, 2:3] = yaw

    def set_alpha(self, alpha: torch.Tensor, dst_index=None):

        dst_index = self.proc_indices(dst_index)
        self._rpy_ba[dst_index, 1:2] = alpha

    def set_beta(self, beta: torch.Tensor, dst_index=None):
        dst_index = self.proc_indices(dst_index)
        self._rpy_ba[dst_index, 2:3] = beta

    def set_mu(self, mu: torch.Tensor, dst_index=None):

        dst_index = self.proc_indices(dst_index)
        self._rpy_ea[dst_index, 0:1] = mu

    def set_gamma(self, gamma: torch.Tensor, dst_index=None):
        dst_index = self.proc_indices(dst_index)
        self._rpy_ea[dst_index, 1:2] = gamma

    def set_chi(self, chi: torch.Tensor, dst_index=None):
        dst_index = self.proc_indices(dst_index)
        self._rpy_ea[dst_index, 2:3] = chi
