import torch
from abc import abstractmethod
from typing import Literal, TYPE_CHECKING
from collections.abc import Sequence
from copy import deepcopy

from ..base_model import BaseModel
from ...utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    rpy2quat,
    rpy2quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from ..missile import BaseMissile


class BaseAircraft(BaseModel):
    STATUS_ALIVE = 0
    STATUS_CRASH = 1
    STATUS_SHOTDOWN = 2

    def __init__(
        self,
        tas: torch.Tensor,
        carried_missiles: BaseMissile | None = None,
        model_type="Aircraft",
        **kwargs,
    ) -> None:
        """飞机基类 BaseAircraft

        Args:
            tas (torch.Tensor): 初始真空速, 单位: m/s, shape: (num_models, 1)
            carried_missiles (BaseMissile | None, optional): 导弹挂载. Defaults to None.
            model_type (str, optional): Tacview model type. Defaults to "Aircraft".
            **kwargs: 其他参数, 参见 BaseModel.__init__
        """
        super().__init__(model_type=model_type, **kwargs)
        device = self.device
        dtype = self.dtype
        nenvs = self.batchsize

        # 初始条件
        self.health_point = (
            torch.zeros((self.batchsize, 1), device=device, dtype=dtype) + 100.0
        )  # health point, shape: (B,1)
        self._init_tas = tas.to(device=device, dtype=dtype).view(nenvs, 1)
        self._init_rpy_eb = torch.zeros((nenvs, 3), device=device, dtype=dtype)
        self._init_rpy_ba = torch.zeros((nenvs, 3), device=device, dtype=dtype)
        #
        # simulation variables
        # 本体飞控状态
        self._tas = (
            self._init_tas.clone()
        )  # true air speed 真空速, unit: m/s, shape: (B,1)
        self._vel_g = torch.zeros(
            (nenvs, 3), device=device, dtype=dtype
        )  # 地轴系速度坐标 shape: (B,3)
        self._vel_b = torch.zeros(
            (nenvs, 3), device=device, dtype=dtype
        )  # 体轴系速度坐标 (U,V,W) shape: (B,3)
        self._omega_b = torch.zeros(
            (nenvs, 3), device=device, dtype=dtype
        )  # 体轴系下的旋转角速度 (P,Q,R) shape: (B,3)
        self._Q_gb = torch.zeros(
            (nenvs, 3), device=device, dtype=dtype
        )  # 地轴/体轴 四元数 shape: (B,4)
        self._Q_ba = torch.zeros(
            (nenvs, 4), device=device, dtype=dtype
        )  # 体轴/风轴 四元数 shape: (B,4)
        self._rpy_gb = torch.zeros(
            (nenvs, 3), device=device, dtype=dtype
        )  # 地轴/体轴 欧拉角 (roll, pitch, yaw) shape:(B,3)
        self._rpy_ga = torch.zeros(
            (nenvs, 3), device=device, dtype=dtype
        )  # 体轴/风轴 欧拉角 (mu, gamma, chi) shape:(B,3)
        self._rpy_ba = torch.zeros(
            (nenvs, 3), device=device, dtype=dtype
        )  # 体轴/风轴 欧拉角 (0, alpha, beta) shape:(B,3)

        self.carried_missiles = carried_missiles

    @property
    def Q_gb(self) -> torch.Tensor:
        """地轴系/体轴系四元数"""
        return self._Q_gb

    @property
    def Q_ba(self) -> torch.Tensor:
        """体轴系/风轴系四元数"""
        return self._Q_ba

    @property
    def Q_ga(self) -> torch.Tensor:
        """地轴系/风轴系四元数"""
        return quat_mul(self._Q_gb, self._Q_ba)

    @property
    def tas(self) -> torch.Tensor:
        """true air speed, unit: m/s, shape: (B, 1)"""
        return self._tas

    @tas.setter
    def tas(self, value: torch.Tensor):
        self._tas.copy_(value)

    @property
    def velocity_b(self) -> torch.Tensor:
        """惯性速度的NED体轴系分量(U,V,W), unit: m/s, shape: (B, 3)"""
        return self._vel_b

    @property
    def velocity_g(self) -> torch.Tensor:
        """惯性速度的NED地轴系分量(Vn, Ve, Vd), unit: m/s, shape: (B, 3)"""
        return self._vel_g

    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)

        super().reset(env_indices)
        self.health_point[env_indices] = 100.0
        self._tas[env_indices] = self._init_tas[env_indices]
        self._rpy_gb[env_indices] = self._init_rpy_eb[env_indices]
        self._rpy_ba[env_indices] = self._init_rpy_ba[env_indices]

        self._ppgt_rpy2Qgb()
        self._ppgt_rpy2Qba()

    def run(self, action: torch.Tensor):
        super().run()

    def activate(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)
        self._status[env_indices] = BaseAircraft.STATUS_ALIVE

    # def launch_missile(self, target_aircraft: "BaseAircraft") -> None:
    #     if self.missiles_num > 0:
    #         missile = self.carried_missiles.pop()
    #         missile.launch(carrier_aircraft=self, target_aircraft=target_aircraft)
    #         self.launched_missiles.append(missile)

    def is_alive(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_indices = self.proc_indices(env_indices)
        return self._status[env_indices] == BaseAircraft.STATUS_ALIVE

    def crash(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)
        self._status[env_indices] = BaseAircraft.STATUS_CRASH

    def is_crash(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        env_indices = self.proc_indices(env_indices)
        return self._status[env_indices] == BaseAircraft.STATUS_CRASH

    def is_shotdown(
        self, env_indices: Sequence[int] | torch.Tensor | None = None, update=True
    ) -> torch.Tensor:

        if update:
            # update status
            flag = self.health_point <= 1e-6
            if torch.any(flag):
                indices = torch.where(flag)[0]

                self._status[indices] = BaseAircraft.STATUS_SHOTDOWN

        env_indices = self.proc_indices(env_indices)
        return self._status[env_indices] == BaseAircraft.STATUS_SHOTDOWN

    # propagation modules

    def _ppgt_rpy2Qgb(self):
        """体轴系姿态->四元数"""
        self._Q_gb.copy_(rpy2quat(self._rpy_gb))

    def _ppgt_rpy2Qba(self):
        """风轴系姿态->四元数"""
        self._Q_ba.copy_(rpy2quat(self._rpy_ba))

    def _ppgt_Qgb2rpy(self):
        """四元数->体轴系姿态"""
        self._rpy_gb.copy_(rpy2quat_inv(self._Q_gb, self._rpy_gb[..., 0:1]))

    def _ppgt_Qba2rpy(self):
        """四元数->风轴系姿态"""
        self._rpy_ba.copy_(rpy2quat_inv(self._Q_ba))  # 风轴到体轴不定义滚转(恒为0)

    def _ppgt_vb2vg(self):
        """体轴系v->地轴系v"""
        self._vel_g.copy_(quat_rotate(self._Q_gb, self._vel_b))

    def _ppgt_vg2vb(self):
        """地轴系v->体轴系v"""
        self._vel_b.copy_(quat_rotate_inverse(self._Q_gb, self._vel_g))

    def _ppgt_vb2tas(self):
        """机体系vb->真空速tas"""
        torch.norm(self._vel_b, p=2, dim=-1, keepdim=True, out=self._tas)

    def _ppgt_vg2tas(self):
        """地惯系vg->真空速tas"""
        torch.norm(self._vel_g, p=2, dim=-1, keepdim=True, out=self._tas)

    @property
    def roll(self):
        """体轴滚转角, unit: rad, shape: (B, 1)"""
        return self._rpy_gb[..., 0:1]
    
    @property
    def pitch(self):
        """体轴俯仰角, unit: rad, shape: (B, 1)"""
        return self._rpy_gb[..., 1:2]
    
    @property
    def yaw(self):
        """体轴偏航角, unit: rad, shape: (B, 1)"""
        return self._rpy_gb[..., 2:3]
    
    @property
    def alpha(self):
        """迎角, unit: rad, shape: (B, 1)"""
        return self._rpy_ba[..., 1:2]
    
    @property
    def beta(self):
        """横滚角, unit: rad, shape: (B, 1)"""
        return self._rpy_ba[..., 2:3]

