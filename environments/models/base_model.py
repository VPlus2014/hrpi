import logging
import torch
from abc import ABC, abstractmethod
from typing import Literal
from collections.abc import Sequence

LOGR = logging.getLogger(__name__)


class BaseModel(ABC):
    STATUS_INACTIVATE = -1

    def __init__(
        self,
        model_name: str,
        position_g: torch.Tensor,
        sim_step_size_ms: int = 1,
        model_color: Literal["Red", "Blue"] | str = "Red",
        model_type: Literal["Aircraft", "Missile"] | str = "Aircraft",
        device=torch.device("cpu"),
        dtype=torch.float32,
        **kwargs,
    ) -> None:
        """模型基类 BaseModel

        Args:
            model_name (str): Tacview call sign
            position_g (torch.Tensor): NED地轴系下的初始位置, 单位:m, shape: (n, 3)
            sim_step_size_ms (int, optional): 仿真步长, 单位:ms. Defaults to 1.
            model_color (Literal["Red", "Blue"] | str, optional): Tacview Color. Defaults to "Red".
            model_type (Literal["Aircraft", "Missile"], optional): Tacview Type. Defaults to "Aircraft".
            device (torch.device, optional): 所在torch设备. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): torch浮点类型. Defaults to torch.float32.
        """

        super().__init__()
        self.model_name = model_name
        self.model_color = model_color
        self.sim_step_size_ms = sim_step_size_ms
        self.model_type = model_type
        device = torch.device(device)

        # simulation variables
        assert len(position_g.shape) == 2 and position_g.shape[-1] == 3
        self._init_position_g = position_g.to(
            device=device, dtype=dtype
        )  # model initial position in NED local frame, unit: m, shape: [nenvs, 3]
        self._position_g = (
            self._init_position_g.clone()
        )  # model position in NED local frame, unit: m, shape: [nenvs, 3]

        # simulation paramters
        self._g = 9.8  # acceleration of gravity, unit: m/s^2
        self._rho = 1.29  # atmosphere density, unit: kp/m^3

        self._status = BaseModel.STATUS_INACTIVATE * torch.ones(
            size=(self.batchsize, 1), dtype=torch.int64, device=device
        )  # shape: (B,1)
        self._sim_time_ms = torch.zeros(
            (self.batchsize, 1), dtype=torch.int64, device=device
        ) # (B,1)
        if len(kwargs):
            LOGR.info(
                f"{self.__class__.__name__}.__init__() received {len(kwargs)} unkown keyword arguments, which are ignored."
            )

    @property
    def batchsize(self) -> int:
        """批容量"""
        return self._init_position_g.shape[0]

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
    def position_g(self) -> torch.Tensor:
        """
        model position in NED coordinate system, unit: m, shape: (n, 3)
        """
        return self._position_g

    @position_g.setter
    def position_g(self, value: torch.Tensor):
        self._position_g.copy_(value)

    @property
    def altitude_m(self) -> torch.Tensor:
        """海拔高度, unit: m"""
        return -1 * self.position_g[..., -1:]

    @property
    def device(self) -> torch.device:
        """torch device"""
        return self.position_g.device

    @property
    def dtype(self) -> torch.dtype:
        """torch dtype"""
        return self.position_g.dtype

    @property
    @abstractmethod
    def velocity_g(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """model velocity in NED local frame, unit: m/s"""
        ...

    @property
    def sim_time_ms(
        self, env_indices: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """model simulation time, unit: ms, shape: (..., 1)"""
        env_indices = self.proc_indices(env_indices)
        return self._sim_time_ms[env_indices]

    def reset(self, env_indices: Sequence[int] | torch.Tensor | None = None):
        env_indices = self.proc_indices(env_indices)

        self._position_g[env_indices] = self._init_position_g[env_indices]
        self._status[env_indices] = BaseModel.STATUS_INACTIVATE
        self._sim_time_ms[env_indices] = 0.0

    def run(self):
        self._sim_time_ms += self.sim_step_size_ms

    def is_alive(self) -> torch.Tensor:
        """判断是否存活"""
        raise NotImplementedError
