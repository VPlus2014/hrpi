from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
import os
import logging
import numpy as np
import pymap3d
from abc import ABC, abstractmethod
from typing import Literal, Sequence, Union
from ..utils.math_np import (
    quat_conj,
    quat_mul,
    quat_rotate,
    quat_rotate_inv,
    rpy2quat_inv,
    rpy2quat,
    uvw2alpha_beta,
    norm_,
    IntNDArr,
    BoolNDArr,
    FloatNDArr,
    DoubleNDArr,
    bkbn,
)
from numpy.typing import NDArray

LOGR = logging.getLogger(__name__)

SupportedIndexType = Union[Sequence[int], NDArray[np.intp], int, slice, type(Ellipsis)]
SupportedMaskType = Union[NDArray[np.bool_], type(Ellipsis)]
_SliceAll = slice(None)


class BaseModel(ABC):
    STATUS_INACTIVE = -1  # 未启动
    STATUS_ALIVE = 0  # 运行中
    STATUS_DYING = 1  # 即将结束
    STATUS_DEAD = 2  # 结束

    logr = LOGR
    DEBUG: bool = False

    def __init__(
        self,
        group_shape: Sequence[int] | int = 1,
        device="cpu",
        dtype: type[np.float64] = np.float64,  # 科学计算默认
        sim_step_size_ms: int = 1,
        use_gravity: bool = True,
        g: np.ndarray | float = 9.8,  # 默认重力加速度 m/s^2
        use_eb=True,
        use_ew=True,
        use_wb=True,
        use_geodetic: bool = True,
        lat0: np.ndarray | float = 0,
        lon0: np.ndarray | float = 0,
        alt0: np.ndarray | float = 0,
        use_mass=False,
        use_inertia=False,
        vis_radius: float = 1.0,
        acmi_id: np.ndarray | int = 0,
        acmi_name: str | Sequence[str] = "",
        acmi_color: str | Sequence[str] = "Red",
        acmi_type: str | Sequence[str] = "",
        call_sign: str | Sequence[str] = "",
        acmi_parent: np.ndarray | int = 0,
        debug=False,
    ) -> None:
        """
        质点模型组 BaseModel
        约定:
            所有非标量数据都是矩阵;
        Args:
            sim_step_size_ms (int, optional): 仿真步长, 单位:ms. Defaults to 1.
            group_shape (int|Sequence[int], optional): 组容量N/形状(...,N), Defaults to 1.
            device (np.device, optional): 所在torch设备. Defaults to np.device("cpu").
            dtype (np.dtype, optional): torch浮点类型. Defaults to np.float32.
            use_gravity (bool, optional): 是否启用重力(无则不支持计算重力). Defaults to True.
            g (float, optional): 重力加速度, 单位:m/s^2. Defaults to 9.8.
            use_eb (bool, optional): 是否启用地轴-体轴系状态. Defaults to True.
            use_ew (bool, optional): 是否启用地轴-风轴系状态. Defaults to True.
            use_wb (bool, optional): 是否启用风轴-体轴系状态. Defaults to True.
            use_geodetic (bool, optional): 是否使用地理坐标. Defaults to True.
            lat0 (np.ndarray | float, optional): 坐标原点纬度, 单位:deg. float|shape: (...,N,1)
            lon0 (np.ndarray | float, optional): 坐标原点经度, 单位:deg. float|shape: (...,N,1)
            alt0 (np.ndarray | float, optional): 坐标原点高度, 单位:m. float|shape: (...,N,1)
            acmi_id (np.ndarray | int, optional): Tacview Object ID (整数形式).
            acmi_color (Literal["Red", "Blue"] | str, optional): Tacview Color. Defaults to "Red".
            acmi_name (str, Sequence[str], optional): Tacview 模型名(必须数据库中可检索否则无法正常渲染)
            acmi_type (str, Sequence[str], optional): Tacview Object Type(符合ACMI标准). Defaults to "".
            acmi_parent (np.ndarray | int, optional): Tacview 父对象 ID. Defaults to "".
            call_sign (str, Sequence[str], optional): Tacview 呼号. Defaults to "".
            vis_radius (float, optional): 可视半径. Defaults to 1.0.
        """
        super().__init__()
        self.__class__._objcnt += 1
        self._tmp_gid = (os.getpid() << 16) | self.__class__._objcnt
        self.DEBUG = debug
        self._device = _device = device  # = np.device(device)
        self._dtype = dtype
        assert dtype in (
            np.float32,
            np.float64,
        ), "dtype only support float32/float64"
        if isinstance(group_shape, int):
            group_shape = (group_shape,)
        assert len(group_shape) > 0, ("group_shape must be not empty", group_shape)
        self._group_shape = _group_shape = (*group_shape,)
        self._batch_size = int(np.prod(_group_shape))
        _0f1 = np.zeros(
            _group_shape + (1,),
            dtype=dtype,
            # device=_device,
        )  # (...,1)

        self.status = np.zeros(
            _group_shape + (1,),
            dtype=np.int32,
            # device=_device,
        )
        """仿真运行状态, shape: (...,N,1)"""
        self._sim_step_size_ms = sim_step_size_ms
        """仿真步长, unit: ms; int"""
        self._sim_time_ms = np.zeros(
            _group_shape + (1,),
            dtype=np.int64,
            # device=_device,
        )
        """仿真时钟 unit: ms, shape: (...,N,1)"""
        self.health_point = np.zeros(
            _group_shape + (1,),
            # device=_device,
            dtype=dtype,
        )
        """health point, shape: (...,N,1)"""

        self._vis_radius = np.empty(
            _group_shape + (1,),
            # device=_device,
            dtype=dtype,
        )
        """可视半径, shape: (...,N,1)"""
        self._vis_radius[...] = np.asarray(
            vis_radius,
            dtype=dtype,
            # device=_device,
        )

        # simulation variables

        # cache variables
        self._pos_e = np.empty(
            _group_shape + (3,),
            # device=_device,
            dtype=dtype,
        )
        """position in NED local frame, unit: m, shape: (...,N,3)"""
        self._vel_e = np.empty(
            _group_shape + (3,),
            # device=_device,
            dtype=dtype,
        )
        """velocity in NED local frame, unit: m/s, shape: (...,N,3)"""

        # 常用缓存
        self._0F = _0 = _0f1  # (...,N,1)
        """zeros(group_shape+(1,), dtype=dtype), shape: (...,N,1)"""
        self._1F = _1 = np.ones_like(_0)  # (...,N,1)
        """ones(group_shape+(1,), dtype=dtype), shape: (...,N,1)"""
        self._E1F = np.concatenate([_1, _0, _0], axis=-1)
        self._E2F = np.concatenate([_0, _1, _0], axis=-1)
        self._E3F = np.concatenate([_0, _0, _1], axis=-1)
        self._MASK1 = np.ones(
            _group_shape,
            dtype=np.bool_,
            # device=_device,
        )
        """mask for vector data, shape: (...,N,)"""

        self._g = np.empty(
            _group_shape + (1,),
            # device=_device,
            dtype=dtype,
        )
        self._g[...] = np.asarray(
            g,
            dtype=dtype,
            # device=_device,
        )
        self.use_gravity = use_gravity
        """是否启用NED地轴系重力加速度向量缓存, shape: (...,N,1)"""
        if use_gravity:
            """重力加速度, 单位: m/s^2"""
            self._g_e = np.concatenate([_0, _0, self._g], axis=-1)
            """重力加速度NED地轴系坐标, shape: (...,N,3)"""

        # 本体飞控状态
        #
        self._tas = np.zeros(
            _group_shape + (1,),
            # device=_device,
            dtype=dtype,
        )
        """true air speed 真空速, unit: m/s, shape: (...,N,1)"""

        self._use_eb = use_eb
        if use_eb:
            self._vel_b = np.zeros(
                _group_shape + (3,),
                # device=_device,
                dtype=dtype,
            )
            """惯性速度体轴坐标 (U,V,W) shape: (...,N,3)"""
            self._Q_eb = np.zeros(
                _group_shape + (4,),
                # device=_device,
                dtype=dtype,
            )
            """地轴/体轴 四元数 shape: (...,N,4)"""
            self._rpy_eb = np.zeros(
                _group_shape + (3,),
                # device=_device,
                dtype=dtype,
            )
            """地轴/体轴 欧拉角 (roll, pitch, yaw) shape:(...,N,3)"""
            self._omega_b = np.zeros(
                _group_shape + (3,),
                # device=_device,
                dtype=dtype,
            )
            """体轴系下的旋转角速度 (P,Q,R) shape: (...,N,3)"""
        if use_inertia:
            assert use_eb, "use_inertia must be used with use_eb"
            self._I_b = np.empty(
                _group_shape + (3, 3),
                # device=_device,
                dtype=dtype,
            )
            """体轴惯性矩 shape: (...,N,3,3)"""
            self._I_b_inv = np.empty(
                _group_shape + (3, 3),
                # device=_device,
                dtype=dtype,
            )

        self._use_ew = use_ew
        if use_ew:
            self._rpy_ew = np.zeros(
                _group_shape + (3,),
                # device=_device,
                dtype=dtype,
            )
            """地轴/风轴 欧拉角 (mu, gamma, chi) shape:(...,N,3)"""
            self._Q_ew = np.zeros(
                _group_shape + (4,),
                # device=_device,
                dtype=dtype,
            )
            """地轴/风轴 四元数 shape: (...,N,4)"""
            self._vel_w = np.zeros(
                _group_shape + (3,),
                # device=_device,
                dtype=dtype,
            )
            """惯性速度风轴分量 (TAS,0,0) shape: (...,N,3)"""

        self._use_wb = use_wb
        if use_wb:
            self._rpy_wb = np.zeros(
                _group_shape + (3,),
                # device=_device,
                dtype=dtype,
            )
            """风轴/体轴 欧拉角 (0, alpha, -beta) shape:(...,N,3)"""
            self._Q_wb = np.zeros(
                _group_shape + (4,),
                # device=_device,
                dtype=dtype,
            )
            """风轴/体轴 四元数 shape: (...,N,4)"""

        self._use_geodetic = use_geodetic
        if use_geodetic:
            self._blh0 = np.empty(
                _group_shape + (3,),
                # device=_device,
                dtype=dtype,
            )
            """坐标原点(纬度,经度,高度), 单位:(deg,m), shape: (...,N,3)"""
            self._blh0[..., 0:1] = lat0 + _0f1
            self._blh0[..., 1:2] = lon0 + _0f1
            self._blh0[..., 2:3] = alt0 + _0f1

            self._blh = np.empty(
                _group_shape + (3,),
                # device=_device,
                dtype=dtype,
            )
            """当前 (纬度,经度,高度), 单位:(deg,m), shape: (...,N,3)"""

            # self._lat.copy_(self._lat0)
            # self._lon.copy_(self._lon0)
            # self._alt.copy_(self._alt0)
            # pos_e->(lat, lon, alt)
            # self._ppgt_ned2blh(None)

        self._use_mass = use_mass
        if use_mass:
            self._mass = np.empty(
                _group_shape + (1,),
                # device=_device,
                dtype=dtype,
            )

        # if len(kwargs):
        #     msg = (
        #         self.__class__.__name__,
        #         f"received {len(kwargs)} unkown keyword arguments",
        #         list(kwargs.keys()),
        #     )
        #     LOGR.warning(msg)

        self.acmi_id = np.empty(
            _group_shape + (1,),
            # device=_device,
            dtype=np.int32,
        )
        """Tacview Object ID, shape: (...,N,1) Tensor[int64]"""
        self.acmi_id[...] = np.asarray(
            acmi_id,
            # device=_device,
            dtype=self.acmi_id.dtype,
        )

        self.acmi_color = np.empty(
            _group_shape + (1,),
            dtype=object,
        )
        """Tacview 颜色, shape: (...,N,1) NDArray[str]"""
        self.acmi_color[..., 0] = acmi_color

        self.acmi_name = np.empty(
            _group_shape + (1,),
            dtype=object,
        )
        """Tacview 模型名称, shape: (...,N,1) NDArray[str]"""
        self.acmi_name[..., 0] = acmi_name

        self.acmi_type = np.empty(
            _group_shape + (1,),
            dtype=object,
        )
        """Tacview 对象类型, shape: (...,N,1) NDArray[str]"""
        self.acmi_type[..., 0] = acmi_type

        self.acmi_parent = np.empty(
            _group_shape + (1,),
            dtype=self.acmi_id.dtype,
            # device=_device,
        )
        """Tacview 父对象 ID, shape: (...,N,1) Tensor[int32]"""
        self.acmi_parent[...] = np.asarray(
            acmi_parent,
            dtype=self.acmi_parent.dtype,
            # device=_device,
        )

        self.call_sign = np.empty(
            _group_shape + (1,),
            dtype=object,
        )
        """Tacview 呼号, shape: (...,N,1) NDArray[str]"""
        self.call_sign[..., 0] = call_sign

    _objcnt = 0  # (同一进程共享)

    @property
    def batch_size(self) -> int:
        """组容量(拉平以后)"""
        return self._batch_size

    @property
    def group_shape(self):
        """组形状"""
        return self._group_shape

    @property
    def device(self):
        """torch device"""
        return self._device

    @property
    def dtype(self):
        """torch dtype"""
        return self._dtype

    def proc_index(self, index: SupportedIndexType | None):
        """对索引做预处理"""
        if index is None or index == Ellipsis:
            idxs = slice(None)
        elif isinstance(index, slice):
            idxs = index
        elif isinstance(index, np.ndarray):
            idxs = index
        elif isinstance(index, int):
            idxs = [index]
        elif isinstance(index, Sequence):
            idxs = index
        else:
            idxs = np.asarray(
                index,
                dtype=np.intp,
                # device=self.device,
            )
        # 不做检查
        return idxs

    def proc_to_mask(self, mask: SupportedMaskType | None):
        """
        调整到与 group_shape 同尺寸的 mask
        """
        tgt = self._MASK1  # ref
        if mask is None or mask is Ellipsis or mask is tgt:
            msk = tgt
        elif isinstance(mask, np.ndarray):
            assert mask.dtype == np.bool_, "mask must be bool type"
            if mask.shape == tgt.shape:
                msk = mask
            else:
                if mask.ndim == tgt.ndim:
                    msk = mask
                elif mask.ndim == tgt.ndim + 1:
                    msk = mask.squeeze(-1)
                else:
                    raise ValueError("mask shape mismatch", mask.shape, tgt.shape)

                msk = np.logical_and(
                    tgt, msk
                )  # mask&self._MASK1 做的是字节位运算,输入为int数组时会发生意料之外的结果!
                assert msk.shape == tgt.shape, (
                    "mask shape mismatch",
                    msk.shape,
                    tgt.shape,
                )
            # msk = mask.to(self.device)
        elif isinstance(mask, slice):
            if mask == _SliceAll:
                msk = tgt
            else:
                raise NotImplementedError("slice mask not supported yet")
        else:
            raise TypeError("unsupported mask type", type(mask))
        return msk

    @property
    def sim_step_size_ms(self) -> int:
        """仿真步长, 单位:ms"""
        return self._sim_step_size_ms

    @property
    def sim_step_size_s(self) -> float:
        """仿真步长, 单位:s"""
        return self._sim_step_size_ms * 1e-3

    def vis_radius(self) -> DoubleNDArr:
        """遮蔽半径, 单位:m, shape: (...,N,1)"""
        return self._vis_radius

    def position_e(self) -> DoubleNDArr:
        """position in NED local frame, unit: m, shape: (...,N,3)"""
        return self._pos_e

    def blh(self) -> DoubleNDArr:
        """当前(纬度,经度,高度), 单位:(deg,m), shape: (...,N,3)"""
        return self._blh

    def blh0(self) -> DoubleNDArr:
        """坐标原点(纬度,经度,高度), 单位:(deg,m), shape: (...,N,3)"""
        return self._blh0[..., :]

    def longitude_deg(self) -> DoubleNDArr:
        """longitude, unit: rad, shape: (...,N,1)"""
        return self._blh[..., 1:2]

    def latitude_deg(self) -> DoubleNDArr:
        """latitude, unit: rad, shape: (...,N,1)"""
        return self._blh[..., 0:1]

    def altitude_m(self) -> DoubleNDArr:
        """海拔 altitude, unit: m, shape: (...,N,1)"""
        return self._blh[..., 2:3]

    def g_e(self) -> DoubleNDArr:
        """NED地轴系重力加速度向量, unit: m/s^2, shape: (...,N,3)"""
        return self._g_e

    def sim_time_ms(self):
        """model simulation time, unit: ms, shape: (...,N,1)"""
        return self._sim_time_ms

    def sim_time_s(
        self,
    ) -> DoubleNDArr:
        """model simulation time, unit: s, shape: (...,N,1)"""
        return self.sim_time_ms() * 1e-3

    @abstractmethod
    def reset(self, mask: SupportedMaskType | None):
        """状态复位"""
        mask = self.proc_to_mask(mask)

        self._sim_time_ms[mask, :] = 0.0
        self.health_point[mask, :] = 100.0
        self.set_status(self.STATUS_INACTIVE, mask)

    def set_status(self, status: int | NDArray, dst_idx: SupportedMaskType | None):
        """设置仿真运行状态"""
        dst_idx = self.proc_to_mask(dst_idx)
        self.status[dst_idx, :] = status

    def activate(self, mask: SupportedMaskType | None):
        """运行状态->激活"""
        self.set_status(self.STATUS_ALIVE, mask)

    @abstractmethod
    def run(self, mask: SupportedMaskType | None):
        """仿真推进"""
        self.update_sim_time(mask)

    def update_sim_time(self, mask: SupportedMaskType | None):
        """更新仿真时间"""
        mask = self.proc_to_mask(mask)
        self._sim_time_ms[mask, :] += self._sim_step_size_ms

    def _status_is(self, value: int | NDArray) -> BoolNDArr:
        """判断仿真状态是否为value, shape=(...,N,1)"""
        return cast(BoolNDArr, bkbn.equal(self.status, value))

    def is_inactive(self) -> BoolNDArr:
        """判断sims是否就绪, shape=(...,N,1)"""
        return self._status_is(self.STATUS_INACTIVE)

    def is_alive(self) -> BoolNDArr:
        """判断sims是否运行, shape=(...,N,1)"""
        return self._status_is(self.STATUS_ALIVE)

    def is_dying(self) -> BoolNDArr:
        """判断sims是否即将死亡, shape=(...,N,1)"""
        return self._status_is(self.STATUS_DYING)

    def is_dead(self) -> BoolNDArr:
        """判断sims是否死亡, shape=(...,N,1)"""
        return self._status_is(self.STATUS_DEAD)

    def Q_eb(self) -> DoubleNDArr:
        """地轴系/体轴系四元数"""
        return self._Q_eb

    def Q_wb(self) -> DoubleNDArr:
        """风轴系/体轴系四元数"""
        return self._Q_wb

    def Q_ew(self) -> DoubleNDArr:
        """地轴系/风轴系四元数"""
        return self._Q_ew

    def tas(self) -> DoubleNDArr:
        """true air speed, unit: m/s, shape: (...,N,1)"""
        return self._tas

    def velocity_b(self) -> DoubleNDArr:
        """惯性速度 NED体轴系坐标 (U,V,W), unit: m/s, shape: (...,N,3)"""
        return self._vel_b

    def velocity_e(self) -> DoubleNDArr:
        """惯性速度 NED地轴系坐标 (V_N, V_E, V_D), unit: m/s, shape: (...,N,3)"""
        return self._vel_e

    def velocity_w(self) -> DoubleNDArr:
        """惯性速度 NED风轴系坐标 (TAS,0,0), unit: m/s, shape: (...,N,3)"""

        return self._vel_w

    def rpy_eb(self) -> DoubleNDArr:
        """体轴系 (roll, pitch, yaw) unit: rad, shape: (...,N,3)"""
        return self._rpy_eb

    def rpy_ew(self) -> DoubleNDArr:
        """风轴系 (mu, gamma, chi) unit: rad, shape: (...,N,3)"""
        return self._rpy_ew

    def mass(self) -> DoubleNDArr:
        """质量, unit: kg, shape: (...,N,1)"""
        return self._mass

    def inertia(self) -> DoubleNDArr:
        """体轴惯性矩阵, unit: kg*m^2, shape: (...,N,3,3)"""
        return self._I_b

    def inertia_inv(self) -> DoubleNDArr:
        """体轴惯性矩阵的逆矩阵, unit: kg^{-1}*m^{-2}, shape: (...,N,3,3)"""
        return self._I_b_inv

    # propagation modules
    def _ppgt_z2alt(self, mask: SupportedMaskType | None = None):
        """Z坐标->海拔"""
        mask = self.proc_to_mask(mask)
        self._blh[mask, 2:3] = self._blh0[mask, 2:3] - self._pos_e[mask, 2:3]

    def _ppgt_alt2z(self, mask: SupportedMaskType | None = None):
        """海拔->Z坐标"""
        mask = self.proc_to_mask(mask)
        self._pos_e[mask, 2:3] = self._blh0[mask, 2:3] - self._blh[mask, 2:3]

    def _ppgt_ned2blh(self, mask: SupportedMaskType | None = None):
        """NED地轴系坐标->纬经高"""
        mask = self.proc_to_mask(mask)
        p_np = self._pos_e[mask, :]
        # p_np = p_np.cpu().numpy()
        blh0 = self._blh0[mask, :]
        # blh0 = blh0.cpu().numpy()
        n, e, d = np.split(p_np, 3, axis=-1)
        lat0, lon0, alt0 = np.split(blh0, 3, axis=-1)
        lat, lon, alt = pymap3d.ned2geodetic(n, e, d, lat0, lon0, alt0)
        blh = np.concatenate([lat, lon, alt], axis=-1)
        self._blh[mask, :] = np.asarray(
            blh,
            # device=self.device,
            dtype=self.dtype,
        )

    def _ppgt_rpy_eb2Qeb(self, mask: SupportedMaskType | None = None):
        """地轴/体轴 欧拉角->四元数"""
        mask = self.proc_to_mask(mask)
        self._Q_eb[mask, :] = rpy2quat(self._rpy_eb[mask, :])

    def _ppgt_rpy_wb2Qwb(self, mask: SupportedMaskType | None = None):
        """体轴/风轴 欧拉角->四元数"""
        mask = self.proc_to_mask(mask)
        self._Q_wb[mask, :] = rpy2quat(self._rpy_wb[mask, :])

    def _ppgt_rpy_ew2Qew(self, mask: SupportedMaskType | None = None):
        """地轴/风轴 欧拉角->四元数"""
        mask = self.proc_to_mask(mask)
        self._Q_ew[mask, :] = rpy2quat(self._rpy_ew[mask, :])

    def _ppgt_Qeb2rpy_eb(self, mask: SupportedMaskType | None = None):
        """地轴/体轴 四元数->欧拉角"""
        mask = self.proc_to_mask(mask)
        self._rpy_eb[mask, :] = rpy2quat_inv(
            self._Q_eb[mask, :], self._rpy_eb[mask, 0:1]
        )

    def _ppgt_Qwb2rpy_wb(self, mask: SupportedMaskType | None = None):
        """风轴/体轴 四元数->欧拉角"""
        mask = self.proc_to_mask(mask)
        self._rpy_wb[mask, :] = rpy2quat_inv(
            self._Q_wb[mask, :], self._rpy_wb[mask, 0:1]
        )
        self._rpy_wb[mask, 0:1] = 0  # 风轴到体轴不定义滚转,恒为0

    def _ppgt_Qew2rpy_ew(self, mask: SupportedMaskType | None = None):
        """地轴/风轴 四元数->欧拉角"""
        mask = self.proc_to_mask(mask)
        self._rpy_ew[mask, :] = rpy2quat_inv(
            self._Q_ew[mask, :], self._rpy_ew[mask, 0:1]
        )

    def _ppgt_Vb2Ve(self, mask: SupportedMaskType | None = None):
        """惯性速度 体轴系->地轴系"""
        mask = self.proc_to_mask(mask)
        self._vel_e[mask, :] = quat_rotate(self._Q_eb[mask, :], self._vel_b[mask, :])

    def _ppgt_Ve2Vb(self, mask: SupportedMaskType | None = None):
        """惯性速度 地轴系->体轴系"""
        mask = self.proc_to_mask(mask)
        self._vel_b[mask, :] = quat_rotate_inv(
            self._Q_eb[mask, :], self._vel_e[mask, :]
        )

    def _ppgt_Ve2tas(self, mask: SupportedMaskType | None = None):
        """地轴系惯性速度->真空速tas"""
        mask = self.proc_to_mask(mask)
        self._tas[mask, :] = norm_(self._vel_e[mask, :])

    def _ppgt_Vb2tas(self, mask: SupportedMaskType | None = None):
        """体轴系惯性速度->真空速tas"""
        mask = self.proc_to_mask(mask)
        self._tas[mask, :] = norm_(self._vel_b[mask, :])

    def _ppgt_Vw2tas(self, mask: SupportedMaskType | None = None):
        """风轴系惯性速度->真空速tas"""
        mask = self.proc_to_mask(mask)
        self._tas[mask, :] = self._vel_w[mask, 0:1]

    def _ppgt_tas2Vw(self, mask: SupportedMaskType | None = None):
        """真空速->风轴系惯性速度"""
        mask = self.proc_to_mask(mask)
        self._vel_w[mask, 0:1] = self._tas[mask, :]
        self._vel_w[mask, 1:3] = 0.0

    def _ppgt_Vw2Vb(self, mask: SupportedMaskType | None = None):
        """真空速->体轴系惯性速度"""
        mask = self.proc_to_mask(mask)
        self._vel_b[mask, :] = quat_rotate_inv(
            self._Q_wb[mask, :], self._vel_w[mask, :]
        )

    def _ppgt_Vw2Ve(self, mask: SupportedMaskType | None = None):
        """真空速->地轴系惯性速度"""
        mask = self.proc_to_mask(mask)
        self._vel_e[mask, :] = quat_rotate(self._Q_ew[mask, :], self._vel_w[mask, :])

    def _ppgt_Vb2rpy_wb(self, mask: SupportedMaskType | None = None):
        """体轴系惯性速度-> 体轴/风轴 欧拉角(迎角,侧滑角)"""
        mask = self.proc_to_mask(mask)
        alpha, beta = uvw2alpha_beta(self._vel_b[mask, :])
        self.set_alpha(alpha, mask)
        self.set_beta(beta, mask)

    def _ppgt_QebQwb2Qew(self, mask: SupportedMaskType | None = None):
        """地轴/体轴 & 风轴/体轴 -> 地轴/风轴 四元数"""
        mask = self.proc_to_mask(mask)
        self._Q_ew[mask, :] = quat_mul(
            self._Q_eb[mask, :], quat_conj(self._Q_wb[mask, :])
        )

    def _ppgt_QewQwb_to_Qeb(self, mask: SupportedMaskType | None = None):
        """地轴/风轴 & 风轴/体轴 -> 地轴/体轴 四元数"""
        mask = self.proc_to_mask(mask)
        self._Q_eb[mask, :] = quat_mul(self._Q_ew[mask, :], self._Q_wb[mask, :])

    def roll(self):
        """体轴滚转角 in (-pi,pi], unit: rad, shape: (...,N,1)"""
        mask = Ellipsis
        return self._rpy_eb[mask, 0:1]

    def pitch(self):
        """体轴俯仰角 in [-pi/2,pi/2], unit: rad, shape: (...,N,1)"""
        mask = Ellipsis
        return self._rpy_eb[mask, 1:2]

    def yaw(self):
        """体轴偏航角 in (-pi,pi], unit: rad, shape: (...,N,1)"""
        mask = Ellipsis
        return self._rpy_eb[mask, 2:3]

    def alpha(self):
        """迎角 in [-pi/2,pi/2], unit: rad, shape: (...,N,1)"""
        mask = Ellipsis
        return self._rpy_wb[mask, 1:2]

    def beta(self):
        """侧滑角 in [-pi,pi], unit: rad, shape: (...,N,1)"""
        mask = Ellipsis
        return -(self._rpy_wb[mask, 2:3])

    def mu(self):
        """速度系滚转角 in (-pi,pi], unit: rad, shape: (...,N,1)"""
        mask = Ellipsis
        return self._rpy_ew[mask, 0:1]

    def gamma(self):
        """速度系俯仰角 in [-pi/2,pi/2], unit: rad, shape: (...,N,1)"""
        mask = Ellipsis
        return self._rpy_ew[mask, 1:2]

    def chi(self):
        """速度系偏航角 in (-pi,pi], shape: (...,N,1)"""
        mask = Ellipsis
        return self._rpy_ew[mask, 2:3]

    def set_roll(self, value: np.ndarray, dst_index: SupportedMaskType | None):
        """赋值 体轴滚转角(无级联操作) 等价于 roll[dst_index,:]=value"""
        dst_index = self.proc_to_mask(dst_index)
        self._rpy_eb[dst_index, 0:1] = value

    def set_pitch(self, value: np.ndarray, dst_index: SupportedMaskType | None):
        """赋值 体轴俯仰角(无级联操作)  等价于 pitch[dst_index,:]=value"""
        dst_index = self.proc_to_mask(dst_index)
        self._rpy_eb[dst_index, 1:2] = value

    def set_yaw(self, value: np.ndarray, dst_index: SupportedMaskType | None):
        """赋值 体轴偏航角(无级联操作) 等价于 yaw[dst_index,:]=value"""
        dst_index = self.proc_to_mask(dst_index)
        self._rpy_eb[dst_index, 2:3] = value

    def set_alpha(self, value: np.ndarray, dst_index: SupportedMaskType | None):
        """赋值 迎角(无级联操作) 等价于 alpha[dst_index,:]=value"""
        dst_index = self.proc_to_mask(dst_index)
        self._rpy_wb[dst_index, 1:2] = value

    def set_beta(self, value: np.ndarray, dst_index: SupportedMaskType | None):
        """赋值 侧滑角(无级联操作) 等价于 beta[dst_index,:]=value"""
        dst_index = self.proc_to_mask(dst_index)
        self._rpy_wb[dst_index, 2:3] = -value

    def set_mu(self, value: np.ndarray, dst_index: SupportedMaskType | None):
        """赋值 航迹滚转角(无级联操作) 等价于 mu[dst_index,:]=value"""
        dst_index = self.proc_to_mask(dst_index)
        self._rpy_ew[dst_index, 0:1] = value

    def set_gamma(self, value: np.ndarray, dst_index: SupportedMaskType | None):
        """赋值 航迹俯仰角(无级联操作) 等价于 gamma[dst_index,:]=value"""
        dst_index = self.proc_to_mask(dst_index)
        self._rpy_ew[dst_index, 1:2] = value

    def set_chi(self, value: np.ndarray, dst_index: SupportedMaskType | None):
        """赋值 航迹偏航角(无级联操作) 等价于 chi[dst_index,:]=value"""
        dst_index = self.proc_to_mask(dst_index)
        self._rpy_ew[dst_index, 2:3] = value
