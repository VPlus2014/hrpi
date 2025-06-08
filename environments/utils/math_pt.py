# 250605 飞仿相关张量运算扩展(针对pytorch)
from __future__ import annotations
import math
from typing import TYPE_CHECKING, TypeVar

_PI = math.pi
_2PI = math.tau

if TYPE_CHECKING:
    from typing_extensions import TypeAlias
from typing import Sequence, Callable, Union
import torch

_bkbn = torch
_sum = _bkbn.sum
_stack = _bkbn.stack
_cat = _bkbn.cat
_zeros_like = _bkbn.zeros_like
_ones_like = _bkbn.ones_like
_sin = _bkbn.sin
_cos = _bkbn.cos
_atan2 = _bkbn.atan2
_asin = _bkbn.asin
_where = _bkbn.where
_abs = _bkbn.abs
_pow = _bkbn.pow
_sqrt = _bkbn.sqrt
_split = _bkbn.split
_chunk = _bkbn.chunk
_norm = _bkbn.norm
_cross = _bkbn.cross
_clamp = _bkbn.clamp
_clip = _clamp
_reshape = _bkbn.reshape
_ones = _bkbn.ones
_zeros = _bkbn.zeros
_broadcast_arrays = _bkbn.broadcast_tensors
_Tensor = torch.Tensor
_T = TypeVar("_T", _Tensor, torch.NumberType, float, int)


def affcmb(w, a, b):
    """Affine combination of two tensors.

    $$
    (1-w) * a + w * b
    $$

    Args:
        w: Weights for the affine combination, shape (..., 1).
        a: First scalar or tensor, shape (..., dims).
        b: Second scalar or tensor, shape (..., dims).

    Returns:
        Affine combination of a and b, shape (..., dims).
    """
    return a + (b - a) * w


def affcmb_inv(y, a, b):
    """
    仿射组合的逆运算

    Args:
        y: a+w*(b-a), shape (..., 1)
        a: 端点1, shape (..., dims)
        b: 端点2, shape (..., dims)

    Returns:
        w: 仿射系数
    """
    m = b - a
    y_ = y - a
    eps = 1e-6
    _mis0 = (m < eps) & (m > -eps)
    _fix = _mis0 + 0.0
    w = (y_ * (1 - _fix)) / (m + _fix)
    return w


def B01toI(x):
    """B(0,1)=[-1,1]->I=[0,1]"""
    return (x + 1) * 0.5


def ItoB01(x):
    """I=[0,1]->B(0,1)=[-1,1]"""
    return x * 2 - 1


@torch.jit.script
def _normalize(x: _Tensor, eps: float = 1e-9) -> _Tensor:
    return x / x.norm(dim=-1, p=2, keepdim=True).clip(eps)


@torch.jit.script
def _quat_split_keepdim(q: _Tensor):
    q0, q1, q2, q3 = q.split([1, 1, 1, 1], dim=-1)
    return q0, q1, q2, q3


@torch.jit.script
def _quat_re(q: _Tensor):
    return q.split([1, 3], -1)[0]  # [...,0:1] 不能jit


@torch.jit.script
def _quat_im(q: _Tensor):
    return q.split([1, 3], -1)[1]  # # [...,1:4] 不能jit


@torch.jit.script
def _quat_rect_re(q: _Tensor) -> _Tensor:
    """(解决双倍缠绕问题)返回实部非负的等价四元数"""
    reQ = _quat_re(q)
    return _where(reQ < 0, -q, q)


@torch.jit.script
def _an2quat(
    angle_rad: _Tensor,
    axis: _Tensor,
):
    assert (
        angle_rad.shape[:-1] == axis.shape[:-1]
    ), "Angle and axis must have the same shape except for the last dimension. "
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    assert axis.shape[-1] == 3, "Axis must be a 3D vector."
    r = torch.norm(axis, p=2, dim=-1, keepdim=True)
    axis = axis / r.clamp(min=1e-9)

    half_angle = angle_rad * 0.5
    q0 = _cos(half_angle)
    qv = _sin(half_angle) * axis
    q = torch.cat([q0, qv], dim=-1)  # (...,4)
    q = _quat_rect_re(q)  # 确保实部非负
    return q


def an2quat(
    angle_rad: _Tensor | float,
    axis: _Tensor,
) -> _Tensor:
    if not isinstance(angle_rad, _Tensor):
        angle_rad = _zeros_like(axis[..., 0:1]) + angle_rad  # assert: 广播相容
    return _an2quat(angle_rad, axis)


# 定义基元旋转矩阵
def Lx(angle_rad: _Tensor) -> _Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)

    L_flat = (one, zero, zero, zero, cos, sin, zero, -sin, cos)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))


def Ly(angle_rad: _Tensor) -> _Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)

    L_flat = (cos, zero, -sin, zero, one, zero, sin, zero, cos)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))


def Lz(angle_rad: _Tensor) -> _Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)

    L_flat = (cos, sin, zero, -sin, cos, zero, zero, zero, one)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))


@torch.jit.script
def _Qx(angle_rad: _Tensor) -> _Tensor:
    """绕x轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cos = torch.cos(ah)
    sin = torch.sin(ah)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, sin, zero, zero)
    return torch.cat(Q_flat, -1)


@torch.jit.script
def _Qy(angle_rad: _Tensor) -> _Tensor:
    """绕y轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cos = torch.cos(ah)
    sin = torch.sin(ah)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, zero, sin, zero)
    return torch.cat(Q_flat, -1)


@torch.jit.script
def _Qz(angle_rad: _Tensor) -> _Tensor:
    """绕z轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cos = torch.cos(ah)
    sin = torch.sin(ah)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, zero, zero, sin)
    return torch.cat(Q_flat, -1)


def Qx(angle_rad: _Tensor) -> _Tensor:
    """绕x轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qx(angle_rad)


def Qy(angle_rad: _Tensor) -> _Tensor:
    """绕y轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qy(angle_rad)


def Qz(angle_rad: _Tensor) -> _Tensor:
    """绕z轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qz(angle_rad)


@torch.jit.script  # 15%
def _rpy2mat(rpy: _Tensor):
    r, p, y = torch.chunk(rpy, 3, dim=-1)  # (...,1)
    newshape = y.shape[:-1] + (3, 3)
    _1 = torch.ones_like(y)
    _0 = torch.zeros_like(y)
    c1 = torch.cos(r)
    s1 = torch.sin(r)
    c2 = torch.cos(p)
    s2 = torch.sin(p)
    c3 = torch.cos(y)
    s3 = torch.sin(y)
    r1 = torch.cat([_1, _0, _0, _0, c1, -s1, _0, s1, c1], -1).reshape(
        newshape
    )  # (...,3,3)
    r2 = torch.cat([c2, _0, s2, _0, _1, _0, -s2, _0, c2], -1).reshape(
        newshape
    )  # (...,3,3)
    r3 = torch.cat([c3, -s3, _0, s3, c3, _0, _0, _0, _1], -1).reshape(
        newshape
    )  # (...,3,3)
    return r3 @ r2 @ r1  # (...,3,3)


@torch.jit.script  # 21%
def _rpy2mat_inv(Reb: _Tensor, roll_ref_rad: _Tensor):
    assert (
        roll_ref_rad.shape[-1] == 1
    ), "Roll reference angle's last dimension must be 1."
    assert (
        Reb.shape[:-2] == roll_ref_rad.shape[:-1]
    ), "Reb and roll_ref_rad must have the same shape except for the last dimension."
    s2 = torch.clip(-Reb[..., 2, 0], -1.0, 1.0)
    pitch = torch.arcsin(s2)
    c2s1 = Reb[..., 2, 1]  # R32
    c2c1 = Reb[..., 2, 2]  # R33
    c2s3 = Reb[..., 1, 0]  # R21
    c2c3 = Reb[..., 0, 0]  # R11

    eps = 1e-4  # 万向节死锁阈值(jit 警告: 别放在参数表里)
    gl = 1 - torch.abs(s2) < eps
    #
    roll_ref_rad = roll_ref_rad.squeeze(-1)  # (...,)
    s1 = torch.sin(roll_ref_rad)
    c1 = torch.cos(roll_ref_rad)
    roll_ref_rad = torch.arctan2(s1, c1)
    #
    s2s1 = s1 * s2
    s2c1 = c1 * s2
    R22 = Reb[..., 1, 1]
    R12 = Reb[..., 0, 1]
    R23 = Reb[..., 1, 2]
    R13 = Reb[..., 0, 2]
    s3 = s2s1 * R22 - c1 * R12 + s2c1 * R23 + s1 * R13
    c3 = c1 * R22 + s2s1 * R12 - s1 * R23 + s2c1 * R13

    roll = torch.where(gl, roll_ref_rad, torch.arctan2(c2s1, c2c1))
    yaw = torch.where(gl, torch.arctan2(s3, c3), torch.arctan2(c2s3, c2c3))

    rpy = _stack([roll, pitch, yaw], -1)
    return rpy


@torch.jit.script
def _quat_norm(q: _Tensor) -> _Tensor:
    return q.norm(2, -1, True)


@torch.jit.script
def _quat_normalize(q: _Tensor) -> _Tensor:
    return q / q.norm(2, -1, True).clip_(1e-9, None)


def quat_norm(q: _Tensor) -> _Tensor:
    """Computes the norm of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The norm of the quaternion. shape: (..., 1).
    """
    return _quat_norm(q)


def quat_normalize(q: _Tensor) -> _Tensor:
    """Normalizes a quaternion to have unit length.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).
    Returns:
        The normalized quaternion in (w, x, y, z). shape: (..., 4).
    """
    return _quat_normalize(q)


def normalize(x: _Tensor, eps: float = 1e-9) -> _Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Rets:
        Normalized tensor of shape (N, dims).
    """
    return _normalize(x)


def is_normalized(v: _Tensor, eps: float = 1e-6) -> _Tensor:
    return ((v.norm(2, -1, keepdim=True) - 1).abs() <= eps).all()


@torch.jit.script
def _quat_conj(q: _Tensor) -> _Tensor:
    q0, qv = _split(q, [1, 3], -1)
    return _cat((q0, -qv), dim=-1)


@torch.jit.script
def _quat_inv(q: _Tensor) -> _Tensor:
    return _quat_conj(q) / _clip(q.pow(2).sum(-1, keepdim=True), 1e-16, None)


def quat_inv(q: _Tensor) -> _Tensor:
    """Computes the inverse of a quaternion."""
    return _quat_inv(q)


def quat_split_keepdim(q: _Tensor):
    """拆分为分量.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        Re(q): 实部, shape: (..., 1).
        Im(q): 虚部, shape: (..., 3).
    """
    return _quat_split_keepdim(q)


def quat_re(q: _Tensor) -> _Tensor:
    """Get the real part of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The real part of the quaternion. shape: (..., 1).
    """
    return _quat_re(q)


def quat_im(q: _Tensor) -> _Tensor:
    """Get the imaginary part of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The imaginary part of the quaternion. shape: (..., 3).
    """
    return _quat_im(q)


def quat_conj(q: _Tensor) -> _Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). shape: (..., 4).

    Rets:
        The conjugate quaternion in (w, x, y, z). shape: (..., 4).
    """
    return _quat_conj(q)


def quat2prodmat(q: _Tensor) -> _Tensor:
    r"""Computes the matrix of a quaternion.
    四元数左乘等效为矩阵乘法（线性算子作用）

    $$
    q \otimes p = M(q) p, \forall p \in\mathbb{H}
    $$

    Args:
        q: The quaternion orientation in (w, x, y, z). shape: (..., 4).

    Rets:
        The matrix of quaternion. shape: (..., 4, 4).
    """
    q0, q1, q2, q3 = q.unbind(dim=-1)
    nq1 = -q1
    nq2 = -q2
    nq3 = -q3
    mat_flat = (q0, nq1, nq2, nq3, q1, q0, nq3, q2, q2, q3, q0, nq1, q3, nq2, q1, q0)
    return torch.stack(mat_flat, -1).reshape(q0.shape + (4, 4))


def mati(q: _Tensor) -> _Tensor:
    """Computes the matrix of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). shape: (..., 4).

    Rets:
        The matrix inverse of quaternion. shape: (..., 4, 4).
    """
    q0, q1, q2, q3 = q.unbind(dim=-1)

    mat_flat = (q0, -q1, -q2, -q3, q1, q0, q3, -q2, q2, -q3, q0, q1, q3, q2, -q1, q0)
    return torch.stack(mat_flat, -1).reshape(q0.shape + (4, 4))


@torch.jit.script
def _quat2mat_sqrt(q: _Tensor) -> _Tensor:
    q0, q1, q2, q3 = q.split([1, 1, 1, 1], -1)  # (...,1)
    nq0 = -q0
    nq1 = -q1
    nq2 = -q2
    nq3 = -q3
    m = _stack(
        [
            _cat([q0, nq1, nq2, nq3], -1),
            _cat([nq1, nq0, q3, nq2], -1),
            _cat([nq2, nq3, nq0, q1], -1),
            _cat([nq3, q2, nq1, nq0], -1),
        ],
        -2,
    )
    return m


def quat2mat_sqrt(q: _Tensor) -> _Tensor:
    r"""Computes the square root of the rotation matrix of a quaternion.

    $$
    S=J M(Q)\\
    S^2 h = Q\otimes h \otimes Q^*, \forall h \in \mathbb{H}
    $$

    Args:
        q: The quaternion orientation in (w, x, y, z). shape: (..., 4).

    Rets:
        The square root of the rotation matrix of the quaternion. shape: (..., 4, 4).
    """
    return _quat2mat_sqrt(q)


@torch.jit.script
def _quat_mul(p: _Tensor, q: _Tensor) -> _Tensor:
    reP, imP = _split(p, [1, 3], dim=-1)
    reQ, imQ = _split(q, [1, 3], dim=-1)
    pq = _cat(
        [
            reP * reQ - (imP * imQ).sum(-1, keepdim=True),
            reP * imQ + reQ * imP + _cross(imP, imQ, -1),
        ],
        -1,
    )
    # 暴力版(慢)
    # p0, p1, p2, p3 = _chunk(p, 4, -1)  # (...,1)
    # q0, q1, q2, q3 = _chunk(q, 4, -1)  # (...,1)
    # r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    # r1 = p0 * q1 + q0 * p1 + p2 * q3 - p3 * q2
    # r2 = p0 * q2 + q0 * p2 + p3 * q1 - p1 * q3
    # r3 = p0 * q3 + q0 * p3 + p1 * q2 - p2 * q1
    # pq = _cat([r0, r1, r2, r3], -1)  # (...,4)
    return pq


import numpy as np


@torch.jit.script
def _rpy2quat(rpy_rad: _Tensor) -> _Tensor:
    r_hf, p_hf, y_hf = torch.chunk(rpy_rad * 0.5, 3, dim=-1)  # (...,1)
    _0 = _zeros_like(r_hf)
    q1 = _cat([_cos(r_hf), _sin(r_hf), _0, _0], -1)
    q2 = _cat([_cos(p_hf), _0, _sin(p_hf), _0], -1)
    q3 = _cat([_cos(y_hf), _0, _0, _sin(y_hf)], -1)
    return _quat_mul(_quat_mul(q3, q2), q1)


def quat_mul(p: _Tensor, q: _Tensor) -> _Tensor:
    r"""Multiply two quaternions together.

    $$
    Re(p \otimes q) = Re(p) * Re(q) - Im(p) \dot Im(q)
    Im(p \otimes q) = Re(p) * Im(q) + Re(q) * Im(p) + Im(p) \times Im(q)\\
    $$

    Args:
        p: The first quaternion in (w, x, y, z). shape: (..., 4).
        q: The second quaternion in (w, x, y, z). shape: (..., 4).

    Rets:
        The product of the two quaternions in (w, x, y, z). shape: (..., 4).
    """
    return _quat_mul(p, q)


def _crossmat_kern(v: _Tensor) -> _Tensor:
    x, y, z = v.unbind(dim=-1)  # (...,1)
    _0 = torch.zeros_like(x)
    return _cat([_0, -z, y, z, _0, -x, -y, x, _0], -1).reshape(
        v.shape[:-1] + (3, 3)
    )  # (...,3,3)


def crossmat(v: _Tensor):
    r"""
    左叉积矩阵
    $v_\times a= v \times a$
    Args:
        v: The vector in (x, y, z). shape: (..., 3).
    Returns:
        The cross product matrix of vector v. shape: (..., 3, 3).
    """
    return _crossmat_kern(v)


@torch.jit.script
def _quat2mat(q: _Tensor) -> _Tensor:
    # 旧版:只支持输入q为 (B,4) 二阶张量
    # rotation_matrix = torch.bmm(mati(q), quat2prodmat(quat_conjugate(q)))
    # return rotation_matrix[..., 1:, 1:]

    # 新版:支持任意尺寸&计算耗时减少 20%
    # assert is_normalized(q), "Quaternion must be normalized."
    q0, q1, q2, q3 = quat_split_keepdim(q)  # (...,1)
    _2q0 = q0 + q0
    _2q1 = q1 + q1
    _2q2 = q2 + q2
    _2q3 = q3 + q3
    _2q0q1 = _2q0 * q1
    _2q0q2 = _2q0 * q2
    _2q0q3 = _2q0 * q3
    _2q1q2 = _2q1 * q2
    _2q1q3 = _2q1 * q3
    _2q2q3 = _2q2 * q3
    _2q0q0_1 = _2q0 * q0 - 1.0
    A11 = _2q1 * q1 + _2q0q0_1
    A22 = _2q2 * q2 + _2q0q0_1
    A33 = _2q3 * q3 + _2q0q0_1
    A12 = _2q1q2 - _2q0q3
    A21 = _2q1q2 + _2q0q3
    A13 = _2q1q3 + _2q0q2
    A31 = _2q1q3 - _2q0q2
    A23 = _2q2q3 - _2q0q1
    A32 = _2q2q3 + _2q0q1
    m = _stack(
        [
            _cat([A11, A12, A13], -1),
            _cat([A21, A22, A23], -1),
            _cat([A31, A32, A33], -1),
        ],
        -2,
    )  # (...,3,3)
    return m


def quat2mat(q: _Tensor) -> _Tensor:
    r"""
    四元数->3D旋转矩阵
    $$
    R(q) Im(h) = |q| Im(q \otimes h \otimes q^*)
    $$
    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Rets:
        The rotation matrix of quaternion. shape: (..., 3, 3).
    """
    return _quat2mat(q)


@torch.jit.script
def _quat_rot(q: _Tensor, v: _Tensor):
    reQ, imQ = q.split([1, 3], -1)
    _2q0 = reQ + reQ
    _2qv = imQ + imQ
    u = (
        (_2q0 * reQ - 1.0) * v
        + _2qv * _sum(imQ * v, -1, keepdim=True)
        + _2q0 * _cross(imQ, v, -1)
    )
    return u


def quat_rotate(q: _Tensor, v: _Tensor) -> _Tensor:
    r"""Rotate a vector by a quaternion along the last dimension of q and v.

    $$
    Im(Q \otimes (0,v) \otimes Q^*)
    $$

    Args:
        q: Normalized quaternion in (w, x, y, z). shape: (..., 4).
        v: 3D vector in (x, y, z). shape: (..., 3).

    Returns:
        The rotated vector in (x, y, z). shape: (..., 3).
    """
    return _quat_rot(q, v)


def quat_rotate_inv(q: _Tensor, v: _Tensor) -> _Tensor:
    r"""Inverse rotate a vector by a quaternion along the last dimension of q and v.

    $$
    Im(Q^* \otimes (0,v) \otimes Q)
    $$

    Args:
        q: Normalized quaternion in (w, x, y, z). shape: (..., 4).
        v: 3D vector in (x, y, z). shape: (..., 3).

    Returns:
        The rotated vector in (x, y, z). shape: (..., 3).
    """
    # rotation_matrix = quat2rotmat(quat_conjugate(q))
    # u = torch.bmm(rotation_matrix, v.unsqueeze(-1)).squeeze(-1)

    # assert is_normalized(q), "Quaternion must be normalized."
    return quat_rotate(quat_conj(q), v)


def quat_enu_ned() -> _Tensor:
    return quat_mul(
        Qz(_Tensor([[torch.pi * 0.5]])),
        quat_mul(Qy(_Tensor([[0]])), Qx(_Tensor([[torch.pi]]))),
    )


def rpy2mat(
    rpy: _Tensor,
) -> _Tensor:
    r"""
    Z-Y-X 内旋矩阵
    $$
    R = R_z(\psi) R_y(\theta) R_x(\phi)
    $$
    Args:
        rpy (_NDArr): (滚转\psi,俯仰\theta,偏航\phi),unit:rad, shape (..., 3)
    Returns:
        _NDArr: 旋转矩阵, shape (..., 3, 3)
    """
    return _rpy2mat(rpy)


def rpy2mat_inv(Reb: _Tensor, roll_ref_rad: _Tensor | float = 0.0) -> _Tensor:
    """
    rpy2mat 的逆映射，当发生万向节死锁时，需要给出 roll_ref_rad
    Args:
        Reb (_NDArr): 旋转矩阵, shape (..., 3, 3)
        roll_ref_rad (_NDArr | float): 滚转角参考值, 单位: rad, shape (..., 1) 或标量
    """
    assert Reb.shape[-2:] == (3, 3), "expected matrix shape (...,3,3), got {}".format(
        Reb.shape
    )
    if not isinstance(roll_ref_rad, _Tensor):
        roll_ref_rad = roll_ref_rad + _zeros_like(Reb[..., 0, 0:1])  # assert: 广播相容
    return _rpy2mat_inv(Reb, roll_ref_rad)


def rpy2quat(rpy_rad: _Tensor) -> _Tensor:
    r"""
    输入欧拉角(rad)，输出规范四元数
    $$
    Q=Q_{e3,yaw}*Q_{e2,pitch}*Q_{e1,roll}
    $$
    Args:
        rpy_rad (_NDArr): 输入欧拉角(rad), shape: (..., 3)
    Returns:
        _NDArr: 规范四元数, shape: (..., 4)
    """
    return _rpy2quat(rpy_rad)


def rpy2quat_inv(q: _Tensor, roll_ref_rad: _Tensor | float = 0.0) -> _Tensor:
    r"""四元数反解欧拉角
    $$
    q=Q_z(yaw)\otimes Q_y(pitch) \otimes Q_x(roll)
    $$

    Args:
        q (_NDArr): 规范四元数 shape: (..., 4).
        roll_ref_rad (_NDArr | float, optional): 当前滚转角(死锁时备用) shape: (..., 1) or scalar. Defaults to 0.0.

    Returns:
        _NDArr: 欧拉角(滚转\phi,俯仰\theta,偏航\psi),单位:rad, shape (..., 3)
    """
    Reb = _quat2mat(q)
    rpy = rpy2mat_inv(Reb, roll_ref_rad)
    return rpy


def euler_from_quat(q: _Tensor, roll_ref_rad: _Tensor | float = 0.0) -> _Tensor:
    # q0, q1, q2, q3 = q.unbind(dim=-1)
    # # 隐患: 万向节死锁求解

    # phi = torch.arctan2(2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2))
    # theta = torch.arcsin(torch.clamp(-2 * (q3 * q1 - q0 * q2), -1, 1))
    # psi = torch.arctan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q2 * q2 + q3 * q3))

    # return torch.stack([phi, theta, psi], dim=-1)

    return rpy2quat_inv(q, roll_ref_rad)


@torch.jit.script
def _ned2aer(xyz: _Tensor) -> _Tensor:
    _R = torch.norm(xyz, p=2, dim=-1, keepdim=True)
    x, y, z = _split(xyz, [1, 1, 1], dim=-1)  # (...,1)
    _is0 = _R < 1e-3  # 过零处理
    _1 = _ones_like(x)
    _0 = _zeros_like(x)
    x = _where(_is0, _1, x)
    y = _where(_is0, _0, y)
    z = _where(_is0, _0, z)
    rxy2 = _pow(y, 2) + _pow(x, 2)
    rxy = _sqrt(rxy2)
    slantRange = torch.sqrt(rxy2 + torch.pow(z, 2))

    elev = _atan2(-z, rxy)  # -> [-pi/2,pi/2]
    azi = _atan2(x, y)  # -> [0,2pi)
    # azi = azi % _2PI
    return torch.stack([azi, elev, slantRange], dim=-1)


def ned2aer(xyz: _Tensor) -> _Tensor:
    r"""求解 NED xyz 直角坐标对应的 方位角 azimuth, 俯仰角 elevation, 距离 r\
    即 (r,0,0) 依次 绕Z内旋 azimuth, 绕Y内旋 elevation 得到 (x,y,z), 右手定则

    Args:
        xyz (_NDArr): NED 直角坐标 shape: (...,3)

    Returns:
        aer (_NDArr): shape: (...,3)
            azimuth \in [-pi,pi)
            elevation \in [-pi/2,pi/2]
            slant range \in [0,inf)
    """
    return _ned2aer(xyz)


@torch.jit.script
def _aer2ned(aer: _Tensor) -> _Tensor:
    az, el, r = torch.chunk(aer, 3, -1)  # (...,1)

    rxy = r * torch.cos(el)
    z = -r * torch.sin(el)
    x = rxy * torch.cos(az)
    y = rxy * torch.sin(az)
    return torch.cat([x, y, z], -1)


def aer2ned(aer: _Tensor) -> _Tensor:
    """
    ned2aer 的逆映射
    Args:
        aer (_NDArr): 方位角 azimuth, 俯仰角 elevation, 距离 r\
            即 (r,0,0) 依次 绕Z内旋 azimuth, 绕Y内旋 elevation 得到 (x,y,z), 右手定则 shape: (...,3)

    Returns:
        ned (_NDArr): NED 直角坐标 shape: (...,3)
    """
    return _aer2ned(aer)


@torch.jit.script
def _uvw2alpha_beta(uvw: _Tensor):
    uvw = normalize(uvw)
    u, v, w = torch.chunk(uvw, 3, -1)  # (...,1)
    alpha = torch.atan2(w, u)
    beta = torch.asin(v)
    return alpha, beta


def uvw2alpha_beta(uvw: _Tensor):
    r"""
    NED 体轴速度分量 (U,V,W)->(\alpha,\beta)\
    坐标系 旋转关系\
    $$
    \Phi_v R_z(-\beta) R_y(\alpha) = \Phi_b
    $$

    (U,V,W) 单位向量的分解

    $$
    i_{v/b} = ( \cos\beta \cos\alpha, 
                \sin\beta, 
                \cos\beta \sin\alpha)
    $$

    Args:
        uvw (_NDArr): 体轴速度分量 shape: (...,3)

    Returns:
        alpha (_NDArr): 迎角 \in [-pi/2,pi/2] shape: (...,1)
        beta (_NDArr): 侧滑角 \in (-pi,pi] shape: (...,1)
    """
    # assert (norm(uvw)>0).all()
    return _uvw2alpha_beta(uvw)


@torch.jit.script
def _vec_cosine(v1: _Tensor, v2: _Tensor, n1: _Tensor, n2: _Tensor):
    eps = 1e-6
    v1_is_zero = n1 < eps
    v2_is_zero = n2 < eps
    any_zero = v1_is_zero | v2_is_zero
    c = torch.where(
        any_zero,
        torch.zeros_like(n1),
        torch.sum(v1 * v2, -1, keepdim=True) / (n1 * n2),
    )
    return c


def vec_cosine(
    v1: _Tensor,
    v2: _Tensor,
    n1: _Tensor | float | None = None,
    n2: _Tensor | float | None = None,
):
    r"""
    计算两个向量的余弦值
    $$
    cos(v1,v2)=\frac{v1\cdot v2}{|v1||v2|}
    $$
    Args:
        v1 (_NDArr): 向量1 shape: (...,3)
        v2 (_NDArr): 向量2 shape: (...,3)
        n1 (_NDArr|float|None): 向量1的长度 shape: (...,1) or scalar
        n2 (_NDArr|float|None): 向量2的长度 shape: (...,1) or scalar

    Returns:
        _NDArr: 余弦值 shape: (...,1)
    """
    if not isinstance(n1, _Tensor):
        n1 = torch.norm(v1, p=2, dim=-1, keepdim=True)
    if not isinstance(n2, _Tensor):
        n2 = torch.norm(v2, p=2, dim=-1, keepdim=True)
    return _vec_cosine(v1, v2, n1, n2)


def _herp(
    position_0: _Tensor,
    velocity_0: _Tensor,
    position_1: _Tensor,
    velocity_1: _Tensor,
    t: _Tensor,
) -> _Tensor:
    position = torch.cat([position_0, position_1], dim=-1)
    velocity = torch.cat([velocity_0, velocity_1], dim=-1)
    dtype = position.dtype
    device = position.device
    x = torch.cat(
        [
            torch.zeros(size=(position.shape[0], 1), dtype=dtype, device=device),
            torch.ones(size=(position.shape[0], 1), dtype=dtype, device=device),
        ],
        dim=-1,
    )

    r, c1 = x.shape
    _, c2 = t.shape

    x_prime_1 = x.repeat(1, c1).reshape(r, c1, c1).permute(0, 2, 1)
    x_prime_mask = torch.eye(c1, device=device).repeat(r, 1, 1)
    x_prime_2 = x_prime_1 - x_prime_mask * x_prime_1
    x_prime = x_prime_2.unsqueeze(1).repeat(1, c2, 1, 1)

    t_prime_1 = t.unsqueeze(2).repeat(1, 1, c1)
    t_prime = t_prime_1.unsqueeze(3).repeat(1, 1, 1, c1)

    l_num_1 = t_prime - x_prime
    l_num_mask = (
        torch.eye(c1, device=device).repeat(r * c2, 1, 1).reshape(r, c2, c1, c1)
    )
    l_num_2 = l_num_1 - l_num_mask * l_num_1 + l_num_mask
    l_num = torch.prod(l_num_2, dim=2)

    l_den_1 = torch.prod((x_prime_1.permute(0, 2, 1) - x_prime_1) + x_prime_mask, dim=1)
    l_den = l_den_1.unsqueeze(1).repeat(1, c2, 1)

    l = l_num / l_den

    l_prime_1 = 1.0 / ((x_prime_1.permute(0, 2, 1) - x_prime_1) + x_prime_mask)
    l_prime_2 = torch.sum(l_prime_1, dim=1) - 1
    l_prime = l_prime_2.unsqueeze(1).repeat(1, c2, 1)

    # Create x_prime_3 for functions A and B
    x_prime_3 = x.unsqueeze(1).repeat(1, c2, 1)

    # Calculate function B
    B = (t_prime_1 - x_prime_3) * l * l

    # Calculate function A
    A = (1 - 2 * (t_prime_1 - x_prime_3) * l_prime) * l * l

    # Calculate final result H
    A_prime = position.unsqueeze(1).repeat(1, c2, 1) * A
    B_prime = velocity.unsqueeze(1).repeat(1, c2, 1) * B
    H = torch.sum(A_prime, dim=2) + torch.sum(B_prime, dim=2)

    return H


def herp(
    position_0: _Tensor,
    velocity_0: _Tensor,
    position_1: _Tensor,
    velocity_1: _Tensor,
    t: _Tensor,
) -> _Tensor:
    """
    args:
        position_0: shape: [n, 3]
        velocity_0: shape: [n, 3]
        position_1: shape: [n, 3]
        velocity_1: shape: [n, 3]
        t: shape: [n, m]

    rets:
        position: shape: [n, m, 3]
    """
    n = _herp(
        position_0[..., 0:1],
        velocity_0[..., 0:1],
        position_1[..., 0:1],
        velocity_1[..., 0:1],
        t,
    )  # n, shape: [n, m]
    e = _herp(
        position_0[..., 1:2],
        velocity_0[..., 1:2],
        position_1[..., 1:2],
        velocity_1[..., 1:2],
        t,
    )  # n, shape: [n, m]
    d = _herp(
        position_0[..., 2:3],
        velocity_0[..., 2:3],
        position_1[..., 2:3],
        velocity_1[..., 2:3],
        t,
    )  # n, shape: [n, m]

    position = torch.stack([n, e, d], dim=-1)
    return position


def lerp(v_0: _Tensor, v_1: _Tensor, t: _Tensor) -> _Tensor:
    v_0 = v_0.unsqueeze(1).repeat(1, t.shape[-1], 1)
    v_1 = v_1.unsqueeze(1).repeat(1, t.shape[-1], 1)
    t = t.unsqueeze(-1).repeat(1, 1, v_0.shape[-1])

    v_t = (1 - t) * v_0 + t * v_1
    return v_t


def nlerp(q_0: _Tensor, q_1: _Tensor, t: _Tensor) -> _Tensor:
    """
    args:
        q_0: shape: [n, 4]
        q_1: shape: [n, 4]
        t: shape: [n, m]

    rets:
        q_t: shape: [n, m, 4]
    """

    # 钝角检测, 解决双倍覆盖问题
    dot = torch.einsum("ij,ij->i", [q_0, q_1])
    indices = torch.where(dot < 0)[0]
    q_1[indices] *= -1

    q_t = lerp(q_0, q_1, t)

    q_t = normalize(q_t)  # TODO:注意可能会导致问题，有待验证
    return q_t


_DynamicsFuncType: TypeAlias = Union[
    Callable[
        [
            _Tensor | float,  # $0 时间
            Sequence[_Tensor | float],  # $1 状态
        ],
        Sequence[_Tensor],
    ],
    Callable,
]


def ode_euler(
    f: _DynamicsFuncType,
    t0: _Tensor | float,
    x0: Sequence[_Tensor],
    dt: _Tensor | float,
):
    """欧拉法"""
    w1 = f(t0, x0)
    x_next = [xi + dt * wi for xi, wi in zip(x0, w1)]
    return x_next


def ode_rk23(
    f: _DynamicsFuncType,
    t0: _Tensor | float,
    x0: Sequence[_Tensor],
    dt: _Tensor | float,
) -> list[_Tensor]:
    """改进欧拉法"""
    dt_hf = dt * 0.5
    w1 = f(t0, x0)
    w2 = f(t0 + dt, [xi + dt * wi for xi, wi in zip(x0, w1)])
    x_next = [xi + dt_hf * (wi1 + wi2) for xi, wi1, wi2 in zip(x0, w1, w2)]
    return x_next


def ode_rk45(
    f: _DynamicsFuncType,
    t0: _Tensor | float,
    x0: Sequence[_Tensor],
    dt: _Tensor | float,
) -> list[_Tensor]:
    """定步长 4阶 Runge-Kutta 法"""
    dt_hf = dt * 0.5
    t1 = t0 + dt_hf

    w1 = f(t0, x0)
    w2 = f(t1, [xi + dt_hf * wi for xi, wi in zip(x0, w1)])
    w3 = f(t1, [xi + dt_hf * wi for xi, wi in zip(x0, w2)])
    w4 = f(t0 + dt, [xi + dt * wi for xi, wi in zip(x0, w3)])

    dt_ = dt / 6.0
    x_next = [
        xi + dt_ * (wi1 + (wi2 + wi2) + (wi3 + wi3) + wi4)
        for xi, wi1, wi2, wi3, wi4 in zip(x0, w1, w2, w3, w4)
    ]
    return x_next


def modin(x: _Tensor, a: _Tensor | float, m: _Tensor | float):
    r"""
    a+((x-a) mod m)
    if m=0, return a
    if m>0, y $\in [a,a+m)$
    if m<0, y $\in (a-m,a]$
    """
    if not (isinstance(a, _Tensor) and isinstance(m, _Tensor)):
        _0 = torch.zeros_like(x)
        if not isinstance(a, _Tensor):
            a = _0 + a  # assert: 广播相容
        if not isinstance(m, _Tensor):
            m = _0 + m  # assert: 广播相容
    y = torch.where(m == 0, a, (x - a) % m + a)
    return y


def delta_reg(a: _Tensor, b: _Tensor, r: _Tensor | float = _PI):
    r"""
    计算 a-b 在 R=(-r,r] 上的最小幅度值

    $$
    \argmin_{ d\in R=(-r,r]: d=a-b (mod |R|) } |d|
    $$
    """
    # assert torch.all(r > 0), "r must be positive."
    diam = r + r
    d = modin(a - b, 0, diam)  # in [0,2r)
    d = torch.where(d <= r, d, d - diam)  # in (-r,r]
    return d


def delta_deg_reg(a: _Tensor, b: _Tensor):
    r"""
    $$
    \argmin_{ d\in R=(-180,180]: d=a-b (mod |R|) } |d|
    $$"""
    return delta_reg(a, b, 180)


def delta_rad_reg(a: _Tensor, b: _Tensor):
    r"""
    $$
    \argmin_{ d\in R=(-pi,pi]: d=a-b (mod |R|) } |d|
    $$"""
    return delta_reg(a, b, _PI)


# !!!@torch.jit.script
def _calc_zem(
    p1: _Tensor,
    v1: _Tensor,
    p2: _Tensor,
    v2: _Tensor,
    tmin: float = 0.0,
    tmax: float = math.inf,
) -> tuple[_Tensor, _Tensor]:
    assert len(p1.shape) == len(v1.shape) == len(p2.shape) == len(v2.shape) >= 2, (
        "p1,v1,p2,v2 must have the same dims>=2, got.",
        p1.shape,
        v1.shape,
        p2.shape,
        v2.shape,
    )
    p1 = p1.unsqueeze(-2)  # (...,n,1,d)
    v1 = v1.unsqueeze(-2)  # (...,n,1,d)
    p2 = p2.unsqueeze(-3)  # (...,1,m,d)
    v2 = v2.unsqueeze(-3)  # (...,1,m,d)
    # p1, v1, p2, v2 = _broadcast_arrays(p1, v1, p2, v2)  # (...,n,m,d)
    dp = p1 - p2  # (...,n,m,d)
    dv = v1 - v2  # (...,n,m,d)
    pv = (dp * dv).sum(-1, keepdim=True)  # (...,n,m,1)
    vv = (dv * dv).sum(-1, keepdim=True)  # (...,n,m,1)
    vvis0 = vv <= 1e-12
    # _0f = _zeros_like(pv)  # (...,n,m,1)
    t_miss = _where(vvis0, tmin, -pv / (vv + vvis0))  # (...,n,m,1)
    if not _bkbn.isfinite(t_miss).all():
        raise ValueError("t_miss contains NaN or inf.")
    t_miss = _clip(t_miss, tmin, tmax)  # 投影时间
    md = _norm(dp + dv * t_miss, dim=-1, keepdim=True)  # (...,n,m,1)
    return md, t_miss


def calc_zem(
    p1: _Tensor,
    v1: _Tensor,
    p2: _Tensor,
    v2: _Tensor,
    tmin: float = 0.0,
    tmax: float = math.inf,
) -> tuple[_Tensor, _Tensor]:
    r"""
    零控脱靶量

    $$
    \min_\{d(t)=\|(p_1+t v_1)-(p_2+t v_2)\|_2 | t\in T=[t_\min,t_\max]\}
    $$

    Args:
        p1: shape=(...,n,d) 群体1的初始位置(t=0)
        v1: shape=(...,n,d) 群体1的速度
        p2: shape=(...,m,d) 群体2的初始位置(t=0)
        v2: shape=(...,m,d) 群体2的速度
        tmin: 最小时间, 默认为0
        tmax: 最大时间, 默认为 \infty
    Returns:
        MD: 脱靶量 shape=(...,n,m,1)\
                $MD[i,j]:=min_{t\in T} d(t)$\

        t_miss: 脱靶时间 shape=(...,n,m,1)\
                $t_{miss}[i,j]:=\min\argmin_{t\in T} d(t)$\
    """
    return _calc_zem(p1, v1, p2, v2, tmin, tmax)


# !!!@torch.jit.script
def _los_is_visible_v1(
    los: _Tensor,  # (...,N,d)
    rball: _Tensor,  # (...,N,1)
    mask: _Tensor,  # (...,N,1)
):
    r"""
    $$
    visible_i:= m_i\land \bigwedge_{j\neq i}(m_j\to (I_1 p_i)\cap \bar B(p_j,r_j)=\emptyset)
    $$

    """
    n = los.shape[-2]
    _0f3 = _zeros_like(los)
    #
    dij = calc_zem(_0f3, los, los, _0f3, 0.0, 1.0)[
        0
    ]  # (...,N,N) |p_j-\Pi_{I_1 p_i}p_j|
    rj = rball.unsqueeze(-3)  # (...,1,N,1)
    fij = dij > rj  # (...,N,N) (I_1 p_i)\cap\bar B(p_j,r_j)=\emptyset
    #
    _I = torch.eye(n, device=los.device, dtype=torch.bool).view(
        [1] * len(los.shape[:-2]) + [n, n, 1]
    )
    mj = mask.unsqueeze(-3)  # (...,1,N,1)
    ifthen = _I | (~mj) | fij
    mi = mask.unsqueeze(-2)  # (...,N,1,1)
    vis = mi & ifthen  # (...,N,N,1)
    vi = vis.all(dim=-2)  # (...,N,1)
    #
    bij = (~_I) & mi & (~fij) & mj  # (...,N,N,1) i blocked by j
    b1i = bij.any(-2)  # (...,N,1)
    b2i = bij.any(-3)  # (...,N,1)
    fus = b1i & b2i  # (...,N,1)
    return vi, fus


def _los_is_visible_v2(
    los: _Tensor,  # (...,N,d)
    rball: _Tensor,  # (...,N,1)
    mask: _Tensor,  # (...,N,1)
):
    r"""
    $$
    visible_i:= m_i\land (
        0\in\bar B(p_i,r_i) \lor (
        0\notin\bar B(p_i,r_i)\land(
        \bigwedge_{j\neq i}... todo
            )
        )
    )
    $$
    """
    raise NotImplementedError("todo")


def los_is_visible(
    los: _Tensor, rball: _Tensor, mask: _Tensor | None = None
) -> tuple[_Tensor, ...]:
    r"""
    判断视线是否不被任何球遮挡
    Args:
        los (_Tensor): 视线组 shape=(...,N,d)
        rball (_Tensor): 球半径 shape=(...,N,1)
        mask (_Tensor | None, optional): 掩码,确认各位的信息是否可用, shape=(...,N,1);
            Defaults to None->默认全部可用.
    Returns:
        visible (_Tensor): 视线是否不受阻, shape=(...,N,1)
        fused (_Tensor): 混杂, shape=(...,N,1)
    """
    if mask is None:
        # mask = _ones(len(los.shape) * [1], dtype=torch.bool, device=los.device)
        mask = _ones_like(rball, dtype=torch.bool)
    assert len(los.shape) == len(rball.shape) == len(mask.shape) >= 2, (
        "los, radius, and mask must have the same dims>=2, got",
        len(los.shape),
        len(rball.shape),
        len(mask.shape),
    )
    assert rball.shape[-1] == 1, ("rball.shape[-1] must be 1, got", rball.shape[-1])
    assert mask.shape[-1] == 1, ("mask.shape[-1] must be 1, got", mask.shape[-1])
    n = los.shape[-2]
    assert rball.shape[-2] == 1 or n == rball.shape[-2], (
        "rball.shape[-2] must be 1 or",
        n,
        "bug got",
        rball.shape[-2],
    )
    assert mask.shape[-2] == 1 or n == mask.shape[-2], (
        "mask.shape[-2] must be 1 or",
        n,
        "bug got",
        mask.shape[-2],
    )
    return _los_is_visible_v1(los, rball, mask)


def _demo():  # 自测
    from timeit import timeit
    import time

    # init seed
    seed = int(time.time())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ntest = 100
    bsz = (64,)
    th_float = torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_rpy2mat = False
    test_rpy2quat = False
    test_quat_mul = False
    test_quat_rot = False
    test_zem = False
    test_block = True

    rpy_low = torch.reshape(
        torch.asarray(
            [-math.pi, -math.pi / 2, -math.pi], dtype=th_float, device=device
        ),
        [1] * len(bsz) + [3],
    )
    rpy_high = torch.reshape(
        torch.asarray([math.pi, math.pi / 2, math.pi], dtype=th_float, device=device),
        [1] * len(bsz) + [3],
    )

    def f():
        if test_block:
            n = 4 * 50
            d = 3
            los = torch.randn([*bsz, n, d], dtype=th_float, device=device)
            rball = torch.rand([*bsz, n, 1], dtype=th_float, device=device)
            rst = los_is_visible(los, rball)
        if test_zem:
            n = 50
            m = 50
            d = 3
            p1 = torch.randn([*bsz, n, d], dtype=th_float, device=device)
            v1 = torch.randn([*bsz, n, d], dtype=th_float, device=device)
            p2 = torch.randn([*bsz, m, d], dtype=th_float, device=device)
            v2 = torch.randn([*bsz, m, d], dtype=th_float, device=device)
            zem, tmiss = calc_zem(p1, v1, p2, v2)

        if test_rpy2mat:
            rpy = affcmb(
                torch.rand([*bsz, 3], dtype=th_float, device=device), rpy_low, rpy_high
            )
            roll = rpy[..., 0:1]
            Reb = rpy2mat(rpy)
            rpy2 = rpy2mat_inv(Reb, roll)
            err = (rpy2 - rpy).abs().max()
            assert err < 1e-5, "rpy2mat_inv error."

        if test_rpy2quat:
            rpy = affcmb(
                torch.rand([*bsz, 3], dtype=th_float, device=device), rpy_low, rpy_high
            )
            q = rpy2quat(rpy)
            roll = rpy[..., 0:1]
            rpy2 = rpy2quat_inv(q, roll)
            err = (rpy2 - rpy).abs().max()
            assert err < 1e-3, "rpy2quat_inv error."

        if test_quat_mul:
            p = quat_normalize(torch.rand([*bsz, 4], dtype=th_float, device=device))
            assert is_normalized(p), "Quaternion normalization error."
            q = quat_normalize(torch.rand([*bsz, 4], dtype=th_float, device=device))
            assert is_normalized(q), "Quaternion normalization error."
            r = quat_mul(p, q)
            assert is_normalized(r), "expect |p*q|==1 for |p|=|q|=1"

            q2 = q * 10
            q2inv = quat_inv(q2)
            p2 = quat_mul(quat_mul(p, q2), q2inv)
            rer = (p2 - p).abs().max()
            assert rer < 1e-6, "quat_mul error."

        if test_quat_rot:
            q = quat_normalize(torch.rand([*bsz, 4], dtype=th_float, device=device))
            u = normalize(
                torch.rand([*bsz, 3], dtype=th_float, device=device)
            )  # 单位球面测试
            v = quat_rotate(q, u)
            qconj = quat_conj(q)
            u2 = quat_rotate(qconj, v)
            rer = (u - u2).abs().max()
            assert rer < 1e-4, "Quaternion rotation error is too large."

        # v2 = quat_rotate_inverse(q, u)
        # Rq = quat2rotmat(q)
        # v = Rq @ v.unsqueeze(-1)
        # m = _quat2rotmat_sqrt(q)

    print("Testing...")
    t = timeit(f, number=ntest)
    print(t)
    # print(r, r.shape)
    # qvq = quat_rotate(q, v)
    # v2 = quat_rotate_inverse(q, qvq)
    # print(qvq, qvq.shape)


if __name__ == "__main__":
    _demo()
