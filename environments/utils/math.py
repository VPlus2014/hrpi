# 250601
from __future__ import annotations
from typing import TYPE_CHECKING, cast
import torch

_stack = torch.stack
_cat = torch.cat
_zeros_like = torch.zeros_like
_ones_like = torch.ones_like
_sin = torch.sin
_cos = torch.cos
_atan2 = torch.atan2
_asin = torch.asin
_where = torch.where


def affcmb(
    w: torch.Tensor,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
) -> torch.Tensor:
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


@torch.jit.script
def _normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clip(eps)


@torch.jit.script
def _quat_split_keepdim(q: torch.Tensor):
    q0, q1, q2, q3 = q.split([1, 1, 1, 1], dim=-1)
    return q0, q1, q2, q3


@torch.jit.script
def _quat_re(q: torch.Tensor):
    return q.split([1, 3], -1)[0]


@torch.jit.script
def _quat_im(q: torch.Tensor):
    return q.split([1, 3], -1)[1]


@torch.jit.script
def _quat_rect_re(q: torch.Tensor) -> torch.Tensor:
    """(解决双倍缠绕问题)返回实部非负的等价四元数"""
    reQ = _quat_re(q)
    return _where(reQ < 0, -q, q)


@torch.jit.script
def _an2quat(
    angle_rad: torch.Tensor,
    axis: torch.Tensor,
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
    angle_rad: torch.Tensor | float,
    axis: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = _zeros_like(axis[..., 0:1]) + angle_rad  # assert: 广播相容
    return _an2quat(angle_rad, axis)


# 定义基元旋转矩阵
def Lx(angle_rad: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)

    L_flat = (one, zero, zero, zero, cos, sin, zero, -sin, cos)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))


def Ly(angle_rad: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)

    L_flat = (cos, zero, -sin, zero, one, zero, sin, zero, cos)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))


def Lz(angle_rad: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)

    L_flat = (cos, sin, zero, -sin, cos, zero, zero, zero, one)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))


@torch.jit.script
def _Qx(angle_rad: torch.Tensor) -> torch.Tensor:
    """绕x轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cos = torch.cos(ah)
    sin = torch.sin(ah)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, sin, zero, zero)
    return torch.cat(Q_flat, -1)


@torch.jit.script
def _Qy(angle_rad: torch.Tensor) -> torch.Tensor:
    """绕y轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cos = torch.cos(ah)
    sin = torch.sin(ah)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, zero, sin, zero)
    return torch.cat(Q_flat, -1)


@torch.jit.script
def _Qz(angle_rad: torch.Tensor) -> torch.Tensor:
    """绕z轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cos = torch.cos(ah)
    sin = torch.sin(ah)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, zero, zero, sin)
    return torch.cat(Q_flat, -1)


def Qx(angle_rad: torch.Tensor) -> torch.Tensor:
    """绕x轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qx(angle_rad)


def Qy(angle_rad: torch.Tensor) -> torch.Tensor:
    """绕y轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qy(angle_rad)


def Qz(angle_rad: torch.Tensor) -> torch.Tensor:
    """绕z轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qz(angle_rad)


@torch.jit.script  # 15%
def _rpy2mat(rpy: torch.Tensor):
    psi, theta, phi = torch.split(rpy, [1, 1, 1], dim=-1)  # (...,1)
    newshape = psi.shape[:-1] + (3, 3)
    _1 = torch.ones_like(psi)
    _0 = torch.zeros_like(psi)
    c1 = torch.cos(phi)
    s1 = torch.sin(phi)
    c2 = torch.cos(theta)
    s2 = torch.sin(theta)
    c3 = torch.cos(psi)
    s3 = torch.sin(psi)
    rx = torch.cat([_1, _0, _0, _0, c1, -s1, _0, s1, c1], -1).reshape(
        newshape
    )  # (...,3,3)
    ry = torch.cat([c2, _0, s2, _0, _1, _0, -s2, _0, c2], -1).reshape(
        newshape
    )  # (...,3,3)
    rz = torch.cat([c3, -s3, _0, s3, c3, _0, _0, _0, _1], -1).reshape(
        newshape
    )  # (...,3,3)
    return rz @ ry @ rx  # (...,3,3)


@torch.jit.script  # 21%
def _rpy2mat_inv(Reb: torch.Tensor, roll_ref_rad: torch.Tensor):
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
def _quat_norm(q: torch.Tensor) -> torch.Tensor:
    return q.norm(p=2, dim=-1, keepdim=True)


@torch.jit.script
def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / _quat_norm(q).clamp(min=1e-9)


def quat_norm(q: torch.Tensor) -> torch.Tensor:
    """Computes the norm of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The norm of the quaternion. shape: (..., 1).
    """
    return _quat_norm(q)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Normalizes a quaternion to have unit length.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).
    Returns:
        The normalized quaternion in (w, x, y, z). shape: (..., 4).
    """
    return _quat_normalize(q)


def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Rets:
        Normalized tensor of shape (N, dims).
    """
    return _normalize(x)


def is_normalized(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return ((v.norm(p=2, dim=-1, keepdim=True) - 1).abs() <= eps).all()


@torch.jit.script
def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    q0 = _quat_re(q)
    qv = _quat_im(q)
    return _cat((q0, -qv), dim=-1)


@torch.jit.script
def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    return _quat_conj(q) / (_quat_re(q) ** 2).clamp(min=1e-12)


def quat_split_keepdim(q: torch.Tensor):
    """拆分为分量.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        Re(q): 实部, shape: (..., 1).
        Im(q): 虚部, shape: (..., 3).
    """
    return _quat_split_keepdim(q)


def quat_re(q: torch.Tensor) -> torch.Tensor:
    """Get the real part of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The real part of the quaternion. shape: (..., 1).
    """
    return _quat_re(q)


def quat_im(q: torch.Tensor) -> torch.Tensor:
    """Get the imaginary part of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The imaginary part of the quaternion. shape: (..., 3).
    """
    return _quat_im(q)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). shape: (..., 4).

    Rets:
        The conjugate quaternion in (w, x, y, z). shape: (..., 4).
    """
    return _quat_conj(q)


def quat2prodmat(q: torch.Tensor) -> torch.Tensor:
    r"""Computes the matrix of a quaternion.
    四元数左乘等效为矩阵乘法（线性算子作用）

    $$
    q \otimes p = M(q) p, \forall p
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


def mati(q: torch.Tensor) -> torch.Tensor:
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
def _quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # 旧版:仅支持 (B,4) 尺寸
    # r = torch.bmm(quat2prodmat(p), q.unsqueeze(-1)).squeeze(-1)
    # return r

    # 新版:支持任意尺寸&计算耗时减少 25%
    reP = quat_re(p)
    imP = quat_im(p)
    reQ = quat_re(q)
    imQ = quat_im(q)
    r = torch.cat(
        [
            reP * reQ - torch.sum(imP * imQ, -1, keepdim=True),
            reP * imQ + reQ * imP + torch.cross(imP, imQ, -1),
        ],
        -1,
    )
    return r


@torch.jit.script
def _rpy2quat(rpy_rad: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = torch.split(rpy_rad, [1, 1, 1], dim=-1)  # (...,1)
    r_hf = roll * 0.5  # (...,1)
    p_hf = pitch * 0.5
    y_hf = yaw * 0.5
    _0 = torch.zeros_like(r_hf)
    q1 = torch.cat([torch.cos(r_hf), _sin(r_hf), _0, _0], -1)
    q2 = torch.cat([torch.cos(p_hf), _0, _sin(p_hf), _0], -1)
    q3 = torch.cat([torch.cos(y_hf), _0, _0, _sin(y_hf)], -1)
    return _quat_mul(_quat_mul(q3, q2), q1)


def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
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


def _crossmat_kern(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)  # (...,1)
    _0 = torch.zeros_like(x)
    return _cat([_0, -z, y, z, _0, -x, -y, x, _0], -1).reshape(
        v.shape[:-1] + (3, 3)
    )  # (...,3,3)


def crossmat(v: torch.Tensor):
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
def _quat2mat(q: torch.Tensor) -> torch.Tensor:
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


def quat2mat(q: torch.Tensor) -> torch.Tensor:
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
def _quat_rot(q: torch.Tensor, v: torch.Tensor):
    # 旧版:仅支持 (B,4) 尺寸
    # rotation_matrix = quat2mat(q)
    # u = torch.bmm(rotation_matrix, v.unsqueeze(-1)).squeeze(-1)
    # return u

    # 新版:支持任意尺寸&计算耗时减少 13%
    # assert is_normalized(q), "Quaternion must be normalized."
    reQ, imQ = q.split([1, 3], -1)
    _2q0 = reQ + reQ
    _2qv = imQ + imQ
    u = (
        (_2q0 * reQ - 1.0) * v
        + _2qv * ((imQ * v).sum(-1, keepdim=True))
        + _2q0 * torch.cross(imQ, v, -1)
    )
    return u


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""Rotate a vector by a quaternion along the last dimension of q and v.

    $$
    Im(Q \otimes (0,v) \otimes Q^*)
    $$

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).
        v: The vector in (x, y, z). shape: (..., 3).

    Returns:
        The rotated vector in (x, y, z). shape: (..., 3).
    """
    return _quat_rot(q, v)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""Inverse rotate a vector by a quaternion along the last dimension of q and v.

    $$
    Im(Q^* \otimes (0,v) \otimes Q)
    $$

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).
        v: The vector in (x, y, z). shape: (..., 3).

    Returns:
        The rotated vector in (x, y, z). shape: (..., 3).
    """
    # rotation_matrix = quat2rotmat(quat_conjugate(q))
    # u = torch.bmm(rotation_matrix, v.unsqueeze(-1)).squeeze(-1)

    # assert is_normalized(q), "Quaternion must be normalized."
    return quat_rotate(quat_conjugate(q), v)


def quat_enu_ned() -> torch.Tensor:
    return quat_mul(
        Qz(torch.tensor([[torch.pi * 0.5]])),
        quat_mul(Qy(torch.tensor([[0]])), Qx(torch.tensor([[torch.pi]]))),
    )


def rpy2mat(
    rpy: torch.Tensor,
):
    r"""
    Z-Y-X 内旋矩阵
    $$
    R = R_z(\psi) R_y(\theta) R_x(\phi)
    $$
    Args:
        rpy (torch.Tensor): (滚转\psi,俯仰\theta,偏航\phi),unit:rad, shape (..., 3)
    Returns:
        torch.Tensor: 旋转矩阵, shape (..., 3, 3)
    """
    return _rpy2mat(rpy)


def rpy2mat_inv(Reb: torch.Tensor, roll_ref_rad: torch.Tensor | float = 0.0):
    """
    rpy2mat 的逆映射，当发生万向节死锁时，需要给出 roll_ref_rad
    Args:
        Reb (torch.Tensor): 旋转矩阵, shape (..., 3, 3)
        roll_ref_rad (torch.Tensor | float): 滚转角参考值, 单位: rad, shape (..., 1) 或标量
    """
    assert Reb.shape[-2:] == (3, 3), "expected matrix shape (...,3,3), got {}".format(
        Reb.shape
    )
    if not isinstance(roll_ref_rad, torch.Tensor):
        roll_ref_rad = roll_ref_rad + _zeros_like(Reb[..., 0, 0:1])  # assert: 广播相容
    return _rpy2mat_inv(Reb, roll_ref_rad)


def rpy2quat(rpy_rad: torch.Tensor) -> torch.Tensor:
    r"""
    输入欧拉角(rad)，输出规范四元数
    $$
    Q=Q_{e3,yaw}*Q_{e2,pitch}*Q_{e1,roll}
    $$
    Args:
        rpy_rad (torch.Tensor): 输入欧拉角(rad), shape: (..., 3)
    Returns:
        torch.Tensor: 规范四元数, shape: (..., 4)
    """
    return _rpy2quat(rpy_rad)


def rpy2quat_inv(
    q: torch.Tensor, roll_ref_rad: torch.Tensor | float = 0.0
) -> torch.Tensor:
    r"""四元数反解欧拉角
    $$
    q=Q_z(yaw)\otimes Q_y(pitch) \otimes Q_x(roll)
    $$

    Args:
        q (torch.Tensor): 规范四元数 shape: (..., 4).
        roll_ref_rad (torch.Tensor | float, optional): 当前滚转角(死锁时备用) shape: (..., 1) or scalar. Defaults to 0.0.

    Returns:
        torch.Tensor: 欧拉角(滚转\phi,俯仰\theta,偏航\psi),单位:rad, shape (..., 3)
    """
    Reb = _quat2mat(q)
    rpy = rpy2mat_inv(Reb, roll_ref_rad)
    return rpy


def euler_from_quat(
    q: torch.Tensor, roll_ref_rad: torch.Tensor | float = 0.0
) -> torch.Tensor:
    # q0, q1, q2, q3 = q.unbind(dim=-1)
    # # 隐患: 万向节死锁求解

    # phi = torch.arctan2(2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2))
    # theta = torch.arcsin(torch.clamp(-2 * (q3 * q1 - q0 * q2), -1, 1))
    # psi = torch.arctan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q2 * q2 + q3 * q3))

    # return torch.stack([phi, theta, psi], dim=-1)

    return rpy2quat_inv(q, roll_ref_rad)


def ned2aer(ned: torch.Tensor) -> torch.Tensor:
    n, e, d = torch.unbind(ned, dim=-1)  # (...,)
    rxy = torch.sqrt(torch.pow(e, 2) + torch.pow(n, 2))
    slant_range = torch.norm(ned, p=2, dim=-1)

    eps = 1e-6
    rxy_is_0 = rxy < eps
    az = torch.where(
        rxy_is_0,
        torch.zeros_like(n),
        torch.atan2(e, n),
    )
    elev = torch.where(
        rxy_is_0,
        torch.sign(d) * -(torch.pi * 0.5),
        torch.atan2(-d, rxy),
    )
    return torch.stack([az, elev, slant_range], dim=-1)


def aer2ned(aer: torch.Tensor) -> torch.Tensor:
    a, e, r = torch.unbind(aer, dim=-1)

    r_prime = r * torch.cos(e)
    n = r_prime * torch.cos(a)
    e = r_prime * torch.sin(a)
    d = -r * torch.sin(e)

    return torch.stack([n, e, d], dim=-1)


def _herp(
    position_0: torch.Tensor,
    velocity_0: torch.Tensor,
    position_1: torch.Tensor,
    velocity_1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
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
    position_0: torch.Tensor,
    velocity_0: torch.Tensor,
    position_1: torch.Tensor,
    velocity_1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
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


def lerp(v_0: torch.Tensor, v_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    v_0 = v_0.unsqueeze(1).repeat(1, t.shape[-1], 1)
    v_1 = v_1.unsqueeze(1).repeat(1, t.shape[-1], 1)
    t = t.unsqueeze(-1).repeat(1, 1, v_0.shape[-1])

    v_t = (1 - t) * v_0 + t * v_1
    return v_t


def nlerp(q_0: torch.Tensor, q_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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


def _demo():  # 自测
    from timeit import timeit

    # init seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    bsz = (8,)

    def f():
        rpy = affcmb(torch.rand([*bsz, 3], dtype=torch.float32), torch.pi, 2 * torch.pi)
        roll = rpy[..., 0:1]
        reb = rpy2mat(rpy)
        rpy2 = rpy2mat_inv(reb, roll)
        return
        p = quat_normalize(torch.rand([*bsz, 4], dtype=torch.float32))
        q = quat_normalize(torch.rand([*bsz, 4], dtype=torch.float32))
        r = quat_mul(p, q)
        return

        qconj = quat_conjugate(q)
        assert is_normalized(q), "Quaternion must be normalized."
        v = normalize(torch.rand([*bsz, 3], dtype=torch.float32))  # 单位球面测试

        u = quat_rotate(q, v)
        v2 = quat_rotate(qconj, u)
        # v2 = quat_rotate_inverse(q, u)
        rer = (v - v2).abs().max()
        # print("Max error:", err.item())
        assert rer < 1e-4, "Quaternion rotation error is too large."
        # Rq = quat2rotmat(q)
        # v = Rq @ v.unsqueeze(-1)
        # m = _quat2rotmat_sqrt(q)

    print("Testing...")
    t = timeit(f, number=10000)
    print(t)
    # print(r, r.shape)
    # qvq = quat_rotate(q, v)
    # v2 = quat_rotate_inverse(q, qvq)
    # print(qvq, qvq.shape)


if __name__ == "__main__":
    _demo()
