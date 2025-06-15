from __future__ import annotations
from typing import cast
from ._head import *


def _quat_chunk(q: ndarray):
    return split_(q, 4, -1)
    # q0 = q[..., 0:1]
    # q1 = q[..., 1:2]
    # q2 = q[..., 2:3]
    # q3 = q[..., 3:4]
    # return q0, q1, q2, q3


def _quat_re(q: ndarray):
    return q[..., 0:1]  # [...,0:1] 写法不能jit


def _quat_im(q: ndarray):
    return q[..., 1:4]  # # [...,1:4] 写法不能jit


def quat_rect_re(q: ndarray):
    r"""
    (解决双倍缠绕问题)返回实部非负的等价四元数
    $$
    q\mapsto p: Re(p)>=0\land q*h*q^* = p*h*p^*\forall h\in\mathbb{H}
    $$
    """
    reQ = _quat_re(q)
    q = where(reQ < 0, -q, q)
    return q


def _quat_norm(q: ndarray) -> ndarray:
    return norm(q, 2, -1, True)


def quat_from_im(im: ndarray):
    r"""[0, im]"""
    # im = asarray(im)
    return cat([zeros(im.shape[:-1] + (1,), im.dtype), im], axis=-1)


def _quat_normalize(q: ndarray) -> ndarray:
    return q / clip(_quat_norm(q), 1e-9, None)


def quat_norm(q: ndarray) -> ndarray:
    """Computes the norm of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The norm of the quaternion. shape: (..., 1).
    """
    return _quat_norm(q)


def quat_normalize(q: ndarray) -> ndarray:
    """Normalizes a quaternion to have unit length.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).
    Returns:
        The normalized quaternion in (w, x, y, z). shape: (..., 4).
    """
    return _quat_normalize(q)


def _an2quat(
    angle_rad: ndarray,
    axis: ndarray,
):
    assert angle_rad.ndim == axis.ndim, "Angle and axis must have the same ndim"
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    assert axis.shape[-1] == 3, "Axis must be a 3D vector."
    axis = normalize(axis)
    _0 = zeros(axis.shape[:-1] + (1,), axis.dtype)

    half_angle = angle_rad * 0.5
    q0 = cos(half_angle) + _0
    qv = sin(half_angle) * axis
    q = cat([q0, qv], axis=-1)  # (...,4)
    q = quat_rect_re(q)  # 确保实部非负
    return q


def an2quat(
    angle_rad: NDArrOrNum,
    axis: ndarray,
) -> ndarray:
    r"""
    绕轴右手旋转 $(\cos(a/2),\sin(a/2) x^0)$

    Args:
        angle_rad (NDArrOrNum): 旋转角度 scalar or shape: (...,1)
        axis (NDArr): 转轴向量(允许零向量or非单位化向量) shape: (...,3)

    Returns:
        NDArr: _description_
    """
    if not isinstance(angle_rad, ndarray):
        angle_rad = (
            zeros_like([1] * axis.ndim, axis.dtype) + angle_rad
        )  # assert: 广播相容
    return _an2quat(angle_rad, axis)


def _quat_conj(q: ndarray) -> ndarray:
    q0 = quat_re(q)
    qv = quat_im(q)
    return cat((q0, -qv), axis=-1)


def _quat_inv(q: ndarray) -> ndarray:
    return _quat_conj(q) / clip(pow(q, 2).sum(-1, keepdims=True), 1e-16, None)


def quat_inv(q: ndarray) -> ndarray:
    """Computes the inverse of a quaternion."""
    return _quat_inv(q)


def quat_split2(q: ndarray):
    """拆分为 Re(q), Im(q).

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        Re(q): 实部, shape: (..., 1).
        Im(q): 虚部, shape: (..., 3).
    """
    return _quat_re(q), _quat_im(q)


def quat_chunk(q: ndarray):
    """拆分为分量.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        w: 实部, shape: (..., 1).
        x: 虚部x, shape: (..., 1).
        y: 虚部y, shape: (..., 1).
        z: 虚部z, shape: (..., 1).
    """
    return _quat_chunk(q)


def quat_re(q: ndarray) -> ndarray:
    """Get the real part of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The real part of the quaternion. shape: (..., 1).
    """
    return _quat_re(q)


def quat_im(q: ndarray) -> ndarray:
    """Get the imaginary part of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Returns:
        The imaginary part of the quaternion. shape: (..., 3).
    """
    return _quat_im(q)


def quat_conj(q: ndarray) -> ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). shape: (..., 4).

    Rets:
        The conjugate quaternion in (w, x, y, z). shape: (..., 4).
    """
    return _quat_conj(q)


def quat2prodmat(q: ndarray) -> ndarray:
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
    q0, q1, q2, q3 = quat_chunk(q)  # (...,1)
    nq1 = -q1
    nq2 = -q2
    nq3 = -q3
    mat_flat = (q0, nq1, nq2, nq3, q1, q0, nq3, q2, q2, q3, q0, nq1, q3, nq2, q1, q0)
    return cat(mat_flat, axis=-1).reshape(q0.shape + (4, 4))


def _Qx(angle_rad: ndarray):
    """绕x轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cosa = cos(ah)
    sina = sin(ah)
    _0 = zeros_like(cosa)

    Q_flat = (cosa, sina, _0, _0)
    return cat(Q_flat, axis=-1)


def _Qy(angle_rad: ndarray):
    """绕y轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cosa = cos(ah)
    sina = sin(ah)
    _0 = zeros_like(cos)

    Q_flat = (cosa, _0, sina, _0)
    return cat(Q_flat, axis=-1)


def _Qz(angle_rad: ndarray):
    """绕z轴旋转的规范四元数"""
    ah = angle_rad * 0.5
    cosa = cos(ah)
    sina = sin(ah)
    _0 = zeros_like(cos)

    Q_flat = (cosa, _0, _0, sina)
    return cat(Q_flat, axis=-1)


def Qx(angle_rad: ndarray):
    """绕x轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qx(angle_rad)


def Qy(angle_rad: ndarray) -> ndarray:
    """绕y轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qy(angle_rad)


def Qz(angle_rad: ndarray) -> ndarray:
    """绕z轴旋转的规范四元数"""
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    return _Qz(angle_rad)


def mati(q: ndarray) -> ndarray:
    """Computes the matrix of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). shape: (..., 4).

    Rets:
        The matrix inverse of quaternion. shape: (..., 4, 4).
    """
    q0, q1, q2, q3 = quat_chunk(q)  # (...,1)

    mat_flat = (q0, -q1, -q2, -q3, q1, q0, q3, -q2, q2, -q3, q0, q1, q3, q2, -q1, q0)
    return cat(mat_flat, axis=-1).reshape(q0.shape + (4, 4))


def _quat2mat_sqrt(q: ndarray) -> ndarray:
    q0, q1, q2, q3 = quat_chunk(q)  # (...,1)
    nq0 = -q0
    nq1 = -q1
    nq2 = -q2
    nq3 = -q3
    m = stack(
        [
            cat([q0, nq1, nq2, nq3], axis=-1),
            cat([nq1, nq0, q3, nq2], axis=-1),
            cat([nq2, nq3, nq0, q1], axis=-1),
            cat([nq3, q2, nq1, nq0], axis=-1),
        ],
        -2,
    )
    return m


def quat2mat_sqrt(q: ndarray) -> ndarray:
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


def _quat_mul(p: ndarray, q: ndarray) -> ndarray:
    reP, imP = quat_split2(p)
    reQ, imQ = quat_split2(q)
    pq = cat(
        [
            reP * reQ - (imP * imQ).sum(-1, keepdims=True),
            reP * imQ + reQ * imP + cross(imP, imQ, axis=-1),
        ],
        axis=-1,
    )
    # 暴力版(慢)
    # p0, p1, p2, p3 = _chunk(p, 4, axis=-1)  # (...,1)
    # q0, q1, q2, q3 = _chunk(q, 4, axis=-1)  # (...,1)
    # r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    # r1 = p0 * q1 + q0 * p1 + p2 * q3 - p3 * q2
    # r2 = p0 * q2 + q0 * p2 + p3 * q1 - p1 * q3
    # r3 = p0 * q3 + q0 * p3 + p1 * q2 - p2 * q1
    # pq = cat([r0, r1, r2, r3], axis=-1)  # (...,4)
    return pq


def _rpy2quat(rpy_rad: ndarray) -> ndarray:
    assert rpy_rad.shape[-1] == 3, "RPY must be 3D in last dimension."
    r_hf, p_hf, y_hf = chunk(rpy_rad * 0.5, 3, axis=-1)  # (...,1)
    _0 = zeros_like(r_hf)
    q1 = cat([cos(r_hf), sin(r_hf), _0, _0], axis=-1)
    q2 = cat([cos(p_hf), _0, sin(p_hf), _0], axis=-1)
    q3 = cat([cos(y_hf), _0, _0, sin(y_hf)], axis=-1)
    return _quat_mul(_quat_mul(q3, q2), q1)


def quat_mul(p: ndarray, q: ndarray) -> ndarray:
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


def quat_rotate(q: ndarray, v: ndarray) -> ndarray:
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


def quat_rotate_inv(q: ndarray, v: ndarray) -> ndarray:
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
    # u = bmm(rotation_matrix, v.unsqueeze(-1)).squeeze(-1)

    # assert is_normalized(q), "Quaternion must be normalized."
    return quat_rotate(quat_conj(q), v)


def _crossmat(v: ndarray) -> ndarray:
    assert v.shape[-1] == 3, "Vector must be 3D in last dimension."
    x, y, z = chunk(v, 3, axis=-1)  # (...,1)
    _0 = zeros_like(x)
    return cat([_0, -z, y, z, _0, -x, -y, x, _0], axis=-1).reshape(
        v.shape[:-1] + (3, 3)
    )  # (...,3,3)


def crossmat(v: ndarray):
    r"""
    左叉积矩阵
    $v_\times a= v \times a$
    Args:
        v: The vector in (x, y, z). shape: (..., 3).
    Returns:
        The cross product matrix of vector v. shape: (..., 3, 3).
    """
    return _crossmat(v)


def _quat2mat(q: ndarray) -> ndarray:
    # assert is_normalized(q), "Quaternion must be normalized."
    q0, q1, q2, q3 = quat_chunk(q)  # (...,1)
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
    m = stack(
        [
            cat([A11, A12, A13], axis=-1),
            cat([A21, A22, A23], axis=-1),
            cat([A31, A32, A33], axis=-1),
        ],
        -2,
    )  # (...,3,3)
    return m


def quat2mat(q: ndarray) -> ndarray:
    r"""
    四元数->3D旋转矩阵.
    $$
    R(q) Im(h) = |q| Im(q \otimes h \otimes q^*)\\
    R(q)\in SO(3) \iff |q|=1
    $$

    Args:
        q: The quaternion in (w, x, y, z). shape: (..., 4).

    Rets:
        The rotation matrix of quaternion. shape: (..., 3, 3).
    """
    return _quat2mat(q)


def _quat_rot(q: ndarray, v: ndarray):
    reQ, imQ = quat_split2(q)
    _2q0 = reQ + reQ
    _2qv = imQ + imQ
    u = (
        (_2q0 * reQ - 1.0) * v
        + _2qv * sum_(imQ * v, -1, keepdims=True)
        + _2q0 * cross(imQ, v, axis=-1)
    )
    return u


# 定义基元旋转矩阵
def Lx(angle_rad: ndarray) -> ndarray:
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    c = cos(angle_rad)
    s = sin(angle_rad)
    _1 = ones_like(c)
    _0 = zeros_like(c)

    L_flat = (_1, _0, _0, _0, c, -s, _0, s, c)
    return cat(L_flat, axis=-1).reshape(angle_rad.shape[:-1] + (3, 3))


def Ly(angle_rad: ndarray) -> ndarray:
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    c = cos(angle_rad)
    s = sin(angle_rad)
    _1 = ones_like(c)
    _0 = zeros_like(c)

    L_flat = (c, _0, s, _0, _1, _0, -s, _0, c)
    return cat(L_flat, axis=-1).reshape(angle_rad.shape[:-1] + (3, 3))


def Lz(angle_rad: ndarray) -> ndarray:
    assert angle_rad.shape[-1] == 1, "Angle's last dimension must be 1."
    c = cos(angle_rad)
    s = sin(angle_rad)
    _1 = ones_like(c)
    _0 = zeros_like(c)
    L_flat = (c, -s, _0, s, c, _0, _0, _0, _1)
    return cat(L_flat, axis=-1).reshape(angle_rad.shape[:-1] + (3, 3))


def _rpy2mat(rpy: ndarray):
    # r = rpy[..., 0:1]  # (...,1)
    # p = rpy[..., 1:2]
    # y = rpy[..., 2:3]
    r, p, y = split_(rpy, 3, axis=-1)
    newshape = y.shape[:-1] + (3, 3)
    _1 = ones_like(y)
    _0 = zeros_like(y)
    c1 = cos(r)
    s1 = sin(r)
    c2 = cos(p)
    s2 = sin(p)
    c3 = cos(y)
    s3 = sin(y)
    r1 = cat([_1, _0, _0, _0, c1, -s1, _0, s1, c1], axis=-1).reshape(
        newshape
    )  # (...,3,3)
    r2 = cat([c2, _0, s2, _0, _1, _0, -s2, _0, c2], axis=-1).reshape(
        newshape
    )  # (...,3,3)
    r3 = cat([c3, -s3, _0, s3, c3, _0, _0, _0, _1], axis=-1).reshape(
        newshape
    )  # (...,3,3)
    return r3 @ r2 @ r1  # (...,3,3)


def _rpy2mat_inv(Reb: ndarray, roll_ref_rad: ndarray):
    assert (
        roll_ref_rad.shape[-1] == 1
    ), "Roll reference angle's last dimension must be 1."
    assert len(Reb.shape[:-2]) == len(
        roll_ref_rad.shape[:-1]
    ), "expect len(Reb.shape[:-2]) == len(roll_ref_rad.shape[:-1])"
    s2 = clip(-Reb[..., 2, 0], -1.0, 1.0)
    pitch = asin(s2)
    c2s1 = Reb[..., 2, 1]  # R32
    c2c1 = Reb[..., 2, 2]  # R33
    c2s3 = Reb[..., 1, 0]  # R21
    c2c3 = Reb[..., 0, 0]  # R11

    eps = 1e-4  # 万向节死锁阈值(jit 警告: 别放在参数表里)
    gl = 1 - abs_(s2) < eps
    #
    roll_ref_rad = roll_ref_rad.squeeze(-1)  # (...,)
    s1 = sin(roll_ref_rad)
    c1 = cos(roll_ref_rad)
    roll_ref_rad = atan2(s1, c1)  # -> (-pi,pi]
    #
    s2s1 = s1 * s2
    s2c1 = c1 * s2
    R22 = Reb[..., 1, 1]
    R12 = Reb[..., 0, 1]
    R23 = Reb[..., 1, 2]
    R13 = Reb[..., 0, 2]
    s3 = s2s1 * R22 - c1 * R12 + s2c1 * R23 + s1 * R13
    c3 = c1 * R22 + s2s1 * R12 - s1 * R23 + s2c1 * R13

    roll = where(gl, roll_ref_rad, atan2(c2s1, c2c1))
    yaw = where(gl, atan2(s3, c3), atan2(c2s3, c2c3))

    rpy = stack([roll, pitch, yaw], axis=-1)
    return rpy


def rpy2mat(
    rpy: ndarray,
) -> ndarray:
    r"""
    Z-Y-X 内旋矩阵
    $$
    R = R_z(\psi) R_y(\theta) R_x(\phi)
    $$
    Args:
        rpy (_NDArr): (滚转\psi,俯仰\theta,偏航\phi),unit:rad, shape=(..., 3)
    Returns:
        _NDArr: 旋转矩阵, shape=(..., 3, 3)
    """
    return _rpy2mat(rpy)


def rpy2mat_inv(Reb: ndarray, roll_ref_rad: ndarray | float = 0.0) -> ndarray:
    """
    rpy2mat 的逆映射，当发生万向节死锁时，需要给出 roll_ref_rad
    Args:
        Reb (_NDArr): 旋转矩阵, shape=(..., 3, 3)
        roll_ref_rad (_NDArr | float): 滚转角参考值, 单位: rad, shape=(..., 1) 或标量
    """
    assert Reb.shape[-2:] == (3, 3), "expected matrix shape=(...,3,3), got {}".format(
        Reb.shape
    )
    if not isinstance(roll_ref_rad, ndarray):
        _0 = zeros([1] * (len(Reb.shape) - 1))  # (...,1)
        roll_ref_rad = roll_ref_rad + _0  # assert: 广播相容
    return _rpy2mat_inv(Reb, roll_ref_rad)


def rpy2quat(rpy_rad: ndarray) -> ndarray:
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


def rpy2quat_inv(q: ndarray, roll_ref_rad: ndarray | float = 0.0) -> ndarray:
    r"""四元数反解欧拉角
    $$
    q=Q_z(yaw)\otimes Q_y(pitch) \otimes Q_x(roll)
    $$

    Args:
        q (_NDArr): 规范四元数 shape: (..., 4).
        roll_ref_rad (_NDArr | float, optional): 当前滚转角(死锁时备用) shape: (..., 1) or scalar. Defaults to 0.0.

    Returns:
        _NDArr: 欧拉角(滚转\phi,俯仰\theta,偏航\psi),单位:rad, shape=(..., 3)
    """
    Reb = _quat2mat(q)
    rpy = rpy2mat_inv(Reb, roll_ref_rad)
    return rpy


def euler_from_quat(q: ndarray, roll_ref_rad: ndarray | float = 0.0) -> ndarray:
    # q0, q1, q2, q3 = q.unbind(dim=-1)
    # # 隐患: 万向节死锁求解

    # phi = np.arctan2(2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2))
    # theta = np.arcsin(np.clamp(-2 * (q3 * q1 - q0 * q2), -1, 1))
    # psi = np.arctan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q2 * q2 + q3 * q3))

    # return np.stack([phi, theta, psi], axis=-1)

    return rpy2quat_inv(q, roll_ref_rad)
