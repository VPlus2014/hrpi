# 张量运算辅助 250605
from __future__ import annotations
from typing import TYPE_CHECKING, cast
import math
from typing import Any, List, Tuple, TypeVar, Union

import numpy as _bkbn
from numpy.typing import NDArray as _NDArr
from numpy.linalg import norm

_floating = _bkbn.floating
_integer = _bkbn.integer
_float32 = _bkbn.float32
_float64 = _bkbn.float64
_InNum = Union[int, float, _floating, _integer]
_InArr = Union[List, Tuple, _NDArr[_floating], _NDArr[_integer]]
_InArrOrNum = Union[_InArr, _InNum]
_FloatNDArr = Union[_NDArr[_floating], _NDArr[_float32], _NDArr[_float64]]
_T = TypeVar("_T")

_cat = _bkbn.concatenate
_stack = _bkbn.stack
_asarray = _bkbn.asarray
_where = _bkbn.where
_arctan2 = _bkbn.arctan2
_arcsin = _bkbn.arcsin
_clip = _bkbn.clip
_cos = _bkbn.cos
_sin = _bkbn.sin
_abs = _bkbn.abs
_zeros_like = _bkbn.zeros_like
_ones_like = _bkbn.ones_like
_zeros = _bkbn.zeros
_ones = _bkbn.ones
_reshape = _bkbn.reshape
_broadcast_arrays = _bkbn.broadcast_arrays
_norm = norm


def _split_keepdim(v, axis=-1) -> List[_FloatNDArr]:
    return _bkbn.split(v, _bkbn.shape(v)[axis], axis)


_PI = math.pi
_2PI = math.tau
_PI_HALF = _PI * 0.5
_NPI = -_PI


def modin(x, a: _InArrOrNum = 0, m: _InArrOrNum = _2PI) -> _FloatNDArr:
    r"""
    a+((x-a) mod m)
    if m=0, return a
    if m>0, y $\in [a,a+m)$
    if m<0, y $\in (a-m,a]$
    """
    x = _asarray(x)
    y = _where(m == 0, a, (x - a) % m + a)
    return y


def modrad(x, a: _InArrOrNum = _NPI, m: _InArrOrNum = _2PI):
    return modin(x, a, m)


def moddeg(x, a: _InArrOrNum = -180, m: _InArrOrNum = 360):
    return modin(x, a, m)


def R3_wedge(v, axis=-1) -> _FloatNDArr:
    r"""$v_{\wedge} x = v\times x $"""
    v = _asarray(v)
    assert v.shape[axis] == 3, "expected dim[{}] to be 3, got {}".format(axis, v.shape)
    v1, v2, v3 = _split_keepdim(v, axis)  # (...,1)
    axis = axis % len(v.shape)
    _0 = _zeros_like(v1)
    v_wedge = _stack(
        [
            _cat([_0, -v3, v2], axis=axis),
            _cat([v3, _0, -v1], axis=axis),
            _cat([-v2, v1, _0], axis=axis),
        ],
        axis=axis,
    )
    return v_wedge


def delta_reg(a, b, r: _InArrOrNum = _PI) -> _FloatNDArr:
    r"""
    计算 a-b 在 mod R=(-r,r] 上的最小幅度值

    即 $\argmin_{ d\in R=(-r,r]: d=a-b (mod |R|)} |d|$
    """
    a = _asarray(a)
    r = _asarray(r)  # assert all(r > 0)
    diam = r + r
    d = modin(a - b, 0, diam)  # in [0,2r)
    d = _where(d <= r, d, d - diam)  # in (-r,r]
    return d


def delta_deg_reg(a, b):
    r"""$\argmin_{ d\in R=(-180,180]: d=a-b (mod |R|)} |d|$"""
    return delta_reg(a, b, 180)


def delta_rad_reg(a, b):
    r"""$$\argmin_{ d\in R=(-pi,pi]: d=a-b (mod |R|)} |d|$"""
    return delta_reg(a, b, _PI)


def rpy_reg(roll_rad, pitch_rad, yaw_rad):
    r"""
    计算 roll pitch yaw 在 $S=[0,2\pi)\times[-\pi/2,\pi/2]\times[0,2\pi)$ 上的最近等价元

    即 $\argmin_{(r,p,y)\in S: R(r,p,y)=R(roll,pitch,yaw)} \|(r,p,y)-(raw,pitch,yaw)\|_2$
    """
    r = modin(roll_rad, 0.0, _2PI)
    p = modin(pitch_rad, -_PI, _2PI)
    y = modin(yaw_rad, 0.0, _2PI)
    idxs = _abs(p) > _PI_HALF
    p: _FloatNDArr = _where(idxs, modin(_PI - p, -_PI, _2PI), p)
    r: _FloatNDArr = _where(idxs, modin(r + _PI, 0.0, _2PI), r)
    y: _FloatNDArr = _where(idxs, modin(y + _PI, 0.0, _2PI), y)
    return r, p, y


def rpy_NEDLight2Len(roll_rad, pitch_rad, yaw_rad):
    """将光线系的 NED 姿态 转为相机 amuzi 姿态， 从目标坐标系到两个坐标系的旋转顺序均为 ZYX"""
    r, p, y = rpy_reg(roll_rad, pitch_rad, yaw_rad)
    r_c = modrad(-r, -_PI)
    p_c = _PI_HALF - p  # 转为天顶角, in [0, pi]
    y_c = modrad(-y, -_PI)
    return r_c, p_c, y_c


def rpy_NEDLight2Len_inv(roll_rad: float, pitch_rad: float, yaw_rad: float):
    r_l = modrad(-roll_rad, -_PI)
    p_l = _PI_HALF - pitch_rad  # 转为俯仰角, in [-\pi/2, \pi/2]
    y_l = modrad(-yaw_rad, -_PI)
    return r_l, p_l, y_l


def T_NEDLight_Pic():
    r"""坐标旋转矩阵 T_{LP}, $\Phi_L$=NED 光线系, $\Phi_P$= 右-下-前 图像坐标系;
    $\xi_L = T_{LP} \xi_P$
    """
    return _asarray(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ],
    )


def rpy2mat(rpy, axis=-1) -> _FloatNDArr:
    r"""
    欧拉角(rad)转为旋转矩阵(\roll, \pitch, \yaw)-> R_{e_3,\yaw}*R_{e_2,\pitch}*R_{e_1,\roll}

    Args:
        rpy: 绕 x,y,z 轴旋转角度, 尺寸 (...,3)
        axis: roll, pitch, yaw 所在的轴位置
    Returns:
        旋转矩阵 R: (...,3,3)
    """
    rpy = _asarray(rpy)
    axis = axis % len(rpy.shape)
    roll, pitch, yaw = _split_keepdim(rpy, axis)  # (...,1)
    # assert roll.shape[axis] == 1, "expected dim[{}] to be 1, got {}".format(
    #     axis, roll.shape
    # )

    # assert roll.shape[-1] == 1, "expected last dim to be 1, got {}".format(roll.shape)
    _0 = _zeros_like(roll)
    _1 = _0 + 1.0
    c1 = _cos(roll)
    s1 = _sin(roll)
    c2 = _cos(pitch)
    s2 = _sin(pitch)
    c3 = _cos(yaw)
    s3 = _sin(yaw)
    r1 = _stack(
        [
            _cat([_1, _0, _0], axis=axis),
            _cat([_0, c1, -s1], axis=axis),
            _cat([_0, s1, c1], axis=axis),
        ],
        axis=axis,
    )
    r2 = _stack(
        [
            _cat([c2, _0, s2], axis=axis),
            _cat([_0, _1, _0], axis=axis),
            _cat([-s2, _0, c2], axis=axis),
        ],
        axis=axis,
    )
    r3 = _stack(
        [
            _cat([c3, -s3, _0], axis=axis),
            _cat([s3, c3, _0], axis=axis),
            _cat([_0, _0, _1], axis=axis),
        ],
        axis=axis,
    )
    r = r3 @ r2 @ r1
    return r


def rpy2mat_inv(Reb, roll_ref_rad: _InArrOrNum = 0.0, eps=1e-4):
    """rpy2mat 的逆映射，当发生万向节死锁时，需要给出 roll_ref_rad"""
    Reb = _asarray(Reb)
    assert Reb.shape[-2:] == (3, 3), "expected matrix shape (...,3,3), got {}".format(
        Reb.shape
    )

    s2 = _clip(-Reb[..., 2, 0], -1.0, 1.0)
    theta: _FloatNDArr = _arcsin(s2)
    c2s1 = Reb[..., 2, 1]  # R32
    c2c1 = Reb[..., 2, 2]  # R33
    c2s3 = Reb[..., 1, 0]  # R21
    c2c3 = Reb[..., 0, 0]  # R11

    gl = 1 - _abs(s2) < eps
    _0 = _zeros_like(theta)
    roll_ref_rad = _0 + roll_ref_rad
    s1 = _sin(roll_ref_rad)
    c1 = _cos(roll_ref_rad)
    roll_ref_rad = _arctan2(s1, c1)
    s2s1 = s1 * s2
    s2c1 = c1 * s2
    R22 = Reb[..., 1, 1]
    R12 = Reb[..., 0, 1]
    R23 = Reb[..., 1, 2]
    R13 = Reb[..., 0, 2]
    s3 = s2s1 * R22 - c1 * R12 + s2c1 * R23 + s1 * R13
    c3 = c1 * R22 + s2s1 * R12 - s1 * R23 + s2c1 * R13

    phi: _FloatNDArr = _where(gl, roll_ref_rad, _arctan2(c2s1, c2c1))
    psi: _FloatNDArr = _where(gl, _arctan2(s3, c3), _arctan2(c2s3, c2c3))
    return phi, theta, psi


def rot_demo():
    x = _asarray([1, 2, 3])
    print(x)
    for f1, f2 in [
        (ned2enu, enu2ned),
        (ned2nue, nue2ned),
        (enu2nue, nue2enu),
    ]:
        fx = f1(x)
        ix = f2(fx)
        print(x, fx, ix)

    rpy = [1, 2, 3]
    w = R3_wedge(rpy)
    q1 = rpy2quat(rpy)
    q2 = quat_prod(
        quat_from_vec(_asarray([0, 0, rpy[2]])),
        quat_prod(
            quat_from_vec(_asarray([0, rpy[1], 0])),
            quat_from_vec(_asarray([rpy[0], 0, 0])),
        ),
    )
    print(q1)
    print(q2)
    dr = 2e-1
    dp = 2e-1
    dy = 2e-1
    rs_deg = _bkbn.deg2rad(_bkbn.arange(-180, 180, dr))
    ps_deg = _bkbn.deg2rad(_bkbn.linspace(-90, 90, int(180 / dp)))
    ys_deg = _bkbn.deg2rad(_bkbn.arange(-180, 180, dy))
    batchsize = 4096 * 1

    err_tol = 1e-12
    err_eul_max = 0.0
    err_T_max = 0.0
    err_TQ_max = 0.0

    _e1 = _asarray([1, 0, 0], dtype=_float64)
    _e2 = _asarray([0, 1, 0], dtype=_float64)
    _e3 = _asarray([0, 0, 1], dtype=_float64)

    def _proc(rpy_s: list):
        nonlocal err_eul_max, err_T_max, err_TQ_max
        batchsize = len(rpy_s)
        shphead = [batchsize, 1]  # 张量运算测试
        rpy_t = _asarray(rpy_s).reshape(*shphead, 3)  # (N,1,3)
        rpy_s.clear()
        errmsgs: list[str] = []

        Teb = rpy2mat(rpy_t)
        rs_p, ps_p, ys_p = rpy2mat_inv(Teb, rpy_t[..., 0])

        rpy_p = _stack([rs_p, ps_p, ys_p], axis=-1)
        assert (
            rpy_p.shape == rpy_t.shape
        ), f"expected rpy shape {rpy_t.shape}, but got {rpy_p.shape}"

        # 欧拉角误差测试
        errs_eul = modrad(rpy_t - rpy_p, -_PI)
        errs_eul = norm(errs_eul, axis=-1)
        errs_eul_btch_ub = errs_eul.max()
        if errs_eul_btch_ub > err_eul_max:
            err_eul_max = errs_eul_btch_ub

        if errs_eul_btch_ub > err_tol:
            for i in range(len(rpy_t)):
                row = f"[{i}] rpy_t={rpy_t[i]} rpy_p={rpy_p[i]}"
                errmsgs.append(row)
            errmsgs.append(f"err={errs_eul_btch_ub:.6g}")

        Qeb = rpy2quat(rpy_t)

        # v1 = _bkbn.array([0, 0, 1], dtype=_float64)  # 单位向量
        # v2 = quat_rot(Qeb, v1)  # 旋转后的向量
        # v3 = quat_rot(quat_inv(Qeb), v2)  # 逆旋转后的向量
        # err_v = norm(v1 - v3, axis=-1)
        # err_v_btch_ub = err_v.max()
        # if err_v_btch_ub > err_tol:
        #     errmsgs.append(f"|QvQ^*-v|={err_v_btch_ub:.6g}")

        TebQ = quat2mat(Qeb)
        # 矩阵一致性测试
        errs_T = norm(Teb - TebQ, axis=(-2, -1))
        err_T_btch_ub = errs_T.max()
        if err_T_btch_ub > err_tol:
            errmsgs.append(f"|T-T_Q|={err_T_btch_ub:.6g}")
        if err_T_btch_ub > err_T_max:
            err_T_max = err_T_btch_ub

        # 向量旋转测试
        _1s = _bkbn.ones([*shphead, 1])
        e1 = _1s * _e1  # (*bshp,3)
        e2 = _1s * _e2
        e3 = _1s * _e3
        tst_vecs = [e1, e2, e3]
        assert tst_vecs[0].shape == rpy_t.shape
        tst_vecs4mat = [v.reshape(*v.shape, 1) for v in tst_vecs]  # (...,3,1)
        Ys_TebQ = _stack([TebQ @ v for v in tst_vecs4mat], axis=-3).squeeze(-1)
        Ys_Qeb = _stack([quat_rot(Qeb, v) for v in tst_vecs], axis=-2)
        errs_TQ = norm(Ys_Qeb - Ys_TebQ, axis=-1)
        errs_TQ_btch_ub = errs_TQ.max()
        if errs_TQ_btch_ub > err_tol:
            errmsgs.append(f"|T_Q x-Im(Q*(0,v)*Q^{-1})|={errs_TQ_btch_ub:.6g}")
        if errs_TQ_btch_ub > err_TQ_max:
            err_TQ_max = errs_TQ_btch_ub

        if len(errmsgs) > 0:
            print("\n", *errmsgs, sep="\n")

    import time

    _term_ = False

    def _is_stop():
        return _term_

    def _run_test():
        nonlocal _term_
        rpy_s = []
        N = len(rs_deg) * len(ps_deg) * len(ys_deg)
        n = 0
        k0 = 0
        print(f"{float(N):.3g} test cases")
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(rs_deg) * len(ps_deg) * len(ys_deg) // batchsize)
        except ImportError:
            tqdm = None

            t0 = time.time()
        from itertools import product

        for r_rad, p_rad, y_rad in product(rs_deg, ps_deg, ys_deg):
            if _is_stop():
                break
            rpy_s.append([r_rad, p_rad, y_rad])
            if len(rpy_s) >= batchsize:
                dn = len(rpy_s)
                _proc(rpy_s)
                if tqdm:
                    pbar.update()
                else:
                    n += dn
                    dt = time.time() - t0
                    k = int(dt / 0.5)
                    if k > k0:
                        k0 = k
                        _spf = dt / n
                        _eta = int((N - n) * _spf)
                        _eta_m, _eta_s = divmod(_eta, 60)
                        _eta_h, _eta_m = divmod(_eta_m, 60)
                        _eta_d, _eta_h = divmod(_eta_h, 24)

                        if _eta_d:
                            _eta = f"{_eta_d}d "
                        else:
                            _eta = ""
                        _eta += f"{_eta_h:02d}:{_eta_m:02d}:{_eta_s:02d}"
                        msg = "ETA={} {:.04g} fps".format(_eta, 1 / _spf)
                        msg = msg.ljust(80)
                        print("\r" + msg, end="")
        if tqdm:
            pbar.close()
        if len(rpy_s) > 0:
            _proc(rpy_s)
        _term_ = True

    from threading import Thread

    Thread(target=_run_test).start()
    while True:
        if _is_stop():
            break
        try:
            time.sleep(0.5)
        except KeyboardInterrupt:
            break
    _term_ = True

    print(f"err_eul_max={err_eul_max:.6g}")
    print(f"err_T_max={err_T_max:.6g}")
    print(f"err_TQ_max={err_TQ_max:.6g}")


def quat_prod(p, q, axis=-1) -> _FloatNDArr:
    p, q = _broadcast_arrays(p, q)
    p0, p1, p2, p3 = _split_keepdim(p, axis=axis)  # (...,1)
    q0, q1, q2, q3 = _split_keepdim(q, axis=axis)  # (...,1)
    r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    r1 = p0 * q1 + q0 * p1 + p2 * q3 - p3 * q2
    r2 = p0 * q2 + q0 * p2 + p3 * q1 - p1 * q3
    r3 = p0 * q3 + q0 * p3 + p1 * q2 - p2 * q1
    pq = _cat([r0, r1, r2, r3], axis=axis)  # (...,4)
    return pq  # type: ignore


def rpy2quat(rpy_rad, axis=-1):
    r"""
    输入欧拉角(rad)，输出规范四元数 Q=Q_{e3,yaw}*Q_{e2,pitch}*Q_{e1,roll}
    """
    rpy_rad = _asarray(rpy_rad)
    roll, pitch, yaw = _split_keepdim(rpy_rad, axis=axis)
    # assert roll.shape[-1] == 1, "expected last dim to be 1, got {}".format(roll.shape)
    r_hf = roll * 0.5  # (...,1)
    p_hf = pitch * 0.5
    y_hf = yaw * 0.5
    _0 = _zeros_like(r_hf)
    q1 = _cat([_cos(r_hf), _sin(r_hf), _0, _0], axis=axis)
    q2 = _cat([_cos(p_hf), _0, _sin(p_hf), _0], axis=axis)
    q3 = _cat([_cos(y_hf), _0, _0, _sin(y_hf)], axis=axis)
    return quat_prod(quat_prod(q3, q2), q1)


def quat2mat(q, axis=-1, normalize=True) -> _FloatNDArr:
    r"""
    R@v = Q*(0,v)*Q^{-1}
    """
    q_: _FloatNDArr = _asarray(q)
    if normalize:
        q_ = quat_normalize(q_, axis=axis)
    axis = axis % len(q_.shape)

    _old = False
    if _old:
        _coshf = quat_Re(q_)  # (...,1), \cos(\theta/2)
        _coshf = _coshf.reshape(*_coshf.shape, 1)  # (...,1,1)
        _2coshf = _coshf + _coshf  # 2*\cos(\theta/2)
        imQ = quat_Im(q_)  # (...,3), \sin(\theta/2) * n, \|n\|_2=1
        imQ_ = imQ.reshape(*imQ.shape, 1)  # (...,3,1)
        imQ_T = imQ_.swapaxes(-1, -2)  # (...,1,3)
        imQ_wedge = R3_wedge(imQ)
        I3 = _bkbn.eye(3, dtype=q_.dtype)
        I3 = I3.reshape(*([1] * (len(q_.shape) - 1)), 3, 3)
        m1 = (_2coshf * _coshf) * I3 - I3
        m2 = (2 * imQ_) @ imQ_T
        m3 = _2coshf * imQ_wedge
        r = m1 + m2 + m3
    else:
        q0, q1, q2, q3 = _split_keepdim(q_, axis=axis)  # (...,1)
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
        r = _stack(
            [
                _cat([A11, A12, A13], axis=axis),
                _cat([A21, A22, A23], axis=axis),
                _cat([A31, A32, A33], axis=axis),
            ],
            axis=axis,
        )  # (...,3,3)
    return r


def quat_Im(q) -> _FloatNDArr:
    q = _asarray(q)
    return q[..., 1:]


def quat_Re(q) -> _FloatNDArr:
    q = _asarray(q)
    return q[..., 0:1]


def quat_conj(q, axis=-1):
    q = _asarray(q)
    q0, q1, q2, q3 = _split_keepdim(q, axis=axis)
    return _cat([q0, -q1, -q2, -q3], axis=axis)


def quat_norm(q, axis=-1) -> _FloatNDArr:
    r"""$\|q\|$"""
    return norm(q, axis=axis, keepdims=True)


def quat_normalize(q, axis=-1) -> _FloatNDArr:
    return q / quat_norm(q, axis=axis)


def quat_inv(q, axis=-1) -> _FloatNDArr:
    return quat_conj(q, axis=axis) / quat_norm(q, axis=axis)


def quat_from_vec(v, axis=-1, eps=1e-12) -> _FloatNDArr:
    r"""3D vector $v=\theta*n$ to quaternion $(\cos(\theta/2), \sin(\theta/2) * n), \|n\|_2=1$"""
    v = _asarray(v)
    a = norm(v, axis=axis, keepdims=True)  # (...,1)
    _e1 = _bkbn.reshape([1.0, 0.0, 0.0], [1] * len(v.shape[:-1]) + [3])
    _z = a < eps
    n = _where(_z, _e1, v / (a + _z))  # (...,3)
    a = a * 0.5
    cosah = _cos(a)  # (...,1)
    sinah = _sin(a)  # (...,1)
    q = _cat([cosah, sinah * n], axis=axis)  # (...,4)
    return q


def quat_from_im(im) -> _FloatNDArr:
    r"""[0, im]"""
    im = _asarray(im)
    return _cat([_zeros_like(im[..., 0:1]), im], axis=-1)


def quat_rot(q, v, normalize=True) -> _FloatNDArr:
    r"""$Im(q*(0,v)*q^{-1})$"""
    q = _asarray(q)
    v = _asarray(v)
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    if normalize:
        q = quat_normalize(q)
    h = quat_from_im(v)
    h = quat_prod(q, h)
    h = quat_prod(h, quat_conj(q))
    v = quat_Im(h)
    return v


def vec_cosine(
    v1,
    v2,
    norm1: _NDArr | None = None,  # type: ignore
    norm2: _NDArr | None = None,  # type: ignore
):
    r"""计算两个向量的 余弦 与 夹角, 若含零向量则余弦为1,夹角取0

    v1,v2,norm1,norm2 必须广播兼容

    Args:
        v1: (...,dim)
        v2: (...,dim)
        norm1: (...,1), 若为None则在内部计算
        norm2: (...,1), 若为None则在内部计算
    Returns:
        cosa: (...,1), 余弦值 $c[i]:=cos\angle(v1[i,:],v2[i,:])$
    """
    v1 = _asarray(v1)
    v2 = _asarray(v2)
    assert len(v1.shape) == len(v2.shape), "v1 and v2 are not broadcastable"
    if norm1 is None:
        norm1: _FloatNDArr = norm(v1, axis=-1, keepdims=True)  # (...,1)
    if norm2 is None:
        norm2: _FloatNDArr = norm(v2, axis=-1, keepdims=True)  # (...,1)
    use_broadcast = False  # 不广播更快
    if use_broadcast:
        v1, v2, norm1, norm2 = _broadcast_arrays(v1, v2, norm1, norm2)
    _dot = _bkbn.sum(v1 * v2, axis=-1, keepdims=True)  # (...,1)
    _any0 = _bkbn.isclose(norm1, 0) | _bkbn.isclose(norm2, 0)
    _cosa = _where(_any0, 1, _dot / ((norm1 * norm2) + _any0))  # (..., N1,N2,1)
    _cosa: _FloatNDArr = _clip(_cosa, -1, 1)  # (..., N1,N2,1)
    return _cosa


def ned2enu(xyz, axis=-1) -> _FloatNDArr:
    xyz = _asarray(xyz)
    n, e, d = _split_keepdim(xyz, axis)
    xyz = _cat([e, n, -d], axis)
    return xyz


def enu2ned(xyz, axis=-1) -> _FloatNDArr:
    xyz = _asarray(xyz)
    e, n, u = _split_keepdim(xyz, axis)
    xyz = _cat([n, e, -u], axis)
    return xyz


def nue2ned(xyz, axis=-1) -> _FloatNDArr:
    xyz = _asarray(xyz)
    n, u, e = _split_keepdim(xyz, axis)
    xyz = _cat([n, e, -u], axis)
    return xyz


def ned2nue(xyz, axis=-1) -> _FloatNDArr:
    xyz = _asarray(xyz)
    n, e, d = _split_keepdim(xyz, axis)
    xyz = _cat([n, -d, e], axis)
    return xyz


def nue2enu(xyz, axis=-1) -> _FloatNDArr:
    xyz = _asarray(xyz)
    n, u, e = _split_keepdim(xyz, axis)
    xyz = _cat([e, n, u], axis)
    return xyz


def enu2nue(xyz, axis=-1) -> _FloatNDArr:
    xyz = _asarray(xyz)
    e, n, u = _split_keepdim(xyz, axis)
    xyz = _cat([n, u, e], axis)
    return xyz


def affcmb(alpha, a, b):
    r"""
    $(1-\alpha) * a + \alpha * b$
    """
    a = _asarray(a)
    return a + alpha * (b - a)


def affcmb_inv(y, a, b):
    r"""
    get \alpha such that $y = (1-\alpha) * a + \alpha * b$
    """
    m = b - a
    y_ = y - a
    eps = 1e-12
    _mis0 = (m < eps) & (m > -eps)
    _mis0 = _mis0 + 0.0
    # assert _mis0.any() == False
    # w = _where(_mis0, 0, y_ / (m + _mis0))
    w = (y_ * (1 - _mis0)) / (m + _mis0)
    return w


def calc_zem(
    p1: _NDArr,
    v1: _NDArr,
    p2: _NDArr,
    v2: _NDArr,
    tmin: float = 0,
    tmax: float = math.inf,
) -> tuple[_NDArr, _NDArr]:
    r"""
    零控脱靶量

    $$
    \min_\{d(t)=\|(p_1+t v_1)-(p_2+t v_2)\|_2 | t\in T=[t_\min,t_\max]\}
    $$

    Args:
        p1: shape=(...,n|1,d) 群体1的初始位置(t=0)
        v1: shape=(...,n|1,d) 群体1的速度
        p2: shape=(...,m|1,d) 群体2的初始位置(t=0)
        v2: shape=(...,m|1,d) 群体2的速度
        tmin: 最小时间, 默认为0
        tmax: 最大时间, 默认为\infty$
    Returns:
        MD: 脱靶量 shape=(...,n,m,1)\
                $MD[i,j]:=min_{t\in T} d(t)$\

        t_miss: 脱靶时间 shape=(...,n,m,1)\
                $t_{miss}[i,j]:=\min\argmin_{t\in T} d(t)$\
    """
    assert len(p1.shape) == len(v1.shape) == len(p2.shape) == len(v2.shape), (
        "p1,v1,p2,v2 must have the same broadcastable shape[:-2].",
        p1.shape,
        v1.shape,
        p2.shape,
        v2.shape,
    )
    p1, v1 = _broadcast_arrays(p1, v1)  # (...,n,d)
    n = p1.shape[-2]
    p2, v2 = _broadcast_arrays(p2, v2)  # (...,m,d)
    assert p1.shape[-1] == p2.shape[-1], (
        "p1&v1 and p2&v2 must have the same dim[-1]",
        p1.shape,
        p2.shape,
    )
    m = p2.shape[-2]
    p1 = _bkbn.expand_dims(p1, -2)  # (...,n,1,d|1)
    v1 = _bkbn.expand_dims(v1, -2)  # (...,n,1,d|1)
    p2 = _bkbn.expand_dims(p2, -3)  # (...,1,m,d|1)
    v2 = _bkbn.expand_dims(v2, -3)  # (...,1,m,d|1)
    dp = p1 - p2  # (...,n,m,d)
    dv = v1 - v2  # (...,n,m,d)
    pv = (dp * dv).sum(-1, keepdims=True)  # (...,n,m,1)
    vv = (dv * dv).sum(-1, keepdims=True)  # (...,n,m,1)
    _zeroV = vv <= 1e-6  # 过零处理
    _0f = _zeros_like(pv)  # (...,n,m,1)
    t_miss = _where(_zeroV, _0f, -pv / (vv + _zeroV))  # (...,n,m,1)
    if not _bkbn.isfinite(t_miss).all():
        t_miss
        idxs = _bkbn.where(~(_bkbn.isfinite(t_miss)))
    t_miss = _clip(t_miss, tmin, tmax)  # 投影时间
    md = _norm(dp + dv * t_miss, axis=-1, keepdims=True)  # (...,n,m,1)
    return md, t_miss


if __name__ == "__main__":
    rot_demo()
    pass
