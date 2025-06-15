from pathlib import Path
import sys


from ._head import *
from ._head import bkbn as _bkbn
from .rotation import *
from .coords import *
from .interpolation import *
from .rotation import quat_mul as quat_prod, quat_from_im as quat_from_vec
from .geo4ac import *


def rot_demo():
    x = asarray([1, 2, 3])
    print(x)
    for f1, f2 in [
        (ned2enu, enu2ned),
        (ned2nue, nue2ned),
        (enu2nue, nue2enu),
    ]:
        fx = f1(x)
        ix = f2(fx)
        print(x, fx, ix)

    rpy = bkbn.asarray([1, 2, 3])
    w = crossmat(rpy)
    q1 = rpy2quat(rpy)
    q2 = quat_prod(
        quat_from_vec(asarray([0, 0, rpy[2]])),
        quat_prod(
            quat_from_vec(asarray([0, rpy[1], 0])),
            quat_from_vec(asarray([rpy[0], 0, 0])),
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

    _float64 = bkbn.float64

    _e1 = asarray([1, 0, 0], dtype=_float64)
    _e2 = asarray([0, 1, 0], dtype=_float64)
    _e3 = asarray([0, 0, 1], dtype=_float64)

    def _proc(rpy_s: list):
        nonlocal err_eul_max, err_T_max, err_TQ_max
        batchsize = len(rpy_s)
        shphead = [batchsize, 1]  # 张量运算测试
        rpy_t = asarray(rpy_s).reshape(*shphead, 3)  # (N,1,3)
        rpy_s.clear()
        errmsgs: list[str] = []

        Teb = rpy2mat(rpy_t)
        rs_p, ps_p, ys_p = rpy2mat_inv(Teb, rpy_t[..., 0])

        rpy_p = stack([rs_p, ps_p, ys_p], axis=-1)
        assert (
            rpy_p.shape == rpy_t.shape
        ), f"expected rpy shape {rpy_t.shape}, but got {rpy_p.shape}"

        # 欧拉角误差测试
        errs_eul = modrad(rpy_t - rpy_p, -PI)
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
        errs_T = norm(Teb - TebQ)
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
        Ys_TebQ = stack([TebQ @ v for v in tst_vecs4mat], axis=-3).squeeze(-1)
        Ys_Qeb = stack([quat_rotate(Qeb, v) for v in tst_vecs], axis=-2)
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


def main():  # 自测
    from timeit import timeit
    import time

    # init seed
    seed = int(time.time())
    rng = bkbn.random.default_rng(seed)

    # np.manual_seed(seed)
    # np.cuda.manual_seed(seed)
    # np.cuda.manual_seed_all(seed)
    # np.backends.cudnn.deterministic = True
    # np.backends.cudnn.benchmark = False

    ntest = 100
    bsz = (64,)
    np_float = bkbn.float64
    device = "cpu"
    test_rpy2mat = True
    test_rpy2quat = False
    test_quat_mul = False
    test_quat_rot = False
    test_zem = False
    test_block = False
    test_quat_slerp = False
    test_lerp = False
    test_nlerp = False
    test_geod = False
    test_split = True

    rpy_low = asarray([-math.pi, -math.pi / 2, -math.pi], dtype=np_float).reshape(
        [1] * len(bsz) + [3],
    )
    rpy_high = asarray([math.pi, math.pi / 2, math.pi], dtype=np_float).reshape(
        [1] * len(bsz) + [3],
    )

    def assert_close(x, y, atol=1e-6, rtol=1e-6, title=""):
        tag = ~isclose(x, y, atol=atol, rtol=rtol)
        assert not tag.any(), f"error {title}\n{x[tag]}\n{y[tag]}"

    def _frept1():
        if test_block:
            n = 4 * 50
            d = 3
            los = asarray(rng.normal(size=[*bsz, n, d]), dtype=np_float)
            rball = asarray(rng.random([*bsz, n, 1]), dtype=np_float)
            rst = los_is_visible(los, rball)
        if test_zem:
            n = 50
            m = 50
            d = 3
            p1 = asarray(rng.normal(size=[*bsz, n, 1, d]), dtype=np_float)
            v1 = asarray(rng.normal(size=[*bsz, n, 1, d]), dtype=np_float)
            p2 = asarray(rng.normal(size=[*bsz, 1, m, d]), dtype=np_float)
            v2 = asarray(rng.normal(size=[*bsz, 1, m, d]), dtype=np_float)
            zem, tmiss = calc_zem1(p1 - p2, v1 - v2)

        if test_rpy2mat:
            rpy = affcmb(
                rpy_low, rpy_high, asarray(rng.random([*bsz, 3]), dtype=np_float)
            )
            roll = rpy[..., 0:1]
            Reb = rpy2mat(rpy)
            rpy2 = rpy2mat_inv(Reb, roll)
            err = abs_(rpy2 - rpy).max()
            assert err < 1e-5, "rpy2mat_inv error."

        if test_rpy2quat:
            rpy = affcmb(
                rpy_low, rpy_high, asarray(rng.random([*bsz, 3]), dtype=np_float)
            )
            q = rpy2quat(rpy)
            roll = rpy[..., 0:1]
            rpy2 = rpy2quat_inv(q, roll)
            err = abs_(rpy2 - rpy).max()
            assert err < 1e-3, "rpy2quat_inv error."

        if test_quat_mul:
            p = quat_normalize(asarray(rng.random([*bsz, 4]), dtype=np_float))
            assert is_normalized(p).all(), "Quaternion normalization error."
            q = quat_normalize(asarray(rng.random([*bsz, 4]), dtype=np_float))
            assert is_normalized(q).all(), "Quaternion normalization error."
            r = quat_mul(p, q)
            assert is_normalized(r).all(), "expect |p*q|==1 for |p|=|q|=1"

            q2 = q * 10
            q2inv = quat_inv(q2)
            p2 = quat_mul(quat_mul(p, q2), q2inv)
            rer = abs_(p2 - p).max()
            assert rer < 1e-6, "quat_mul error."

        if test_quat_rot:
            q = quat_normalize(asarray(rng.random([*bsz, 4]), dtype=np_float))
            u = normalize(
                asarray(rng.random([*bsz, 3]), dtype=np_float)
            )  # 单位球面测试
            v = quat_rotate(q, u)
            qconj = quat_conj(q)
            u2 = quat_rotate(qconj, v)
            rer = abs_(u - u2).max()
            assert rer < 1e-4, "Quaternion rotation error is too large."

        if test_quat_slerp:
            B = 10
            N = 5
            q1 = quat_normalize(asarray(rng.random([*bsz, B, 4]), dtype=np_float))
            q2 = quat_normalize(asarray(rng.random([*bsz, 1, 4]), dtype=np_float))
            t = asarray(rng.random([*bsz, B, N]), dtype=np_float)
            qt = quat_slerp(q1, q2, t)
            assert qt.shape == (*bsz, B, N, 4), "quat_interp error."
            assert is_normalized(qt).all(), "Quaternion normalization error."

        if test_lerp:
            d = 3
            N = 10
            x = asarray(rng.random([*bsz, d]), dtype=np_float)
            y = asarray(rng.random([*bsz, d]), dtype=np_float)
            t = asarray(rng.random([*bsz, N]), dtype=np_float)
            z = lerp(x, y, t)
            assert z.shape == (*bsz, N, d), "lerp error."

        if test_nlerp:
            d = 3
            N = 10
            x = asarray(rng.random([*bsz, d]), dtype=np_float)
            y = asarray(rng.random([*bsz, d]), dtype=np_float)
            t = asarray(rng.random([*bsz, N]), dtype=np_float)
            z = nlerp(x, y, t)
            assert z.shape == (*bsz, N, d), "nlerp error."
            tag1 = is_normalized(z)
            tag2 = (abs_(z) <= 1e-6).all(-1, keepdims=True)
            assert (tag1 | tag2).all(), "nlerp error."

    print("Testing...")

    # v2 = quat_rotate_inverse(q, u)
    # Rq = quat2rotmat(q)
    # v = Rq @ v.unsqueeze(-1)
    # m = _quat2rotmat_sqrt(q)

    print("Testing...")
    t = timeit(_frept1, number=ntest)
    print(t)
    if test_split:
        x3 = asarray(rng.random([*bsz, 3]), dtype=np_float)
    # print(r, r.shape)
    # qvq = quat_rotate(q, v)
    # v2 = quat_rotate_inverse(q, qvq)
    # print(qvq, qvq.shape)


if __name__ == "__main__":
    raise RuntimeError("This file is not meant to be run directly.")
