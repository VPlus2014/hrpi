from functools import partial
import time
from timeit import timeit
import numpy as np
from numba import njit, jit


def _fix_v_form1(vs: np.ndarray, end_tag: np.ndarray):
    return vs * (1.0 - end_tag.astype(vs.dtype))


@njit
def _fix_v_form1_jit(vs: np.ndarray, end_tag: np.ndarray):
    return vs * (1.0 - end_tag.astype(vs.dtype))
    # vs_ = vs.copy()
    # for i in range(vs.shape[0]):
    #     vs_[i] = vs[i] * (1.0 - end_tag[i])
    # return vs_


def _fix_v_form2(vs: np.ndarray, end_tag: np.ndarray):
    return np.where(end_tag, 0.0, vs)


@njit
def _fix_v_form2_jit(vs: np.ndarray, end_tag: np.ndarray):
    # vs_ = vs.copy()
    # for i in range(vs.shape[0]):
    #     vs_[i] = np.where(end_tag[i], 0.0, vs[i])
    # return vs_
    return np.where(end_tag, 0.0, vs)


@njit
def gae_bwd_jit(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    trunc: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    r"""Computes advantages with GAE.

    Note: doesn't compute returns but rather advantages. The return
    is given by the output of this + v_s. Note that the advantages plus v_s
    is exactly the same as the TD-lambda target, which is computed by the recursive


    term[i] => v_s[i]=0, rew[i]=0
    term[i+1] => v_s_[i]=0
    trunc[i] => truncated at v_s[i], and v_s_[i] is invalid

    formula:

    .. math::
        V_{t} := V(S_t) * not \text{terminated}_t \n
        G_t^\lambda = r_t + \gamma ( \lambda G_{t+1}^\lambda + (1 - \lambda) V_{t+1} )

    The GAE is computed recursively as:

    .. math::
        \delta_t = (r_t + \gamma V_{t+1} - V_t) * not \text{truncated}_t \n
        A_t^\lambda= \delta_t + \gamma \lambda A_{t+1}^\lambda

    And the following equality holds:

    .. math::
        G_t^\lambda = A_t^\lambda+ V_t

    :param v_s: values in an episode, i.e. $V_t(S_t)$, shape=(T, ..., dimV)
    :param v_s_: next values in an episode, i.e. v_s shifted by 1, equivalent to
        $V_{t+1}$, shape=(T, ..., dimV)
    :param rew: rewards in an episode, i.e. $r_t$
    :param trunc: boolean array indicating whether the episode is truncated at current state;
        trunc[t] is True means the episode is truncated at state.
    :param gamma: discount factor
    :param gae_lambda: lambda parameter for GAE, controlling the bias-variance tradeoff
    :return:
    """
    delta = (rew + v_s_ * gamma - v_s) * (1.0 - trunc.astype(v_s.dtype))
    discount = gamma * gae_lambda
    gae = delta.copy()
    T = len(rew)
    for i in range(T - 2, -1, -1):
        gae[i] += delta[i] + discount * gae[i + 1]
    return gae


def gae_bwd1(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    trunc: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    delta = (rew + v_s_ * gamma - v_s) * (1.0 - trunc.astype(v_s.dtype))
    discount = gamma * gae_lambda
    gae = delta.copy()
    for i in range(len(rew) - 2, -1, -1):
        gae[i] += discount * gae[i + 1]
    return gae


def gae_fwd1(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    trunc: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    # raise NotImplementedError("O(T^2) ")
    delta = (rew + v_s_ * gamma - v_s) * (1.0 - trunc.astype(v_s.dtype))
    gae = np.empty_like(delta)
    beta = gamma * gae_lambda
    betas = np.logspace(
        0, len(rew) - 1, len(rew), base=beta, dtype=delta.dtype
    ).reshape((-1,) + (1,) * len(delta.shape[1:]))
    T = len(rew)
    for i in range(T):
        gae[i] = (delta[i:T] * betas[0 : T - i]).sum(0)
    return gae


def gen_data(T: int, B: int, dimV: int, seed=None, dtype=np.float32):
    rng = np.random.default_rng(seed)

    term_all = rng.random((T + 1, B, 1)) < 0.1
    trunc_all = rng.random((T + 1, B, 1)) < 0.1
    term_all = np.cumsum(term_all, axis=0, dtype=np.bool_)
    trunc_all = np.cumsum(trunc_all, axis=0, dtype=np.bool_)
    trunc_all[-1, ...] = True
    vs_all = rng.normal(size=(T + 1, B, dimV)).astype(dtype)

    term_ = term_all.astype(dtype)
    vs_all = _fix_v_form1(vs_all, term_)
    rs = rng.uniform(size=(T, B, dimV)).astype(dtype)
    rs = _fix_v_form1(rs, term_all[:-1])
    vs = vs_all[:-1]
    vs_ = vs_all[1:]
    term = term_all[:-1]
    trunc = trunc_all[:-1]
    return (vs, vs_, rs, term, trunc)


def main():
    from tqdm import tqdm

    T = 1000
    B = 32
    dimV = 1
    seed = int(time.time())
    # data = gen_data(T, B, 42)
    nt1 = 100
    nt2 = 10
    _funcs = [gae_bwd1, gae_bwd_jit, gae_fwd1]
    # _funcs = [_fix_v_form1, _fix_v_form1_jit, _fix_v_form2, _fix_v_form2_jit]
    for _f in _funcs:
        rng = np.random.default_rng(seed)
        t = 0
        fname = _f.__name__
        qbar = tqdm(range(nt1))
        qbar.set_description(f"{fname}")
        for _ in qbar:
            vs1, vs2, rs, term, trunc = gen_data(
                T, B, dimV, seed=rng.integers(np.iinfo(np.int32).max)
            )
            args = (vs1, vs2, rs, trunc, 0.99, 0.95)
            # args = (vs1, term)
            _fwrapped = partial(_f, *args)
            t += timeit(_fwrapped, number=nt2)
        trajps = B * nt1 * nt2 / t
        print(f"{fname}: {t:.5g}s, {trajps:.2f} traj/s")


if __name__ == "__main__":
    main()
