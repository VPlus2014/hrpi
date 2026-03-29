from copy import deepcopy
from typing import TypeVar

import numpy as np
import torch


_T_NDArr_co = TypeVar("_T_NDArr_co", np.ndarray, torch.Tensor, covariant=True)


# @torch.no_grad()
def calc_gae(
    v_s: _T_NDArr_co,
    v_s_: _T_NDArr_co,
    rew: _T_NDArr_co,
    truncated: _T_NDArr_co,
    gamma: float,
    gae_lambda: float,
) -> _T_NDArr_co:
    r"""Computes advantages with GAE.

    Note: doesn't compute returns but rather advantages. The return
    is given by the output of this + v_s. Note that the advantages plus v_s
    is exactly the same as the TD-lambda target, which is computed by the recursive


    assert: terminated[i] => v_s[i]=0, rew[i]=0
    assert: terminated[i+1] => v_s_[i+1]=0
    assert: truncated[i] => truncated at v_s[i], and then info from t\geq i+1 is not used

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

    :param v_s: $V_t(S_t)$, shape=(T, ..., dimV)
    :param v_s_: $V_t(S_{t+1})$ shape=(T, ..., dimV)
    :param rew: $r_t=r(S_t, A_t, S_{t+1})$, shape=(T, ..., dimV)
    :param truncated: $\text{truncated}(t)$, shape=(T, ..., 1)
    :param gamma: discount factor
    :param gae_lambda: lambda parameter for GAE, controlling the bias-variance tradeoff
    :return:
        advantages: $A_t^\lambda$, shape=(T, ..., dimV).\
            truncated at v_s[i] => advantages[i]=0, and then info from t\geq i+1 is not used.
    """
    T = len(v_s)
    assert len(v_s_.shape) == len(v_s.shape) == len(rew.shape) == len(truncated.shape)
    assert len(v_s_) == T
    assert len(rew) == T
    assert len(truncated) == T
    delta = (rew + v_s_ * gamma - v_s) * (1.0 - truncated[:T])  # faster than where(...)
    alpha = gamma * gae_lambda
    gae = deepcopy(delta)
    if alpha > 0:
        for i in reversed(range(T - 1)):
            gae[i] += delta[i] + alpha * gae[i + 1]
    return gae
