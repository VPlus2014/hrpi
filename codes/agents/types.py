# 用于替换 tianshou.data.types.py
# 此外需要修改 tianshou.data.batch 的泛型支持
from __future__ import annotations
from typing import Generic, Protocol, TypeVar, cast
import numpy as np
import torch


from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol as _BatchProtocol, TArr, TArr_co
from tianshou.data.stats import *
from tianshou.policy.base import TTrainingStats

_T_OBS = TypeVar("_T_OBS", bound=TArr, covariant=True)
_T_ACT = TypeVar("_T_ACT", bound=TArr, covariant=True)

TNestedDictValue = np.ndarray | dict[str, "TNestedDictValue"]


class BatchProtocol(_BatchProtocol[TArr_co], Protocol):

    def to_numpy(self) -> BatchProtocol[np.ndarray]: ...

    def to_torch(self) -> BatchProtocol[torch.Tensor]: ...


class ObsBatchProtocol(BatchProtocol[TArr_co], Protocol):
    """Observations of an environment that a policy can turn into actions.

    Typically used inside a policy's forward
    """

    obs: TArr_co | BatchProtocol[TArr_co]
    info: TArr_co | BatchProtocol[TArr_co]


class RolloutBatchProtocol(ObsBatchProtocol[TArr_co], Protocol):
    """Typically, the outcome of sampling from a replay buffer."""

    obs_next: TArr_co | BatchProtocol[TArr_co]
    act: TArr_co
    rew: TArr_co
    terminated: TArr_co
    truncated: TArr_co

    returns: TArr_co
    weight: TArr_co | float | None


class BatchWithReturnsProtocol(RolloutBatchProtocol[TArr_co], Protocol):
    """With added returns, usually computed with GAE."""

    returns: TArr_co


class PrioBatchProtocol(RolloutBatchProtocol[TArr_co], Protocol):
    """Contains weights that can be used for prioritized replay."""

    weight: TArr_co


class RecurrentStateBatch(BatchProtocol[TArr_co], Protocol):
    """Used by RNNs in policies, contains `hidden` and `cell` fields."""

    hidden: torch.Tensor
    cell: torch.Tensor


class ActBatchProtocol(BatchProtocol[TArr_co], Protocol):
    """Simplest batch, just containing the action. Useful e.g., for random policy."""

    act: TArr_co


class ActStateBatchProtocol(ActBatchProtocol[TArr_co], Protocol):
    """Contains action and state (which can be None), useful for policies that can support RNNs."""

    state: dict | BatchProtocol[TArr_co] | np.ndarray | None
    """Hidden state of RNNs, or None if not using RNNs. Used for recurrent policies.
     At the moment support for recurrent is experimental!"""


class ModelOutputBatchProtocol(ActStateBatchProtocol[TArr_co], Protocol):
    """In addition to state and action, contains model output: (logits)."""

    logits: torch.Tensor


class FQFBatchProtocol(ModelOutputBatchProtocol[TArr_co], Protocol):
    """Model outputs, fractions and quantiles_tau - specific to the FQF model."""

    fractions: torch.Tensor
    quantiles_tau: torch.Tensor


class BatchWithAdvantagesProtocol(BatchWithReturnsProtocol[TArr_co], Protocol):
    """Contains estimated advantages and values.

    Returns are usually computed from GAE of advantages by adding the value.
    """

    adv: torch.Tensor
    v_s: torch.Tensor


class DistBatchProtocol(ModelOutputBatchProtocol[TArr_co], Protocol):
    """Contains dist instances for actions (created by dist_fn).

    Usually categorical or normal.
    """

    dist: torch.distributions.Distribution


class DistLogProbBatchProtocol(DistBatchProtocol[TArr_co], Protocol):
    """Contains dist objects that can be sampled from and log_prob of taken action."""

    log_prob: torch.Tensor


class LogpOldProtocol(BatchWithAdvantagesProtocol[TArr_co], Protocol):
    """Contains logp_old, often needed for importance weights, in particular in PPO.

    Builds on batches that contain advantages and values.
    """

    logp_old: torch.Tensor


class QuantileRegressionBatchProtocol(ModelOutputBatchProtocol[TArr_co], Protocol):
    """Contains taus for algorithms using quantile regression.

    See e.g. https://arxiv.org/abs/1806.06923
    """

    taus: torch.Tensor


class ImitationBatchProtocol(ActBatchProtocol[TArr_co], Protocol):
    """Similar to other batches, but contains `imitation_logits` and `q_value` fields."""

    state: dict | Batch[TArr_co] | np.ndarray | None
    q_value: torch.Tensor
    imitation_logits: torch.Tensor
