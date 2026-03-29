# 250617 numpy 内存紧凑型
from __future__ import annotations
from copy import deepcopy
import logging
from typing import (
    Any,
    Iterable,
    Union,
    Sequence,
    TYPE_CHECKING,
    SupportsIndex,
    cast,
    SupportsInt,
)
from heapq import nsmallest
from .proto4data import RolloutBatchProtocol, Batch, BaseReplayBuffer
import numpy as np


_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
_SupportedIndex = Union[
    int, slice, np.ndarray[Any, np.dtype[np.integer]], Sequence[int]
]
if TYPE_CHECKING:
    from numpy.typing import NDArray
_LOGR = logging.getLogger(__name__)


def _isdtypeof(d, t):
    if isinstance(t, Iterable):
        return any(_isdtypeof(d, _t) for _t in t)
    return np.issubdtype(d, t)


def _check_dtype(
    d: type[np.number] | np.dtype, t: Sequence[type[np.number]] | type[np.number], name
):
    assert _isdtypeof(d, t), f"{name} must be in {t}, but got {d}"
    return np.dtype(d).type


def to_shape(x: _ShapeLike) -> tuple[int, ...]:
    return (*np.array(x, dtype=np.intp, ndmin=1).tolist(),)


def _sprint(*items, sep="\n", end="\n"):
    msg = sep.join(str(x) for x in items) + end
    return msg


def _where2list(muti_index: tuple[NDArray[np.integer], ...]):
    idxs = []
    n = muti_index[0].shape[0]
    for i in range(n):
        idxs.append(tuple(idx[i] for idx in muti_index))
    return idxs


try:
    import torch
except ImportError:
    torch = None


def as_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)


def proc_index(
    index: _SupportedIndex,
    capacity: int,
) -> tuple[NDArray[np.integer] | slice, int]:
    # 访问速度: slice>>ndarray>>list>range
    if isinstance(index, slice):
        _idx = index
        n_select = len(range(*_idx.indices(capacity)))
    elif index is Ellipsis:
        _idx = slice(None)
        n_select = capacity
    else:
        _idx = np.ravel(index)
        n_select = len(_idx)  # iff index.dtype != bool
    if isinstance(_idx, np.ndarray):
        assert np.issubdtype(_idx.dtype, np.integer), (
            "index must be integer array",
            _idx.dtype,
        )
    return _idx, n_select


def calc_index_len(index: _SupportedIndex, capacity: int):
    if isinstance(index, slice):
        n = len(range(*index.indices(capacity)))
    elif index == Ellipsis:
        n = capacity
    else:
        index = np.ravel(index)
        assert np.issubdtype(index.dtype, np.integer), (
            "index must be integer array",
            index.dtype,
        )
        n = len(index)
    return n


_NOTFIND = object()


class RETrajReplayBuffer(BaseReplayBuffer):

    def __init__(
        self,
        max_trajs: SupportsInt,
        max_steps: SupportsInt,
        obs_shape: _ShapeLike,
        act_shape: _ShapeLike,
        rew_shape: _ShapeLike = 1,
        float_dtype: type[np.floating] = np.float64,
        obs_dtype: type[np.floating | np.integer] | None = None,
        act_dtype: type[np.floating | np.integer] | None = None,
        aux_infos: dict[
            str,  # name
            tuple[
                _ShapeLike,  # $0 shape
                type[np.number] | np.dtype | None,  # $1 dtype bounds
                np.number | np.ndarray | int | float | bool | str,  # $2 default value
            ],
        ] = {},
        size_stack_in: SupportsInt = 1,
        logger: logging.Logger = _LOGR,
        debug: bool = False,
        batch_first: bool = False,
    ):
        """
        RAM-Efficient Trajectory Replay Buffer with double buffer.
        Args:
            max_trajs (int): maximum number of done trajectories to store.
            max_steps (int): maximum act steps per trajectory.
            obs_shape (_ShapeLike): shape of observation space.
            act_shape (_ShapeLike): shape of action space, default to 1.
            rew_shape (_ShapeLike): shape of reward space, default to 1.
            float_dtype (type[np.floating]): dtype for float data, default to `np.float64`
            state_dtype: dtype for obs, default to `float_dtype`
            action_dtype:  dtype for act, default to `float_dtype`
            aux_infos (dict): auxiliary information to store, default to empty.
                - shape: shape of auxiliary information, default to 1.
                - dtype: dtype of auxiliary information, default to `float_dtype`
                - default: default value of auxiliary information, default to 0.
            size_stack_in (int): batch size on writing, default to 1.
            batch_first (bool): return samples in batch-major format, \
                True->(B,T,...), False->(T,B,...), default to False.
        """
        # self.sample_indices.__doc__ = BaseReplayBuffer.sample_indices.__doc__
        self.logger = logging.getLogger(logger.name)
        self.DEBUG = debug
        self.batch_first = bool(batch_first)

        self._obs_shape = obs_shape = to_shape(obs_shape)
        assert len(obs_shape) > 0, ("obs_shape must be at least 1D tensor", obs_shape)

        self._act_shape = act_shape = to_shape(act_shape)
        assert len(act_shape) > 0, ("act_shape must be at least 1D tensor", act_shape)

        self._rew_shape = rew_shape = to_shape(rew_shape)
        assert len(rew_shape) == 1, ("rew_shape must be at least 1D tensor", rew_shape)

        self._t_cap = max_steps = int(max_steps)
        assert max_steps > 0, ("max_steps must be positive", max_steps)

        self._max_trajs = max_trajs = int(max_trajs)  # .capacity()
        assert max_trajs > 0, ("max_trajs must be positive", max_trajs)

        self._w_size = _w_size = int(size_stack_in)
        assert _w_size > 0, ("size_w must be positive", _w_size)
        self._w_slot = np.arange(_w_size)  # I_B \to I_N
        np.random.shuffle(self._w_slot)  # shuffle write pointer
        if self.DEBUG:
            self.logger.debug(
                "init iW->iB: "
                + " ".join(f"{iw}-{ib}" for iw, ib in enumerate(self._w_slot))
            )

        self._b_cap = _b_cap = max_trajs + _w_size
        self._b_sz_done = 0  # .size()
        self._t_sz = np.zeros(_b_cap, dtype=np.intp)
        """current act steps (N,)"""

        self._float_dtype = float_dtype = _check_dtype(
            float_dtype or np.float_, (np.floating,), "float_dtype"
        )
        self._obs_dtype = obs_dtype = _check_dtype(
            obs_dtype or float_dtype, (np.floating, np.integer), "obs_dtype"
        )
        self._act_dtype = act_dtype = _check_dtype(
            act_dtype or float_dtype, (np.floating, np.integer), "act_dtype"
        )

        _sphdx = (_b_cap, max_steps + 1)
        _sphda = (_b_cap, max_steps)

        self._obs = np.zeros(_sphdx + obs_shape, dtype=obs_dtype)
        """shape=(N,T+1,*obs_shape)"""
        self._act = np.zeros(_sphda + act_shape, dtype=act_dtype)
        """shape=(N,T,*act_shape)"""
        self._rew = np.zeros(_sphda + rew_shape, dtype=float_dtype)
        """shape=(N,T,dimV)"""
        self._term = np.zeros(_sphdx, dtype=np.bool_)
        """shape=(N,T+1)"""
        self._trunc = np.zeros(_sphdx, dtype=np.bool_)
        """shape=(N,T+1)"""
        self._aux: dict[str, np.ndarray] = {}
        self._aux_default = {}
        for k, (aux_shape, aux_dtype, aux_default) in aux_infos.items():
            aux_shape = to_shape(aux_shape)
            aux_dtype = _check_dtype(aux_dtype or float_dtype, (np.number,), k)
            _aux = np.zeros(_sphda + aux_shape, dtype=aux_dtype)
            self._aux[k] = _aux
            self._aux_default[k] = aux_default
        #
        self._done = np.zeros(_b_cap, dtype=np.bool_)  # term|trunc
        """shape=(N,)"""
        self._return = np.zeros((_b_cap, 1) + rew_shape, dtype=float_dtype)
        """shape=(N,1,dimR)"""

        self._rank = np.zeros(_b_cap, dtype=np.intp)  # priority
        self._ranknext = 1

        self.clean()
        self.__init_super()

    def __init_super(self):
        # super().__init__
        self._meta = {}
        # self.stack_num = self._w_size
        return

    def _erase_trajs(self, index: _SupportedIndex = slice(None)):
        _ib = proc_index(index, self._b_cap)[0]
        self._obs[_ib, ...] = 0
        self._act[_ib, ...] = 0
        self._rew[_ib, ...] = 0.0
        self._term[_ib, 1:, ...] = True
        self._term[_ib, 0, ...] = False  # first step is not terminated
        self._trunc[_ib, 1:, ...] = True  # default to truncated at all S_t^+
        self._trunc[_ib, 0, ...] = False  # first step is not truncated
        self._return[_ib, ...] = 0.0
        for k, aux in self._aux.items():
            if self.DEBUG:
                assert isinstance(aux, np.ndarray), f"buf {k} must be a numpy array"
            aux[_ib, ...] = 0
        self._t_sz[_ib] = 0
        self._done[_ib] = False
        if self.DEBUG:
            self.logger.debug(f"erase trajs@iB={_ib}")
        # self._update_b_size()

    def _update_b_size(self):
        self._b_sz_done = int(self._done.sum())
        if self.DEBUG:
            self.logger.debug(f"done trajs num->{self._b_sz_done}")

    def add(
        self,
        obs,
        act,
        obs_next,
        rew,
        term,
        trunc,
        **aux,
    ):
        """Add one step of data to buffer."""
        logr = self.logger

        _shphd = (self._w_size,)  # (n,)
        obs = np.reshape(as_np(obs), _shphd + self._obs.shape[2:])  # (n, *shapeO)
        act = np.reshape(as_np(act), _shphd + self._act.shape[2:])  # (n, *shapeA)
        obs_next = np.reshape(as_np(obs_next), obs.shape)  # (n, *shapeO)
        rew = np.reshape(as_np(rew), _shphd + self._rew.shape[2:])  # (n, *shapeR)
        term = np.reshape(as_np(term), _shphd)  # (n,)
        trunc = np.reshape(as_np(trunc), _shphd)  # (n,)

        b_idx = self._w_slot
        ta = self._t_sz[b_idx]
        if self.DEBUG:
            assert (ta < self._t_cap).all(), "buffer overflow"

        tx = ta + 1
        self._obs[b_idx, ta, ...] = obs
        self._act[b_idx, ta, ...] = act
        self._obs[b_idx, tx, ...] = obs_next
        self._rew[b_idx, ta, ...] = rew
        self._term[b_idx, tx, ...] = term  # S_{t+1} is terminated

        for _k, _dst in self._aux.items():
            _src = aux.get(_k, None)
            if _src is None:
                continue
            assert isinstance(_dst, np.ndarray), f"{_k} must be a ndarray"
            _src = np.reshape(as_np(_src), _shphd + _dst.shape[2:])  # (N, *shapeK)
            _dst[b_idx, ta, ...] = _src

        self._t_sz[b_idx] = tx  # length++
        full = tx >= self._t_cap
        trunc_ = trunc | full  # (B,) truncation at S_{t+1}
        self._trunc[b_idx, tx, ...] = trunc_
        done = term | trunc_  # (B,)
        self._done[b_idx] |= done
        if done.any():
            _iw = np.where(done)[0]  # \subset [0,B)
            _ib = b_idx[_iw]

            self._return[_ib, :, ...] = self._rew[_ib].sum(
                1, keepdims=True
            )  # 计算return
            if self.DEBUG:
                logr.debug(
                    "done traj iW->iB:\n"
                    + "\n".join(f"{i1_}-{i2_}" for i1_, i2_ in zip(_iw, _ib))
                )
            # self._update_rank(_ib)

            _ib = self._realloc_traj(_iw)  # 重新分配traj槽
            self._w_slot[_iw] = _ib
            if self.DEBUG:
                logr.debug(
                    "realloc iW->iB:\n"
                    + "\n".join(f"{i1_}-{i2_}" for i1_, i2_ in zip(_iw, _ib))
                )

            self._erase_trajs(_ib)
            self._update_b_size()
            self._update_rank(_ib)

        if self.DEBUG:
            try:
                self._check_lib()
            except Exception as e:
                logr.error(f"check_lib error: {e}")
                raise e

    def size_of_done_trajs(self):
        """Number of done trajectories."""
        return self._b_sz_done

    def size_of_nonempty_trajs(self, min_steps=1):
        """Number of trajectories with at least `min_steps` steps."""
        return int((self._t_sz >= min_steps).sum())

    def sample_indices(self, batch_size: int | None, min_steps=1) -> NDArray[np.intp]:
        assert min_steps > 0, ("expect min_steps>0, got", min_steps)
        _valid = self._t_sz >= min_steps
        n_valid = int(_valid.sum())
        if batch_size is None or batch_size == 0:
            batch_size = n_valid
        elif batch_size > 0:
            batch_size = min(batch_size, n_valid)
        else:
            return np.array([], dtype=np.intp)
        # assert 0 <= batch_size <= ntraj, (
        #     f"expected batch_size in [0,{ntraj}], but got",
        #     batch_size,
        # )
        traj_idxs = np.where(_valid)[0]
        if batch_size == n_valid:
            _ib = traj_idxs
        else:
            _ib = np.random.choice(traj_idxs, size=batch_size, replace=False)
        return _ib

    def sample(
        self,
        batch_size: int | None = None,
        compact: bool | None = None,
        batch_first: bool | None = None,
        min_steps=1,
    ) -> tuple[RolloutBatchProtocol, NDArray[np.intp]]:
        """
        随机采样
        Args:
            batch_size: 采样数量 None->all available
            compact: 是否压缩轨迹, 默认 True, 即时间维长度压缩到子集的最大有效长度
            batch_first: True->(N,L,D), False->(L,N,D)
            min_steps: filter out trajectories with less than `min_steps` steps
        """
        _ib = self.sample_indices(batch_size, min_steps=min_steps)
        data = self._get_item(_ib, compact=compact, batch_first=batch_first)
        return data, _ib

    def _check_selected_data(
        self,
        data: RolloutBatchProtocol,
        index: _SupportedIndex,
        batch_first: bool | None = None,
    ):
        batch_first = batch_first or self.batch_first
        obs = data.obs
        act = data.act
        term = data.terminated
        trunc = data.truncated
        rew = data.rew
        # 野值检验
        for k, v in [
            ("obs", obs),
            ("act", act),
            ("terminated", term),
            ("truncated", trunc),
            ("rew", rew),
        ]:
            assert isinstance(v, np.ndarray), f"{k} must be a numpy array"
            valid = np.isfinite(v)
            assert valid.all(), ValueError(
                _sprint(f"{k} contains non-finite value", _where2list(np.where(~valid)))
            )

        for k in self._aux:
            v = data.get(k, None)
            if v is None:
                continue
            assert isinstance(v, np.ndarray), f"{k} must be a numpy array"
            if np.issubdtype(v.dtype, np.floating):
                valid = np.isfinite(v)
                assert valid.all(), ValueError(
                    _sprint(
                        f"{k} contains non-finite value", _where2list(np.where(~valid))
                    )
                )

        if batch_first:
            term = np.swapaxes(term, 0, 1)  # (L+1,B,1)
            trunc = np.swapaxes(trunc, 0, 1)  # (L+1,B,1)
            rew = np.swapaxes(rew, 0, 1)  # (L,B,dimV)

        # 单调性检验
        for k, v in [("terminated", term), ("truncated", trunc)]:
            assert np.all(~v[0]), f"first step of {k} must be False"
            v1 = v[:-1]
            v2 = v[1:]
            mono = v1 <= v2
            assert mono.all(), ValueError(
                _sprint(f"{k} must be monotonic", _where2list(np.where(~mono)))
            )

        # 奖励检查
        done = term | trunc
        done1 = done[:-1]
        invalid = done1 & np.not_equal(rew, 0.0)
        assert not invalid.any(), (
            f"reward of done step must be 0.0",
            np.where(invalid),
        )
        return data

    def _check_lib(self):
        for k, v in [("term", self._term), ("trunc", self._trunc)]:
            v1 = v[:, :-1]
            v2 = v[:, 1:]
            mono = v1 <= v2
            assert mono.all(), ValueError(
                _sprint(f"{k} must be monotonic", _where2list(np.where(~mono)))
            )
        term1 = self._term[:, :-1]
        trunc1 = self._trunc[:, :-1]
        rew = self._rew
        done1 = term1 | trunc1
        done1 = np.reshape(done1, done1.shape[:2] + (1,))
        err = np.not_equal(rew, 0.0) & done1
        assert not err.any(), ValueError(
            _sprint(
                f"reward of done step must be 0.0",
                _where2list(np.where(err)),
            )
        )

    def _get_item(
        self,
        batch_index: _SupportedIndex,
        compact: bool | None = None,
        batch_first: bool | None = None,
        copy=True,
    ):
        _ib = batch_index
        compact = compact or True
        if compact:
            tmax = self._t_sz[_ib].max()
            _ta = slice(None, tmax)
            _tx = slice(None, tmax + 1)
        else:
            _ta = _tx = slice(None)

        batch_first = batch_first or self.batch_first
        if batch_first:
            _cvtr = lambda x: x
        else:
            _cvtr = lambda x: np.swapaxes(x, 0, 1)

        obs = _cvtr(self._obs[_ib, _tx, ...])
        term = _cvtr(self._term[_ib, _tx, ...])
        trunc = _cvtr(self._trunc[_ib, _tx, ...])
        act = _cvtr(self._act[_ib, _ta, ...])
        rew = _cvtr(self._rew[_ib, _ta, ...])
        _aux = {}
        for _key, _src in self._aux.items():
            _dst = _cvtr(_src[_ib, _ta, ...])
            _aux[_key] = _dst

        term = term.reshape(term.shape[:2] + (1,))
        trunc = trunc.reshape(trunc.shape[:2] + (1,))

        data = dict(
            obs=obs,
            act=act,
            rew=rew,
            truncated=trunc,
            terminated=term,
            **_aux,
        )
        data = deepcopy(data)
        data = Batch(data, copy=False)
        data = cast(RolloutBatchProtocol, data)
        if self.DEBUG:
            self._check_selected_data(data, _ib, batch_first=batch_first)
        return data

    def __getitem__(self, batch_index: _SupportedIndex) -> RolloutBatchProtocol:
        return self._get_item(batch_index)

    def reset(self, keep_statistics=False):
        self.clean()
        return super().reset(keep_statistics=keep_statistics)

    def clean(self):
        """清空全部轨迹(writing+done)"""
        self._erase_trajs()
        self._update_b_size()
        self._update_rank(self._w_slot)

    def clean_cache(self):
        """清空正在写入的轨迹"""
        self._erase_trajs(self._w_slot)
        self._update_b_size()
        self._update_rank(self._w_slot)

    def clean_done(self):
        """清空已完成写入的轨迹"""
        idxs = np.where(self._done)[0]
        self._erase_trajs(idxs)
        self._update_b_size()

    def _realloc_traj(self, w_idx: NDArray[np.integer]):
        """find new traj slot for write"""
        m = len(w_idx)
        rks = self._rank
        # speed: range > arange
        new_traj_idx = nsmallest(m, range(len(rks)), key=rks.__getitem__)  # 新地址
        new_traj_idx = np.asarray(new_traj_idx)
        return new_traj_idx

    def _update_rank(self, traj_idx: np.ndarray | slice):
        tgt = self._ranknext
        self._ranknext += 1
        self._rank[traj_idx] = tgt  # 更新优先级
        if self.DEBUG:
            self.logger.debug(
                (
                    "set rank=",
                    tgt,
                    "@iB",
                    traj_idx,
                    # "all rank=",
                    # self._rank,
                )
            )
        if self._ranknext >= self._b_cap:
            self._rank -= tgt
            self._ranknext = 1
            if self.DEBUG:
                self.logger.debug("reset all rank")

    @property
    def num_trajs(self):
        """number of done trajectories."""
        return self._b_sz_done  # @size

    def __len__(self):
        return self._b_sz_done  # @len

    @property
    def max_size(self):
        return self._max_trajs


# def merge_trajs(
#     bufs: List[VanillaReplayBuffer],
#     float_dtype=np.float32,
#     action_dtype=np.float32,
# ) -> RETrajReplayBuffer:
#     lens = [len(buf.obs) for buf in bufs]
#     N = len(bufs)
#     assert N > 0, "Empty buffer list"
#     L = max(lens)
#     assert L > 0, "All are empty buffer"
#     dimX = np.ravel(bufs[0].obs[0]).shape[0]
#     dimA = np.ravel(bufs[0].act[0]).shape[0]
#     dimLogPA = np.ravel(bufs[0].act_log_prob[0]).shape[0]
#     Xs = np.zeros((L + 1, N, dimX), dtype=float_dtype)
#     nonterms = np.zeros((L + 1, N, 1), dtype=np.bool_)
#     trunc = np.zeros((L + 1, N, 1), dtype=np.bool_)
#     As = np.zeros((L, N, dimA), dtype=action_dtype)
#     Rs = np.zeros((L, N, 1), dtype=float_dtype)
#     LogPAs = np.zeros((L, N, dimLogPA), dtype=float_dtype)
#     for i, buf in enumerate(bufs):
#         L_i = lens[i]
#         Xs[:L_i, i, :] = np.asarray(buf.obs, dtype=float_dtype).reshape(L_i, dimX)
#         Xs[L_i, i, :] = np.asarray(buf.obs_next[-1], dtype=float_dtype).reshape(1, dimX)
#         As[:L_i, i, :] = np.asarray(buf.act, dtype=action_dtype).reshape(L_i, dimA)
#         Rs[:L_i, i, :] = np.asarray(buf.rew, dtype=float_dtype).reshape(L_i, 1)
#         LogPAs[:L_i, i, :] = np.asarray(buf.act_log_prob, dtype=float_dtype).reshape(
#             L_i, dimLogPA
#         )
#         # assert not any(buf.term[:-1])
#         nonterms[0, i, :] = True
#         nonterms[1 : L_i + 1, i, :] = np.logical_not(
#             np.asarray(buf.terminated, dtype=np.bool_).reshape(L_i, 1)
#         )
#         trunc[1 : L_i + 1, i] = np.asarray(buf.truncated, dtype=np.bool_).reshape(
#             L_i, 1
#         )

#     return RETrajReplayBuffer(
#         obs=Xs, trunc=trunc, term=~nonterms, act=As, rew=Rs, act_log_prob=LogPAs
#     )
