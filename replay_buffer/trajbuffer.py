from __future__ import annotations
from copy import deepcopy
import logging
from typing import Any, Union, Sequence, TYPE_CHECKING, SupportsIndex, cast
from heapq import nsmallest
from .protocol import RolloutBatchProtocol, Batch
import numpy as np

_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]


_DEBUG = True
_LOGR = logging.getLogger(__name__)
# class Trajectory:

#     def __init__(
#         self,
#         max_steps: int,
#         obs_shape: Sequence[int],
#         act_shape: Sequence[int],
#         float_dtype=np.float32,
#     ):
#         pass


class RETrajReplayBuffer:
    """RAM-efficient trajectory replay buffer."""

    obs: np.ndarray  # (L+1, N, dimX)
    act: np.ndarray  # (L, N, dimA)
    rew: np.ndarray  # (L, N, 1)
    act_log_prob: np.ndarray  # (L, N, 1)
    term: np.ndarray  # (L+1, N, 1)
    trunc: np.ndarray  # (L+1, N, 1)

    def __init__(
        self,
        max_steps: int,  # 最大决策步数 L
        max_trajs: int,  # (面向训练)最大采样轨迹数 N
        num_envs: int,  # (面向环境)并行采样环境数
        obs_shape: Sequence[int],  # 观测空间维度
        act_shape: Sequence[int],  # 动作空间维度
        float_dtype=np.float32,
        state_dtype: np.dtype | Any | None = None,  # 默认为 float_dtype
        action_dtype: np.dtype | Any | None = None,  # 默认为 float_dtype
        logr=_LOGR,
    ):
        assert len(obs_shape) > 0, ("obs_shape must be non-empty", obs_shape)
        assert len(act_shape) > 0, ("act_shape must be non-empty", act_shape)

        self._max_steps = max_steps
        self._max_trajs = max_trajs
        self._traj_cap = _traj_cap = max_trajs + num_envs  # 实际大小
        self._ntraj = 0  # 当前存储轨迹数
        self._num_envs = num_envs
        self._obs_shape = obs_shape = tuple(obs_shape)
        self._act_shape = act_shape = tuple(act_shape)
        self._float_dtype = float_dtype
        self._state_dtype = state_dtype or float_dtype  # None->float_dtype
        self._action_dtype = action_dtype or float_dtype  # None->float_dtype
        self.logr = logging.getLogger(logr.name)

        self.__check()

        self._obs = np.zeros(
            (max_steps + 1, _traj_cap) + obs_shape, dtype=self._state_dtype
        )
        self._term = np.zeros((max_steps + 1, _traj_cap, 1), dtype=np.bool_)
        self._trunc = np.zeros((max_steps + 1, _traj_cap, 1), dtype=np.bool_)
        self._act = np.zeros(
            (max_steps, _traj_cap) + act_shape, dtype=self._action_dtype
        )
        self._rew = np.zeros((max_steps, _traj_cap, 1), dtype=float_dtype)
        self._logpa = np.ones((max_steps, _traj_cap) + act_shape, dtype=float_dtype)
        #
        self._ptrE2N = np.arange(num_envs)  # 正在填充的env编号 env_idx -> traj_idx
        self._alens = np.zeros(_traj_cap, dtype=np.int64)  # 动作轨迹当前长度
        self._done = np.zeros(_traj_cap, dtype=np.bool_)  # 轨迹是否满/结束
        self._ranktime = 0
        self._rank = np.empty(_traj_cap, dtype=np.int64)  # 优先级
        self.clear()

    def __check(self):
        max_steps = self._max_steps  # @__check
        max_trajs = self._max_trajs  # @__check
        num_envs = self._num_envs  # @__check
        obs_shape = self._obs_shape  # @__check
        act_shape = self._act_shape  # @__check
        assert max_steps > 0, ("max_steps must be positive", max_steps)
        assert max_trajs > 0, ("max_trajs must be positive", max_trajs)
        assert num_envs > 0, ("num_envs must be positive", num_envs)
        assert len(obs_shape) > 0, ("obs_shape must be non-empty", obs_shape)
        assert len(act_shape) > 0, ("act_shape must be non-empty", act_shape)

    @property
    def size(self):
        """有效轨迹数"""
        return self._ntraj  # @size

    def __len__(self):
        return self._ntraj  # @len

    def sample(
        self,
        batch_size: int | None = 1,
        compact=True,
    ) -> RolloutBatchProtocol:
        """
        随机采样
        Args:
            batch_size: 采样数量 None->全部
            compact: 是否压缩轨迹, 默认True, 即压缩到子集的最大长度
        """
        ntraj = self._ntraj  # @sample
        if batch_size is None:
            batch_size = ntraj
        assert batch_size <= ntraj, (
            "batch_size must be less than or equal to trajs",
            batch_size,
            ntraj,
        )
        traj_idxs = np.where(self._done)[0]
        if batch_size == ntraj:
            _trj_idx = traj_idxs
        else:
            _trj_idx = np.random.choice(traj_idxs, size=batch_size, replace=False)

        if compact:
            tmax = self._alens[_trj_idx].max()
            ta = slice(None, tmax)
            tx = slice(None, tmax + 1)
        else:
            ta = tx = slice(None)

        obs = self._obs[tx, _trj_idx, ...]
        term = self._term[tx, _trj_idx, ...]
        trunc = self._trunc[tx, _trj_idx, ...]
        act = self._act[ta, _trj_idx, ...]
        rew = self._rew[ta, _trj_idx, ...]
        if self._logpa is not None:
            act_log_prob = self._logpa[ta, _trj_idx, ...]
        else:
            act_log_prob = None
        data = Batch(
            obs=obs,
            act=act,
            rew=rew,
            act_log_prob=act_log_prob,
            done=term,
            truncated=trunc,
            terminated=term,
        )
        data = deepcopy(data)
        data = cast(RolloutBatchProtocol, data)
        return data

    def clear(self):
        self._ptrE2N[:] = np.arange(self._num_envs)
        self._erase_trajs()
        self._reset_rank()
        self._reset_ntaj()

    def _reset_rank(self):
        cap = self._traj_cap
        self._rank[:] = np.arange(cap)
        self._ranktime = cap

    def _erase_trajs(self, idx: Sequence[int] | np.ndarray | slice = slice(None)):
        self._obs[:, idx, ...] = 0
        self._act[:, idx, ...] = 0
        self._rew[:, idx, ...] = 0
        self._term[:, idx, ...] = True
        self._trunc[:, idx, ...] = True
        self._term[0, idx, ...] = False  # 第一个状态不为终止态
        self._trunc[0, idx, ...] = False
        if self._logpa is not None:
            self._logpa[:, idx, ...] = 1

        self._alens[idx] = 0
        self._done[idx] = False
        # self._on_len_change()

    def _realloc_traj(self, env_idx: np.ndarray):
        """为env数据指针查找新的traj槽"""
        m = len(env_idx)
        rks = self._rank
        new_traj_idx = nsmallest(m, range(len(rks)), key=rks.__getitem__)  # 新地址
        new_traj_idx = np.asarray(new_traj_idx)
        return new_traj_idx

    def _update_rank(self, traj_idx: np.ndarray | slice):
        self._ranktime += 1
        self._rank[traj_idx] = self._ranktime  # 更新优先级
        if _DEBUG:
            self.logr.debug(
                (
                    "rank<-",
                    self._ranktime,
                    "@",
                    traj_idx,
                    # "all rank=",
                    # self._rank,
                )
            )

    def _reset_ntaj(self):
        self._ntraj = int(self._done.sum())  # 可采样的轨迹数
        if _DEBUG:
            self.logr.debug(f"ntraj->{self._ntraj}")

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        obs_next: np.ndarray,
        rew: np.ndarray,
        term: np.ndarray,
        trunc: np.ndarray,
        act_log_prob: np.ndarray,
    ):
        """
        面向并行env添加最新的一步数据
        Args:
            obs: (nenvs, shapeX)
            act: (nenvs, shapeA)
            obs_next: (nenvs, shapeX) 后继状态
            rew: (nenvs,1)
            term: (nenvs,1)
            trunc: (nenvs,1)
            act_log_prob: (nenvs,shapeA)
        """
        assert obs.shape[0] == self._num_envs, (
            "batch size error",
            obs.shape[0],
            "expect",
            self._num_envs,
        )
        logr = self.logr
        done = term | trunc
        # 1. 常规数据
        traj_idx = self._ptrE2N
        t_idx = self._alens[traj_idx]
        need_alloc = t_idx == 0  # 空轨迹写前的预处理
        if need_alloc.any():
            _traj_idx = traj_idx[need_alloc]
            if _DEBUG:
                logr.debug(("clean", _traj_idx))
            self._erase_trajs(_traj_idx)  # 擦除轨迹
            self._update_rank(_traj_idx)  # 更新优先级
        tcap = self._max_steps  # 决策片段长度
        if _DEBUG:
            assert (t_idx < tcap).all(), "buffer overflow"
            # print("write", traj_idx)
        self._obs[t_idx, traj_idx, ...] = obs
        self._act[t_idx, traj_idx, ...] = act
        self._rew[t_idx, traj_idx, ...] = rew
        if self._logpa is not None:
            self._logpa[t_idx, traj_idx, ...] = act_log_prob
        #
        self._obs[t_idx + 1, traj_idx, ...] = obs_next
        self._term[t_idx, traj_idx, ...] = term
        self._trunc[t_idx, traj_idx, ...] = trunc

        self._alens[traj_idx] += 1  # 轨迹长度变化

        _done = done.ravel()
        _need_realloc = _done | (self._alens[traj_idx] >= tcap)
        if _need_realloc.any():
            _env_idx = np.where(_need_realloc)[0]  # \subset [0,nenvs)
            _traj_idx0 = traj_idx[_env_idx]
            self._done[_traj_idx0] = True  # 标记结束
            if _DEBUG:
                logr.debug(("done@", _traj_idx0.tolist()))

            _traj_idx = self._realloc_traj(_env_idx)  # 重新分配traj槽
            self._ptrE2N[_env_idx] = _traj_idx
            self._alens[_traj_idx] = 0  # 重置traj长度
            self._done[_traj_idx] = False  # 重置done标志
            if _DEBUG:
                logr.debug(("alloc full envs->new traj", _env_idx, "->", _traj_idx))

            self._reset_ntaj()

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


def demo():
    pass
    # action_dtype = np.int32
    # max_steps = 5
    # ntrajs = 4

    # def _bufmaker():
    #     n = np.random.randint(1, max_steps + 1)
    #     x1s = np.random.rand(n)
    #     acts = np.random.normal(0, 3, size=(n, 4)).astype(action_dtype)
    #     x2s = np.random.rand(n)
    #     x2s[:-1] = x1s[1:]
    #     rews = np.random.rand(n)
    #     term = np.zeros(n, dtype=bool)
    #     term[-1] = np.random.rand() < 0.5
    #     trunc = np.zeros(n, dtype=bool)
    #     trunc[-1] = np.random.rand() < 0.5
    #     logpa = np.random.rand(n)
    #     return VanillaReplayBuffer(
    #         obs=x1s.tolist(),
    #         act=acts.tolist(),
    #         rew=rews.tolist(),
    #         obs_next=x2s.tolist(),
    #         terminated=term.tolist(),
    #         truncated=trunc.tolist(),
    #         act_log_prob=logpa.tolist(),
    #     )

    # np.set_printoptions(precision=4)

    # bufs = [_bufmaker() for _ in range(ntrajs)]
    # for i, buf in enumerate(bufs):
    #     print(f"buf{i} len={np.shape(buf.obs)}")
    # trajbuf = merge_trajs(bufs, action_dtype=np.int32)
    # for k, v in trajbuf.__dict__.items():
    #     print(f"{k} shape={np.shape(v)}, data={v}")
    # import pandas as pd

    # for i in range(ntrajs):
    #     src_term = np.hstack([0, bufs[i].terminated], dtype=int)
    #     src_trunc = np.hstack([0, bufs[i].truncated], dtype=int)
    #     logpa = trajbuf.act_log_prob[:, i, 0]
    #     dst_nterms = trajbuf.term[:, i, 0].astype(int)
    #     dst_trunc = trajbuf.trunc[:, i, 0].astype(int)
    #     msgbuf = []
    #     for t in range(dst_nterms.shape[0]):
    #         logpa_t = logpa[t] if t < logpa.shape[0] else None
    #         src_nterm_t = int(not src_term[t]) if t < src_term.shape[0] else None
    #         dst_nterm_t = int(dst_nterms[t])
    #         src_trunc_t = int(src_trunc[t]) if t < src_trunc.shape[0] else None
    #         dst_trunc_t = int(dst_trunc[t])
    #         assert src_nterm_t is None or src_nterm_t == dst_nterm_t
    #         assert src_trunc_t is None or src_trunc_t == dst_trunc_t
    #         msgbuf.append(
    #             [t, src_nterm_t, dst_nterm_t, src_trunc_t, dst_trunc_t, logpa_t]
    #         )
    #     df = pd.DataFrame(
    #         msgbuf, columns=["t", "nterm0", "nterm1", "trunc0", "trunc1", "logpa"]
    #     )
    #     print(
    #         "traj[{}]:\n{}".format(
    #             i, df.to_string(index=False, max_cols=None, max_rows=None)
    #         )
    #     )
