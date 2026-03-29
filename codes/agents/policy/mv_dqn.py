from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union, cast
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from .proto4policy import *
from .utils import calc_gae, to_torch, to_numpy, to_torch_as
from torch.nn.utils import clip_grad_value_

# from ..modules.utils import MLP
_DEBUG = True

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter
    from torch import nn
    from .proto4policy import BaseReplayBuffer as BaseReplayBuffer

from tianshou.policy import DQNPolicy as _DQNPolicy

DQNStateType = Union[BatchProtocol[torch.Tensor], torch.Tensor, None]
T_DQNState = TypeVar("T_DQNState", bound=DQNStateType)


def get_qa(qs: torch.Tensor, a: torch.Tensor):
    """
    select the q value q(s,a) of a given action a

    Args:
        q_s (torch.Tensor):  shape=(...,dimV,|A|)
        a (torch.Tensor):   shape=(...,)|(...,1|dimV)
    Returns:
        qa (torch.Tensor): q(s,a),shape=(...,dimV,1)
    """
    if a.ndim < qs.ndim:
        a = a.view(a.shape + (1,) * (qs.ndim - a.ndim))  # (...,1,1)
    a = a.broadcast_to(qs.shape[:-1] + (1,))  # (...,dimV,1)
    assert qs.shape[:-1] == a.shape[: qs.ndim - 1], (
        "expect same shape head",
        qs.shape[:-1],
        "got a",
        a.shape[: qs.ndim - 1],
    )
    qa = qs.gather(-1, a)  # (...,dimV,1)
    return qa


@dataclass(kw_only=True)
class MVDQNTrainingStats(TrainingStats):
    loss_total: float
    q_loss_total: float
    v_loss_total: float

    q_target_total: float
    v_target_total: float

    v_pred_total: float
    q_pred_total: float

    q_loss_entries: Sequence[float]
    v_loss_entries: Sequence[float]

    v_target_entries: Sequence[float]
    v_pred_entries: Sequence[float]
    q_target_entries: Sequence[float]
    q_pred_entries: Sequence[float]

    lr: float


@dataclass(kw_only=True)
class MVDQNObsBatchProtocol(ObsBatchProtocol[TArr_co]):
    mask: TArr_co | None = None


@dataclass(kw_only=True)
class MVDQNModelOutProtocol(ModelOutputBatchProtocol[torch.Tensor]):
    mqs: torch.Tensor  # (...,B,dimV,|A|), raw output of Qnet
    mvs: torch.Tensor  # (...,B,dimV,1), raw output of Vnet


@dataclass(kw_only=True)
class MVDQNLearnInputProtocol(
    MVDQNObsBatchProtocol[torch.Tensor],
    RolloutBatchProtocol[torch.Tensor],
    MVDQNModelOutProtocol,
):
    # trunc_f: torch.Tensor  # (...,B,1), truncated flag
    pass


class MV_DQNPolicy(BaseNNPolicy[MVDQNTrainingStats], Generic[T_DQNState]):

    name = "MV_DQNPolicy"

    def __init__(
        self,
        *,
        model: NetBase[T_DQNState],  # Qnet&Vnet
        optim: torch.optim.Optimizer,
        action_space: spaces.Discrete,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        clip_loss_grad: bool = False,
        observation_space: spaces.Space | None = None,
        lr_scheduler: TLearningRateScheduler | None = None,
        use_multi_reward: bool = False,
        values_dim=1,
        values_weight: Any | None = None,
        gae_lambda: float = 1.0,
        name: str = name,
        dtype: torch.dtype = torch.float32,
        device: DeviceLikeType = "cpu",
        greedy_eps: float = 0.1,
        grad_max: float = 0.0,
    ):
        self.set_name(name)
        values_dim = int(values_dim)
        assert values_dim > 0, ("expect values_dim > 0,got", values_dim)
        self._V_dim = values_dim
        self._V_weight = values_weight
        self._use_multi_reward = use_multi_reward
        if not self._use_multi_reward:
            assert values_dim == 1, (
                "expect values_dim==1 when use_multi_reward is False,got",
                values_dim,
            )
        self.gae_lambda = float(gae_lambda)
        assert (
            0 <= gae_lambda <= 1
        ), f"gae_lambda should be in [0,1],but got {gae_lambda}"
        self.max_action_num = int(action_space.n)
        assert (
            greedy_eps >= 0 and greedy_eps <= 1
        ), f"greedy_eps should be in [0,1],but got {greedy_eps}"
        self.greedy_eps = greedy_eps

        _DQNPolicy.__init__
        assert isinstance(
            observation_space, spaces.Box
        ), "DQN only supports Box observation space"
        assert isinstance(
            action_space, spaces.Discrete
        ), "DQN only supports Discrete action space"
        self.dtype = dtype
        assert isinstance(dtype, torch.dtype), f"expect torch.dtype,got {dtype}"
        self.device = torch.device(device)

        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
            lr_scheduler=lr_scheduler,
        )

        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert (
            0.0 <= discount_factor <= 1.0
        ), f"discount factor should be in [0,1] but got: {discount_factor}"
        self.gamma = discount_factor
        assert (
            estimation_step > 0
        ), f"estimation_step should be greater than 0 but got: {estimation_step}"
        self.n_step = estimation_step
        self._target = target_update_freq > 0
        self.freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self.rew_norm = reward_normalization
        self.is_double = is_double
        self.clip_loss_grad = clip_loss_grad
        self.grad_max = grad_max

    def _logits2mqvs(self, logits: torch.Tensor):
        """
        Compute the q&v value based on the network's raw output.
        Args:
            logits: the network's raw output,shape=(...,dimV*|A|)
        Returns:
            mqs: the q values {q_i(s,a)}_{i,a} ,shape=(...,dimV,|A|)
            mvs: the v values {v_i(s)}_{i}     ,shape=(...,dimV,1)
        """
        sizeA = self.max_action_num
        dimV = self._V_dim
        assert logits.shape[-1] == (1 + sizeA) * dimV, (
            f"expect logits.shape[-1] == ({sizeA}+1)*{dimV},got",
            logits.shape[-1],
        )
        mqvs = torch.unflatten(logits, -1, (dimV, sizeA + 1))  # (...,dimV,1+|A|)
        mvs = mqvs[..., 0:1]  # (...,dimV,1)
        mqs = mqvs[..., 1:]  # (...,dimV,|A|)
        return mqs, mvs

    def _fuse_mvs(self, mqs: torch.Tensor):
        """
        fuse multiple q values
        Args:
        """
        assert mqs.shape[-2] == self._V_dim, (
            f"expect mqs.shape[-2] == {self._V_dim},got",
            mqs.shape[-2],
        )
        if self._V_weight is None:
            qs = mqs.mean(dim=-2)  # (...,B,|A|)
        else:  # TODO: convex combination
            raise NotImplementedError(
                "dynamic weighted descision is not implemented yet"
            )
        return qs

    def _fix_q_with_mask(self, logits: torch.Tensor, mask: np.ndarray | None):
        """
        fix the q value with action mask.
        Args:
            logits: q values,shape=(...,|A|)
            mask: feasible action mask,shape=(...,|A|),1->valid,0->invalid

            NOTE:use mask to rectify target q for VI or search
        Returns:
            the q values,shape=(...,dimV,|A|)
        """
        if mask is not None:
            with torch.no_grad():
                msk = to_torch_as(1 - mask, logits)  # (...,|A|)
                assert msk.shape[-1] == logits.shape[-1], (
                    f"expect msk.shape[-1] == logits.shape[-1],got",
                    msk.shape[-1],
                    logits.shape[-1],
                )
                if msk.ndim < logits.ndim:
                    msk = msk.reshape(
                        msk.shape[:-1]
                        + ((1,) * (logits.ndim - msk.ndim))
                        + msk.shape[-1:]
                    )  # (...,|A|)

                # faster than torch.where
                _fix = (logits.min() - 1.0) - logits
                _fix = msk * _fix
            logits = logits + _fix
        return logits

    def forward(
        self,
        batch: MVDQNObsBatchProtocol[torch.Tensor],
        state: T_DQNState | None = None,
        model: Literal["model", "model_old"] = "model",
        greedy_eps: float | None = None,
        **kwargs: Any,
    ) -> MVDQNModelOutProtocol:
        # _DQNPolicy.forward
        greedy_eps = greedy_eps or self.greedy_eps
        assert model in [
            "model",
            "model_old",
        ], f"expect model in ['model','model_old'],got {model}"
        kern_ = self.model if model == "model" else self.model_old
        obsB = batch.obs
        info = getattr(batch, "info", None)
        # TODO: this is convoluted! See also other places where this is done.
        if isinstance(obsB, TArr):
            obs = obsB
        else:
            obs = getattr(obsB, "obs", obsB)
        assert isinstance(obs, torch.Tensor), f"expect obs be Tensor,got {type(obs)}"
        # obs_ = to_torch(obs_,dtype=self.dtype,device=self.device)
        qv_logits, _c = kern_(
            obs,
            state=state,
            info=info,
        )
        #
        qv_logits: torch.Tensor  # (...,B,dimV*(1+|A|))
        _c: torch.Tensor | None  # (...,B,dimH)
        # assert qs_raw_BA.shape[-1] == self.max_action_num * self._V_dim,(
        #     f"expect action_values_BA.shape[-1] == {self.max_action_num*self._V_dim},got",
        #     qs_raw_BA.shape[-1],
        # )
        mqs, mvs = self._logits2mqvs(qv_logits)  # (...,B,dimV,|A|)
        qs = self._fuse_mvs(mqs)  # (...,B,|A|)
        #
        mask = getattr(batch, "mask", None)
        mask = getattr(obsB, "mask", mask)
        sizeA = self.max_action_num
        with torch.no_grad():
            qs_ = self._fix_q_with_mask(qs, mask)  # (...,B,|A|)
            assert qs_.shape[-1] == sizeA, (
                f"expect q.shape[-1] == {sizeA},got",
                qs_.shape[-1],
            )
            act = qs_.argmax(dim=-1)  # (...,B)
            if greedy_eps > 0:
                act_r = torch.randint_like(act, high=sizeA)
                _tag = (
                    torch.rand(act.shape, device=self.device, dtype=self.dtype)
                    <= greedy_eps
                )
                act = torch.where(_tag, act_r, act)  # (...,B)
        if _DEBUG:
            assert torch.isfinite(obs).all(), "obs contains nan or inf"
            assert torch.isfinite(qv_logits).all(), "qv_logits contains nan or inf"
        result = Batch(
            logits=qv_logits,
            act=act,
            state=_c,
            mqs=mqs,
            mvs=mvs,
        )
        return cast(MVDQNModelOutProtocol, result)

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def learn(
        self,
        batch: MVDQNLearnInputProtocol,
    ) -> TrainingStats:
        # _DQNPolicy.learn
        if self._target and self._iter % self.freq == 0:
            self.sync_weight()

        self.optim.zero_grad()
        weight: float | torch.Tensor = batch.pop("weight", 1.0)
        act_old = batch.act  # (...,B)

        # pred
        mout: MVDQNModelOutProtocol = self(batch, greedy_eps=0.0)
        qs = mout.mqs  # (...,B,dimV,|A|)

        qa = get_qa(qs, act_old)  # (...,B,dimV,1)
        qa = qa.squeeze(-1)  # (...,B,dimV)
        dimV = self._V_dim
        assert qa.shape[-1] == dimV, (
            f"expect q.shape[-1] == {dimV},got",
            qa.shape[-1],
        )
        qtarg = batch.returns.reshape(qa.shape)
        vtarg = qs.max(dim=-1).values  # (...,B,dimV)

        vs = mout.mvs  # (...,B,dimV,1)
        vs = vs.squeeze(-1)  # (...,B,dimV)
        assert vs.shape[-1] == dimV, (
            f"expect vs.shape[-1] == {dimV},got",
            vs.shape[-1],
        )

        # term1 = batch.terminated  # (...,B,1)
        trunc1 = batch.truncated  # (...,B,1)
        # done1 = term1 | trunc1
        done1 = trunc1
        msk1 = 1 - done1.type(qa.dtype)
        assert msk1.shape[:-1] == qa.shape[:-1], (
            f"expect msk1.shape[:-1] == {qa.shape[:-1]},got",
            msk1.shape[:-1],
        )
        qa = qa * msk1  # (...,B,dimV)
        vs = vs * msk1  # (...,B,dimV)
        qtarg = qtarg * msk1  # (...,B,dimV)
        vtarg = vtarg * msk1  # (...,B,dimV)
        tde_q = qtarg - qa
        tde_v = vtarg - vs

        qp = qa.reshape(-1, dimV)  # (:,dimV)
        qt = qtarg.reshape(-1, dimV)  # (:,dimV)
        vp = vs.reshape(-1, dimV)  # (:,dimV)
        vt = qtarg.reshape(-1, dimV)  # (:,dimV)
        if self.clip_loss_grad:
            nn.HuberLoss
            loss_q = F.huber_loss(qp, qt, reduction="none")
            loss_v = F.huber_loss(vp, vt, reduction="none")
        else:
            loss_q = (tde_q.square() * weight).reshape(-1, dimV)
            loss_v = (tde_v.square() * weight).reshape(-1, dimV)

        loss = loss_q.mean() + loss_v.mean()
        loss.backward()
        if self.grad_max > 0:
            for _pg in self.optim.param_groups:
                clip_grad_value_(_pg["params"], self.grad_max)
        self.optim.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            assert isinstance(self.lr_scheduler, torch.optim.lr_scheduler.LRScheduler)
            lr = self.lr_scheduler.get_last_lr()[0]
        else:
            lr = float(self.optim.param_groups[0]["lr"])
        batch.weight = tde_q.detach()  # prio-buffer
        self._iter += 1

        with torch.no_grad():
            stat = MVDQNTrainingStats(
                loss_total=float(loss.item()),
                q_loss_total=float(loss_q.mean().item()),
                v_loss_total=float(loss_v.mean().item()),
                #
                v_target_total=float(vt.mean().item()),
                q_target_total=float(qt.mean().item()),
                q_pred_total=float(qp.mean().item()),
                v_pred_total=float(vp.mean().item()),
                #
                q_loss_entries=loss_q.mean(0).cpu().tolist(),
                v_loss_entries=loss_v.mean(0).cpu().tolist(),
                q_target_entries=qt.mean(0).cpu().tolist(),
                v_target_entries=vt.mean(0).cpu().tolist(),
                q_pred_entries=qp.mean(0).cpu().tolist(),
                v_pred_entries=vp.mean(0).cpu().tolist(),
                #
                lr=lr,
            )
        return stat

    @torch.no_grad()
    def process_fn(
        self,
        rollout_batch: RolloutBatchProtocol[TArr],  # can be changed in-place
        buffer: BaseReplayBuffer,
        indices: np.ndarray,
    ) -> MVDQNLearnInputProtocol:
        """
        NOTE:
        用于 learning 预处理
        只对 batch 做处理,不依赖变量 buffer,indices (加他们是为了和接口一致)
        - 统一将数据转到 torch 设备
        - 基于 RETraj&critic 算出 rollout/lambda-return
        - 使用 term&trunc

        Args:
            rollout_batch (Union[Batch,RolloutBatchProtocol]): _description_
            buffer (ReplayBuffer): _description_
            indices (np.ndarray): _description_

        Returns:
            RolloutBatchProtocol: _description_
        """
        _DQNPolicy.process_fn
        dtype = self.dtype
        device = self.device

        obs_all = rollout_batch.obs  # (T+1,B,dimO)
        assert isinstance(obs_all, (torch.Tensor, np.ndarray)), (
            "obs type error",
            type(obs_all),
        )
        assert len(obs_all.shape) == 3, (
            "obs should be a 3-dim tensor",
            tuple(obs_all.shape),
        )
        T = obs_all.shape[0] - 1
        B = obs_all.shape[1]
        assert T >= 0, ("expected len(obs)>=1,got", len(obs_all))

        term = to_torch(
            rollout_batch.terminated, dtype=torch.bool, device=device
        )  # (T+1,B)
        assert term.shape[0] == T + 1, (
            f"expected len(terminated)=={T+1},got",
            term.shape[0],
        )
        term = term.view((T + 1, B, 1))
        term_f = term.type(dtype)
        mskf_term = 1 - term_f

        trunc = to_torch(
            rollout_batch.truncated, dtype=torch.bool, device=device
        )  # (T+1,B)
        assert trunc.shape[0] == T + 1, (
            f"expected len(truncated)=={T+1},got",
            trunc.shape[0],
        )
        trunc = trunc.view((T + 1, B, 1))
        trunc1_f = trunc[:T].type(dtype)

        rews = getattr(rollout_batch, "reward_components", None)  # (T,B,dimV)
        if rews is None:
            rews = rollout_batch.rew
        dimV = self._V_dim
        assert rews.shape == (T, B, dimV), (
            f"expect rews.shape == {(T,B,dimV)},got",
            tuple(rews.shape),
        )
        rews = to_torch(rews, dtype=dtype, device=device)  # (T,B,dimV)
        if _DEBUG:
            assert torch.isfinite(rews).all(), "reward contains nan or inf"

        obs_all = to_torch(obs_all, dtype=dtype, device=device)
        act_old = to_torch(rollout_batch.act, dtype=torch.int64, device=device)  # (T,B)
        act_old = act_old.view((T, B, 1))

        _c = getattr(rollout_batch, "state", None)  # (T,B,dimH)
        assert _c is None, NotImplementedError("hidden state is not supported yet")

        rollout_batch.obs = obs_all  # (T+1,B,dimO)
        mout: MVDQNModelOutProtocol = self(
            rollout_batch,
            state=_c,
            model="model_old" if self._target else "model",
            greedy_eps=0.0,
        )
        # act_opt = mout.act  # (T+1,B)
        mvs = mout.mvs.reshape((T + 1, B, dimV))  # (T+1,B,dimV,1)
        mvs = mskf_term * mvs  # term(t,s)=>v(t,s)=0

        mv1 = mvs[:-1]  # (T,B,dimV,1)
        mv2 = mvs[1:]  # (T,B,dimV,1)
        gae = calc_gae(
            v_s=mv1,
            v_s_=mv2,
            rew=rews,
            truncated=trunc1_f,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )  # (T,B,dimV),trunc(t,s) => gae(t,s)=0
        ret = mv1 + gae
        if _DEBUG:
            assert torch.isfinite(gae).all(), "gae contains nan or inf"
            assert torch.isfinite(ret).all(), "returns contains nan or inf"
        rollout_batch.obs = obs_all[:T]
        rollout_batch.returns = ret
        rollout_batch.act = act_old
        rollout_batch.rew = rews
        rollout_batch.terminated = term[:T]
        rollout_batch.truncated = trunc[:T]
        rollout_batch = cast(MVDQNLearnInputProtocol, rollout_batch)
        # rollout_batch.trunc_f = trunc1_f
        return rollout_batch

    def write_stats(self, stat: MVDQNTrainingStats, writer: SummaryWriter, step: int):
        head = "critic"
        for k, v in stat.__dict__.items():
            if isinstance(v, (float, int)):
                writer.add_scalar(f"{head}/{k}", v, step)
            elif isinstance(v, (list, tuple)):
                for i, vi in enumerate(v):
                    writer.add_scalar(f"{head}/{k}/{i}", vi, step)
