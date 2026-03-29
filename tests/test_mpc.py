from __future__ import annotations


def _setup():  # 确保项目根节点在 sys.path 中
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


_ROOT = _setup()


from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import time



from abc import ABC, abstractmethod
from typing import Any, Union
import torch
import torch.nn as nn
import numpy as np
from codes.utils.time_ext import Timer_Context

DeviceLike = Union[torch.device, str, int]
TensorLike = Union[np.ndarray, torch.Tensor]


class NNModule(nn.Module):
    pass

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype


from typing import Sequence, Union
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# DeviceLike = Union[torch.device, str, int]


class BaseDynamicsModel(NNModule, ABC):

    # @abstractmethod
    def reset(self) -> tuple[torch.Tensor, ...]:
        """-> initial_state, ..."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        discrete time dynamics without memory
        (X_t,U_t)->(X_{t+1}, R_t=r(X_t,U_t,X_{t+1}), ...)
        Args:
            state: (...,dimX)
            action: (...,dimU)
        Returns:
            next_state: (...,dimX)
            reward: (...,1)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimX(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimU(self) -> int:
        raise NotImplementedError


class iLQR(NNModule):
    """iLQR algorithm"""

    DEBUG = False

    def __init__(
        self,
        model: BaseDynamicsModel,  # forward dynamics model
        horizon: int = 10,  # horizon
        lr=0.1,
        group_shape: Sequence[int] | int = (),
        model_real: nn.Module | None = None,  # real dynamics model
        multi_shooting: bool = False,  # whether to use multi-shooting
        vf: nn.Module | None = None,  # terminal value function (max)
        max_grad: float | None = None,  # max grad norm for actor and critic
        use_Ubar_as_var=True,  # 1:classic iLQR 2: optimize Ubar from linear policy
        gamma: float = 1.0,
        state_contraint_weight: float = 1.0,
    ):
        super().__init__()
        self.horizon = horizon = int(horizon)
        assert horizon > 0, ("horizon must be a positive integer", horizon)
        self.model = model
        self.dimX = dimX = int(model.dimX)
        self.dimU = dimU = int(model.dimU)
        group_shape = tuple(np.ravel(group_shape))
        self.group_shape: tuple[int, ...] = group_shape
        self.multi_shooting = multi_shooting
        self.use_Ubar_as_var = use_Ubar_as_var  # use Ubar as varable or only output
        self.use_Xbar_as_var = (
            multi_shooting  # whether to use Xbar as varable or only output
        )
        assert gamma >= 0 and gamma <= 1, ("gamma should be in (0,1]", gamma)
        self.gamma = gamma
        self._max_grad = max_grad
        if max_grad is not None:
            assert max_grad > 0, ("max_grad must be a positive float", max_grad)
        self._w_x = state_contraint_weight
        assert np.isfinite(state_contraint_weight) and state_contraint_weight >= 0, (
            "state_contraint_weight should be a non-negative finite float",
            state_contraint_weight,
        )

        # self.K = nn.Linear(dimX, dimU)
        self.K = MLP_(
            dimX,
            dimU,
            (128, 128),
            use_residual=True,
        )
        self.vf = vf
        self.lr = lr

        self.var_Xbar = nn.Parameter(
            torch.randn(group_shape + (horizon + 1, dimX)),
        )  # (x_0,...,x_N), shape=(...,N+1,dimX)
        self.var_Ubar = nn.Parameter(
            torch.randn(group_shape + (horizon, dimU)),
        )  # (u_0,...,u_N-1), shape=(...,N,dimU)
        self.param_x0 = nn.Parameter(
            torch.zeros(group_shape + (dimX,), requires_grad=False)
        )  # (x_0), shape=(...,dimX)
        self._gammas = nn.Parameter(
            gamma
            ** torch.arange(horizon + 1).reshape(
                (1,) * len(group_shape) + (horizon + 1, 1)
            ),
            requires_grad=False,
        )
        r"""(gamma^k)_{0\leq k\leq N} shape=(...,N+1,1)"""

        opt_params: list[dict[str, Any]] = [
            {"params": self.K.parameters(), "lr": lr, "weight_decay": 1e-4},
        ]
        if self.use_Ubar_as_var:
            opt_params.append({"params": self.var_Ubar, "lr": 1.0})
        if self.use_Xbar_as_var:
            opt_params.append({"params": self.var_Xbar, "lr": 1.0})
        self.actor_optim = optim.Adam(opt_params, self.lr, weight_decay=1e-4)
        if vf is not None:
            self.critic_optim = optim.Adam(vf.parameters(), lr=lr, weight_decay=1e-4)
        self.train()

    @torch.no_grad()
    def reset(self):
        self.param_x0.zero_()
        self.var_Xbar.zero_()

    def _multi_shoot(self, reset_controls=False):
        # self.var_Xbar.data[..., 0, :] = self.param_x0.data
        xs_tgt = self.var_Xbar.detach()
        x1t = xs_tgt[..., :-1, :]  # (...,N,dimX)

        # u1 = self.K(x1t)  # (...,N,dimU)
        if reset_controls:
            u1 = self.K(x1t)  # (...,N,dimU)
            self.var_Ubar.data.copy_(u1)
        else:
            u1 = self.var_Ubar

        rst = self.model.step(x1t, u1)
        x2p = rst[0]  # (...,N,dimX)
        r1 = rst[1]  # (...,N,1)
        xf = x2p[..., -1, :]  # (...,dimX)
        x2t = x2p.detach()

        gammas = self._gammas[..., : self.horizon, :]
        r1 = r1 * gammas
        G0 = r1.sum(dim=-2)
        cost_constrain = None
        if self.use_Xbar_as_var:
            cost_constrain = 0
            cost_trans = F.huber_loss(self.var_Xbar[..., 1:, :], x2t, reduction="sum")
            cost_constrain = cost_constrain + cost_trans * self._w_x

            cost_x0 = F.huber_loss(
                self.var_Xbar[..., 0, :], self.param_x0, reduction="sum"
            )
            cost_constrain = cost_constrain + cost_x0 * self._w_x
        return G0, xf, cost_constrain

    def _single_shoot(self, reset_controls=True):
        g0 = None
        x1 = self.param_x0
        for k in range(self.horizon):
            self.var_Xbar.data[..., k, :] = x1
            if reset_controls:
                u1 = self(x1)
                self.var_Ubar.data[..., k, :] = u1
            else:
                u1 = self.var_Ubar[..., k, :]

            rst = self.model.step(x1, u1)
            x2 = rst[0]  # (...,dimX)
            r1 = rst[1]  # (...,1)
            if g0 is None:
                g0 = r1
            else:
                gammak = self._gammas[..., k, :]
                g0 = g0 + r1 * gammak
            x1 = x2

        xf = x1
        self.var_Xbar.data[..., -1, :] = xf
        assert g0 is not None
        return g0, xf

    def warm_start(self, x0: torch.Tensor, epochs=100):
        """initialize K, b, xbar with given data"""
        self._set_x0(x0)
        with torch.no_grad():  # set initial sol
            self._single_shoot(reset_controls=True)
        self.step(x0, epochs=epochs)

    @torch.no_grad()
    def _set_x0(self, x0: torch.Tensor):
        tgt_shape = self.group_shape + (self.dimX,)
        assert x0.shape == tgt_shape, (
            f"expected x0.shape == {tgt_shape}, got",
            x0.shape,
        )
        self.param_x0.data.copy_(x0)
        self.var_Xbar.data[..., 0, :] = x0
        return

    def forward(self, state: TensorLike):
        """infer control without memory"""
        x = torch.as_tensor(state, device=self.device, dtype=self.dtype)  # (...,dimX)
        u: torch.Tensor = self.K(x)  # (...,dimU)
        # u = torch.tanh(u)

        clip_u = False
        if clip_u:
            umax = 1.0
            u = torch.tanh(u)
            u = u * umax
            # unorm = u.abs()  # with grad
            # with torch.no_grad():
            #     rfix = unorm / umax
            #     rfix = 1 / torch.where(rfix > 1, rfix, 1)
            # u = rfix * u
        # u = torch.clamp(u, -umax, umax)
        return u

    @torch.no_grad()
    def shift(self):
        """receeding horizon"""
        xn = self.var_Xbar[..., -1, :].clone()
        un = self(xn)
        xf = self.model.step(xn, un)[0]
        self.var_Xbar.data[..., :-1, :] = self.var_Xbar.data[..., 1:, :].clone()
        self.var_Xbar.data[..., -1, :] = xf
        self.var_Ubar.data[..., :-1, :] = self.var_Ubar.data[..., 1:, :].clone()
        self.var_Ubar.data[..., -1, :] = un

    def _mpc_vars(self):
        for p in [self.var_Xbar, self.var_Ubar]:
            yield p

    def _update_once(self):
        cost_constrain = None
        # TODO:
        # - limit control
        reset_controls = True
        if self.multi_shooting:
            G0, xf, cost_constrain = self._multi_shoot(reset_controls=reset_controls)
        else:
            G0, xf = self._single_shoot(reset_controls=reset_controls)

        # terminal cost
        # assert G0 is not None
        if self.vf is not None:
            vfinal: torch.Tensor = self.vf(xf)  # (...,1)
            G0 = G0 + vfinal * self._gammas[..., -1, :]

        actor_loss = -G0.mean()
        if cost_constrain is not None:
            actor_loss = actor_loss + cost_constrain

        if self.use_Ubar_as_var:
            x1 = self.var_Xbar[..., :-1, :].detach()
            u1t = self.var_Ubar.detach()
            u1p = self.K(x1)
            guide_loss = F.huber_loss(u1p, u1t, reduction="sum")
            actor_loss = actor_loss + guide_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self._max_grad is not None:
            for i_pg, pg in enumerate(self.actor_optim.param_groups):
                ps = pg["params"]
                for i_p, prm in enumerate(ps):
                    if prm.grad is not None:
                        nn.utils.clip_grad_value_(prm, self._max_grad)
                    else:
                        prm
        self.actor_optim.step()
        return actor_loss.item(), G0.detach()

    def step(self, x0: torch.Tensor, epochs=10):
        """
        set new state as initial state and planning
        """
        x0 = torch.as_tensor(x0, device=self.device, dtype=self.dtype)  # (...,dimX)
        self._set_x0(x0)

        # optimize
        for epoch in range(epochs):
            if self.multi_shooting and not self.use_Xbar_as_var:
                with torch.no_grad():
                    self._single_shoot()
            actor_loss_, G0t = self._update_once()

        assert isinstance(G0t, torch.Tensor)

        if self.vf is not None:
            critic_loss = F.huber_loss(self.vf(self.param_x0), G0t, reduction="mean")
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            critic_loss_ = critic_loss.item()
        else:
            critic_loss_ = 0
        u0 = self.var_Ubar[..., 0, :]
        G0t_ = G0t.mean().item()
        return u0, G0t_, actor_loss_, critic_loss_


class BallModel(BaseDynamicsModel):
    """3DOF ball with acceleration"""

    def __init__(self, group_shape: Sequence[int] | int = (), dt=1e-2):
        super().__init__()
        self.group_shape: tuple[int, ...] = tuple(np.ravel(group_shape))
        group_shape = self.group_shape

        dimU = 2
        dimX = 2 * dimU
        self._dt = dt

        self.__dimX = dimX
        self.__dimU = dimU
        self.Q = nn.Linear(dimX, dimX, bias=False)
        self.R = nn.Linear(dimU, dimU, bias=False)

        wQsqrt = torch.eye(dimX)
        wQsqrt[torch.arange(dimU, dimX), torch.arange(dimU, dimX)] = dt**2
        self.Q.weight.data.copy_(wQsqrt.T @ wQsqrt)
        wRsqrt = torch.eye(dimU) * 1e-2
        self.R.weight.data.copy_(wRsqrt.T @ wRsqrt)

        self.eval()

    @property
    def dimX(self) -> int:
        return self.__dimX

    @property
    def dimU(self) -> int:
        return self.__dimU

    def step(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        assert state.shape[-1] == self.dimX
        assert action.shape[-1] == self.dimU
        x2 = self(state, action)
        c1 = self.get_cost(state, action, x2)
        return x2, -self._dt * c1

    def _dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        u = action
        dimU = self.dimU
        x2 = state[..., dimU:]
        dx1 = x2
        dx2 = u
        dX = torch.cat([dx1, dx2], dim=-1)
        return dX

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._dt
        k1 = self._dynamics(state, action)
        k2 = self._dynamics(state + (0.5 * h) * k1, action)
        k3 = self._dynamics(state + (0.5 * h) * k2, action)
        k4 = self._dynamics(state + h * k3, action)
        x2 = state + (k1 + 2 * k2 + 2 * k3 + k4) * (h / 6)
        return x2

    def get_cost(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor
    ) -> torch.Tensor:
        xQx = (state * self.Q(state)).sum(dim=-1, keepdim=True)
        uRu = (action * self.R(action)).sum(dim=-1, keepdim=True)
        cost = xQx + uRu  # (...,1)
        return cost


class MLP_(NNModule):
    def __init__(
        self, din: int, dout: int, hidden_size: Sequence[int] = (), use_residual=False
    ):
        super().__init__()
        layers = []
        hidden_size = (din,) + tuple(np.ravel(hidden_size))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            if i < len(hidden_size) - 2:
                layers.append(nn.ReLU())
            else:
                # layers.append(nn.Tanh())
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[-1], dout))
        self._kern = nn.Sequential(*layers)
        if use_residual and len(hidden_size) > 1:
            self._lin = nn.Linear(din, dout, bias=False)
        else:
            self._lin = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self._kern(state)
        if self._lin is not None:
            res = self._lin(state)
            x = x + res
        return x


def ps2xlim(ps: Sequence[float | Sequence | np.ndarray], ratio=1.1, eps=1e-2):
    ps = [np.ravel(p) for p in ps]
    ps_ = np.concatenate(ps)
    xmin = np.min(ps_)
    xmax = np.max(ps_)
    mid = 0.5 * (xmin + xmax)
    r = 0.5 * (xmax - xmin) * ratio + eps
    return mid - r, mid + r


def main():
    dtype = torch.float32
    use_cuda = True
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu"
    )
    dt = 0.050
    lr = min(1 / dt * 1e-1, 0.5)
    max_grad = 10.0
    horizon = 10
    update_freq_per_step = 10  # step-wise optimization
    model_sim = BallModel(dt=dt)
    model_real = deepcopy(model_sim).to(device=device, dtype=dtype)
    group_shape = (1,)
    max_steps = 100
    use_vf = False
    pretrn_fname = ""

    task_dir = _ROOT / "tmp" / "mpc_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

    _model_ver = 0

    def model_save(model: nn.Module):
        nonlocal _model_ver
        _model_ver += 1
        fn = task_dir / f"model_{_model_ver}.pth"
        fn.parent.mkdir(exist_ok=True, parents=True)
        try:
            torch.save(model.state_dict(), fn)
            print(f"model>>{fn}")
        except Exception as e:
            print(f"failed to save model to {fn}: {e}")

    def model_load(model: nn.Module, fn: str):
        try:
            model.load_state_dict(torch.load(fn, map_location=device))
            print(f"model<<{fn}")
        except Exception as e:
            print(f"failed to load model from {fn}: {e}")

    if use_vf:
        vf = MLP_(model_sim.dimX, 1, (128, 128))
    else:
        vf = None

    tmr_sim = Timer_Context("sim")
    tmr_react = Timer_Context("react")
    tmr_plan = Timer_Context("plan")
    _tmrs = [tmr_react, tmr_sim, tmr_plan]

    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for multi_shooting in [
        # True,
        False,
    ]:
        policy = iLQR(
            model_sim,
            vf=vf,
            horizon=horizon,
            lr=lr,
            group_shape=group_shape,
            multi_shooting=multi_shooting,
            max_grad=max_grad,
            gamma=0.99,
            state_contraint_weight=1e-3,
            use_Ubar_as_var=True,
        )
        policy.to(device=device, dtype=dtype)
        if Path(pretrn_fname).exists():
            model_load(policy, pretrn_fname)

        print(f"multi_shooting={multi_shooting}")

        ax1.clear()
        ax2.clear()
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title(f"multi_shooting={multi_shooting}\nhorizon={policy.horizon}")
        ax2.set_xlabel("t")
        ax2.set_ylabel("cost")
        traj_real = ax1.plot([], [], "b-", marker="o", label="real")[0]
        traj_pred = ax1.plot(
            [], [], color="gray", marker=".", linestyle="--", alpha=0.5, label="pred"
        )[0]
        target = ax1.plot([0], [0], "r*", label="target")[0]
        ax1.legend()
        costline_actor = ax2.plot([], [], "g-", label="actor_loss")[0]
        costline_critic = ax2.plot([], [], "b-", label="critic_loss")[0]
        costline_G0 = ax2.plot([], [], "r-", label="G0")[0]
        ax2.legend()
        list_actor_loss: deque[float] = deque(maxlen=policy.horizon)
        list_critic_loss: deque[float] = deque(maxlen=policy.horizon)
        list_G0: deque[float] = deque(maxlen=policy.horizon)
        _ques = [list_actor_loss, list_critic_loss, list_G0]

        for itr in range(100):
            for _tmr in _tmrs:
                _tmr.reset()
            for q in _ques:
                q.clear()

            x0 = (
                torch.randn(group_shape + (model_sim.dimX,), device=device, dtype=dtype)
                * 100.0
            )
            with tmr_react:
                policy.warm_start(x0)
            for k in range(max_steps):
                with tmr_react:
                    u0 = policy(x0)

                with tmr_plan:
                    # if k > 0:
                    #     policy.shift()
                    u0, G0, actor_loss, critic_loss = policy.step(
                        x0, epochs=update_freq_per_step
                    )
                    u0 = u0.detach().clone()
                    policy.shift()

                with tmr_sim:
                    with torch.no_grad():
                        x1: torch.Tensor = model_real(x0, u0)
                    x0 = x1

                list_actor_loss.append(actor_loss)
                list_critic_loss.append(critic_loss)
                list_G0.append(G0)
                print(
                    f"step {k-1}, loss={actor_loss:.3g}, {critic_loss:.3g}",
                    *[f"{_tmr.name}={_tmr.t/(k+1):.3f}s" for _tmr in _tmrs],
                )
                _idx = [0] * len(group_shape)
                pos_pred = (
                    policy.var_Xbar[*_idx, :-1, :2].detach().cpu().numpy()
                )  # (N+1,2)
                pos_real = x0[*_idx, :2].detach().cpu().numpy()  # (2,)
                ts = np.arange(max(0, k + 1 - policy.horizon), k + 1)
                _xs = pos_pred[:, 0]
                _ys = pos_pred[:, 1]
                plt.ioff()

                traj_pred.set_data(_xs, _ys)
                traj_real.set_data(pos_real[0:1], pos_real[1:2])

                costline_actor.set_data(ts, np.asarray(list_actor_loss))
                costline_critic.set_data(ts, np.asarray(list_critic_loss))
                costline_G0.set_data(ts, np.asarray(list_G0))
                ax1.set_xlim(ps2xlim([_xs, 0, pos_real[0:1]]))
                ax1.set_ylim(ps2xlim([_ys, 0, pos_real[1:2]]))
                ax2.set_xlim(ps2xlim([ts]))
                ax2.set_ylim(ps2xlim([list_actor_loss, list_critic_loss, list_G0]))
                fig.canvas.draw()
                plt.pause(0.010)
                plt.ion()

            model_save(policy)

    plt.show(block=True)


if __name__ == "__main__":
    main()
