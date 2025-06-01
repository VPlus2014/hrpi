# 250531
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence, List
from typing import Callable, Union, cast
import torch
from torch.distributions import Categorical, Distribution
from torch import Tensor


def _fpp(x: "Tensor", p=2):
    return torch.abs(x).pow(p) / p


def _centralize(x, mu, sig):
    return (x - mu) / sig


def _broadcastable(*sizes: int):
    assert len(sizes) > 0, "sizes should not be empty"
    sz_no = max(sizes)
    return all(sz in [1, sz_no] for sz in sizes)


def _kernel_rbf(x: "Tensor", mu: "Tensor", sig: "Tensor", p=2):
    r"""$exp(-\|(x-\mu)/sig\|_p^p)$"""
    x = _centralize(x, mu, sig).abs().pow(p)
    x = (-x).exp()
    return x


def _kernel_rational(x: "Tensor", mu: "Tensor", sig: "Tensor", p=2):
    r"""$\frac{1}{1+|(x-\mu)/sig|}$"""
    x = _centralize(x, mu, sig).abs().pow(p)
    x = 1 / (1 + x)
    return x


def kernel_gaussian(x: "Tensor", mu: "Tensor", sig: "Tensor"):
    return _kernel_rbf(x, mu, sig, p=2)


def kernel_laplacian(x: "Tensor", mu: "Tensor", sig: "Tensor"):
    return _kernel_rbf(x, mu, sig, p=1)


def kernal_rational_quadratic(x: "Tensor", mu: "Tensor", sig: "Tensor"):
    return _kernel_rational(x, mu, sig, p=2)


_KernelFuncType = Callable[[Tensor, Tensor, Tensor], Tensor]


def _single_discrete(
    mu: "Tensor",
    sig: "Tensor",
    asize: int,  # 状态空间大小
    kernel: "_KernelFuncType" = kernel_gaussian,
    p_lb=None,  # 每一维的概率质量下界, None表示不限制(容易引发数值异常)
):
    r"""
    离散分布的状态空间为 $A=\{0,1,...,|A|-1\}$
    概率计算规则:
    $$
        p(x)=k(x)/\sum_{y \in A} k(y) \\
        k(x)=kernel(x/(|A|-1); mu_i, sig_i), \forall x \in A
    $$
    """
    assert asize > 0, ("asize should be greater than 0, got", asize)
    assert len(mu.shape) == len(sig.shape), (
        "shape of mu and sig should be the same, got",
        mu.shape,
        sig.shape,
    )
    shphd = mu.shape  # (...,)
    mu = mu.unsqueeze(-1)  # (..., 1)
    sig = sig.unsqueeze(-1)  # (..., 1)
    ps = torch.linspace(0, 1.0, asize, device=mu.device, dtype=mu.dtype).view(
        ([1] * len(shphd)) + [-1]
    )  # (..., asize)
    ps = kernel(ps, mu, sig)  # (..., asize)
    if p_lb is not None:
        assert p_lb > 0, (_single_discrete, "got 0>=p_lb", p_lb, kernel)
        _1_eps_n = 1 - p_lb * asize
        assert _1_eps_n > 0
        ps = ps / ps.sum(-1, keepdim=True)
        with torch.no_grad():
            _pmin = ps.min(-1, keepdim=True)[0]
            _b = (p_lb - _pmin) / _1_eps_n
        ps = ps + _b
    pi_ = Categorical(ps)
    return pi_


def _idd(
    mu: "Tensor",  # (..., dimA)
    sig: "Tensor",  # (..., dimA)
    nvec: Sequence[int],  # (dimA,)
    kernel: "Union[_KernelFuncType, Sequence[_KernelFuncType]]" = kernel_gaussian,
    p_lb=None,  # 每一维的概率质量下界, None表示不限制(容易引发数值异常)
):
    _dimA = len(nvec)
    if callable(kernel):
        kernel = [kernel] * len(nvec)
    assert len(mu.shape) == len(sig.shape), (
        "shape of mu and sig should be the same, got",
        mu.shape,
        sig.shape,
    )
    assert _broadcastable(mu.shape[-1], sig.shape[-1], _dimA, len(kernel)), (
        "mu.shape[-1],sig.shape[-1],len(nvec),len(kernels)) should be broadcastable, got",
        mu.shape,
        sig.shape,
        _dimA,
        len(kernel),
    )
    _mus = mu.unbind(-1)
    _sigs = sig.unbind(-1)
    pi_subs = [
        _single_discrete(_mus[i], _sigs[i], nvec[i], kernel=kernel[i], p_lb=p_lb)
        for i in range(_dimA)
    ]  # (..., dimA)
    return pi_subs


def _idd_sample(pi_subs: Sequence[Distribution], sample_shape=torch.Size()):
    a_subs = [pi_.sample(sample_shape) for pi_ in pi_subs]  # (dimA,)
    act = cast(torch.LongTensor, torch.stack(a_subs, dim=-1))  # (..., dimA)
    return act


def _idd_log_prob(pi_subs: Sequence[Distribution], act: torch.LongTensor, joint=True):
    a_subs = act.unbind(dim=-1)  # (dimA,)
    logpa_ = torch.stack(
        [pi_.log_prob(a) for pi_, a in zip(pi_subs, a_subs)], dim=-1
    )  # (..., dimA)
    logpa = logpa_
    if joint:
        logpa = logpa_.sum(dim=-1, keepdim=True)  # (...,1)
    return logpa


def _idd_entropy(pi_subs: Sequence[Distribution], joint=True):
    ent = torch.stack([pi_.entropy() for pi_ in pi_subs], dim=-1)  # (..., dimA)
    if joint:
        ent = ent.sum(dim=-1, keepdim=True)  # (...,1)
    return ent


class IndependentDiscreteDistribution(Distribution):

    def __init__(
        self,
        mu: "Tensor",
        sig: "Tensor",
        nvec: Sequence[int],
        joint: bool = True,
        kernel: "Union[_KernelFuncType, List[_KernelFuncType]]" = kernel_gaussian,
        p_lb=1e-3,  # 每一维的概率质量下界
    ):
        r"""Multi-dimensional independent discrete distribution.

        $$
            \pi(a) := \prod_{i=1}^{dim_A} \pi_i(a_i), \forall a_i \in {0,..., |A_i|-1} \\
            \pi_i(a_i) := kernel_i(a_i; mu_i, sig_i) / Z_i \\
            Z_i := \sum_{a_i \in {0,..., |A_i|-1}} kernel_i(a_i; mu_i, sig_i)
        $$

        Args:
            mu ("Tensor"): 归一化状态空间上的均值, shape=(..., dim(A))
            sig ("Tensor"): 归一化状态空间上的方差, shape=(..., dim(A))
            nvec (Sequence[int]): 各维状态空间大小 $[|A_1|, |A_2|, ..., |A_{dim(A)}|]$
            joint (bool): 是否看作联合分布, 默认为True, 1->对数似然&熵为各维度之和,  0->各维独立计算对数似然&熵
        """
        super().__init__(validate_args=False)
        self._nvec = nvec
        self._kernel = kernel
        self._pis = _idd(mu, sig, nvec=self._nvec, kernel=kernel, p_lb=p_lb)
        self._joint = joint

    def sample(self, sample_shape=torch.Size()):
        dist = self._pis
        act = _idd_sample(dist, sample_shape)  # (..., dimA)
        return act

    def log_prob(self, act: torch.LongTensor):
        dist = self._pis
        logpa = _idd_log_prob(dist, act, joint=self._joint)  # (...,1)
        return logpa

    def entropy(self):
        dist = self._pis
        ent = _idd_entropy(dist, joint=self._joint)  # (...,1)
        return ent

    @property
    def nvec(self):
        return tuple(self._nvec)

    @classmethod
    def demo(cls):
        asize = [4, 5, 6]
        bsz = [2, 3]
        dimA = len(asize)
        p_lb = 1e-6
        mu = torch.rand([bsz[0], 1] + [dimA])
        sig = 0.1 + torch.rand([1, bsz[1]] + [dimA]) * 0.1
        pi_ = cls(mu, sig, asize, joint=False, kernel=kernel_gaussian, p_lb=p_lb)
        print(pi_.__class__.__name__)
        print("kernel:", pi_._kernel)
        print("nvec:", pi_.nvec)
        for shape_sample in [(), (2,)]:
            a = pi_.sample(shape_sample)
            print(a, a.shape, "shape_sample:", shape_sample)
            logpas = pi_._pis[0]
            for subpi in pi_._pis:
                logps = torch.exp(cast(Tensor, subpi.logits))
                plb_real = logps.min()
                print(plb_real, p_lb)
        ent = pi_.entropy()
        print(ent, "entropy shape:", ent.shape)


if __name__ == "__main__":
    IndependentDiscreteDistribution.demo()
