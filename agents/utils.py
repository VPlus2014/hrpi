import numpy as np
import torch

# TODO: 没有考虑到环境并行的问题！！！！！！！！对结果存在未知的影响


def as_tsr(
    data: np.ndarray | torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.tensor(data, device=device, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    else:
        raise TypeError(
            f"data@{as_tsr.__name__} must be np.ndarray or torch.Tensor, got {type(data)}"
        )


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(
        self, shape, device: torch.device
    ):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape, device=device)
        self.S = torch.zeros(shape, device=device)
        self.std = torch.sqrt(self.S)

    def update(self, x: torch.Tensor):
        if len(x.shape) == 2:
            x = x[0]

        self.n += 1
        if self.n == 1:
            self.mean = x.clone()
            self.std = x.clone()
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape, device: torch.device):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


def affcmb(a, b, t):
    """Affine combination of two tensors.

    $$
    (1-t) * a + t * b
    $$

    Args:
        a: First scalar or tensor, shape=(...,d).
        b: Second scalar or tensor, shape=(...,d).
        t: Weights for the affine combination, shape=(..., 1|dims).

    Returns:
        Affine combination of a and b, shape=(...,d).
    """
    return a + (b - a) * t


def affcmb_inv(a, b, y):
    """
    仿射组合的逆运算,含除零处理

    Args:
        a: First scalar or tensor, shape=(...,d).
        b: Second scalar or tensor, shape=(...,d).
        y: a+w*(b-a), shape=(..., 1)

    Returns:
        t: 仿射系数, shape=(...,d).
    """
    m = b - a
    y_ = y - a
    eps = 1e-6
    _bad = (m < eps) & (m > -eps)
    w = torch.where(_bad, 0.5, y_) / torch.where(_bad, 1, m)
    return w
