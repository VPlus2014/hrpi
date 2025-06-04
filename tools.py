from contextlib import ContextDecorator
import time
import numpy as np
import torch
import random
import numpy
import os
from decimal import Decimal, getcontext

getcontext().prec = 4


def as_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x)


def as_tsr(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.asarray(x)


class ConextTimer(ContextDecorator):
    def __init__(self, name: str = ""):
        self.name = name
        self.t = 0
        self.dt = 0
        self._lv = 0

    def reset(self):
        self.t = 0
        self.dt = 0

    def __enter__(self):
        self.push()

    def __exit__(self, *exc):
        self.pop()

    def push(self):
        self._lv += 1
        if self._lv == 1:
            self._t0 = time.time()

    def pop(self):
        if self._lv > 0:
            self._lv -= 1
            if self._lv == 0:
                self.dt = dt = time.time() - self._t0
                self.t += dt


def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    print(f"Seed initialized to {seed}")


def set_max_threads(n: int = 16):
    os.environ["NUMEXPR_MAX_THREADS"] = str(n)
