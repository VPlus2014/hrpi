from dataclasses import dataclass
import torch
import numpy as np
from typing import cast, Any, Protocol
from tianshou.data.batch import BatchProtocol
from tianshou.data import Batch


class RolloutBatchProtocol(BatchProtocol, Protocol):
    obs: torch.Tensor | np.ndarray
    obs_next: torch.Tensor | np.ndarray
    rew: torch.Tensor | np.ndarray
    truncated: torch.Tensor | np.ndarray
    terminated: torch.Tensor | np.ndarray

    act: torch.Tensor | np.ndarray
    act_log_prob: torch.Tensor | np.ndarray
