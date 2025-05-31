import torch
import numpy as np
from typing import cast, Any, Protocol
from tianshou.data.batch import BatchProtocol



class RolloutBatchProtocol(BatchProtocol, Protocol):
    obs: np.ndarray | torch.Tensor
    obs_next: np.ndarray | torch.Tensor
    rew: float | torch.Tensor
    truncated: bool | torch.Tensor
    terminated: bool | torch.Tensor

    act: np.ndarray | torch.Tensor
    act_log_prob: np.ndarray | torch.Tensor

    done: bool | torch.Tensor

# class RolloutExtBatchProtocol(RolloutBatchProtocol, Protocol):
#     ret: np.ndarray
#     adv: np.ndarray
#     v_s: np.ndarray