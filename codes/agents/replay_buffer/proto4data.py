from dataclasses import dataclass
import torch
import numpy as np
from typing import Generic, TypeVar, cast, Any, Protocol
from tianshou.data.batch import BatchProtocol
from tianshou.data import Batch
from tianshou.data.buffer.base import ReplayBuffer as BaseReplayBuffer
from tianshou.data.stats import InfoStats, EpochStats

from ..types import RolloutBatchProtocol
