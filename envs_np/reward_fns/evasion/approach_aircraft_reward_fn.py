# import torch
from typing import TYPE_CHECKING, Sequence
from ..proto4rf import BaseRewardFn
from environments_th.utils.math_pt import euler_from_quat, ned2aer

if TYPE_CHECKING:
    from environments_th.evasion import EvasionEnv


class ApproachNavigationPointRewardFn(BaseRewardFn):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def reset(self, env, env_indices=None, **kwargs):
        pass

    def forward(self, env: "EvasionEnv", plane, **kwargs) -> np.ndarray:
        mis = env.missile
        r = mis.position_e() - plane.position_e()  # 实际的视线
        dr = mis.velocity_e() - plane.velocity_e()  # 实际的相对速度
        Vesc = (r * dr).sum(-1) / np.norm(r, p=2, axis=-1).clamp(min=1e-3)  # 逃逸速率
        # Vc = np.einsum("ij,ij->i", [v, r]) / np.norm(r, p=2, axis=-1)
        return Vesc
