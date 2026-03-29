# interactive multi model predictive control for fixed-wing aircraft

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.integrate as integrate
import time


class Plane3DOFModel:

    g = 9.8

    def __init__(self, group_shape: int, m):
        self.m = m
        self.group_shape: tuple[int, ...] = tuple(
            np.ravel(group_shape).astype(np.int32).tolist()
        )

    def forward(self, x, u):
        # x: state vector [x, y, z, dx, dy, dz]
        # u: input vector [fx, fy, fz]
        # f: state derivative vector [dx, dy, dz, ddx, ddy, ddz]
        xdot = np.zeros_like(x)
        xdot[..., 0:3] = x[..., 3:6]
        xdot[..., 3:6] = u
        return xdot
    
    def __call__(self, x, u):
        return self.forward(x, u)


class Missile3DOFModel(Plane3DOFModel):
    def __init__(self, group_shape: int, m):
        super().__init__(group_shape, m)

    def forward(self, x, u):
        # x: state vector [x, y, z, dx, dy, dz], shape=(B,n,n_target,6)
        pass


class IMM4Plane:

    def __init__(
        self,
        ego: Plane3DOFModel,  # (B,n_plane,1)
        enemy_auto: Missile3DOFModel,  # (B,1,n_missile)
        dt=0.1,
    ):
        self._ego = ego
        self._enemy = enemy_auto
        self.dt = dt
        try:
            assert len(ego.group_shape) == 3, "EGO group shape must be (B,n_P,1)"
            assert (
                len(enemy_auto.group_shape) == 3
            ), "ENEMY group shape must be (B,1,n_M)"
            assert (
                ego.group_shape[0] == enemy_auto.group_shape[0]
            ), "Batch size of EGO and ENEMY must match"
            assert ego.group_shape[-1] == 1, "EGO group shape must be (B,n_P,1)"
            assert (
                enemy_auto.group_shape[-2] == 1
            ), "ENEMY group shape must be (B,1,n_M)"
        except Exception as e:
            raise Exception(ego.group_shape, enemy_auto.group_shape) from e
        self.batch_size = ego.group_shape[0]

    
    def forward(self, x, u):
        # x = (X_ego, X_enemy)
        x_ego = x[0]
        x_enemy = x[1]
        

def main():
    m=45.8
    for bmi in [14.5,15.0]:
        h = math.sqrt(m / bmi)
        print(f"h={h:.2f} m")

if __name__ == "__main__":
    main()