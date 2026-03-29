from typing import List, TypeVar, Union, no_type_check, Sequence
from ._modules import CANNModule, Sequential, Linear, ReLU, init_weights, T_CaMatLike_co
import casadi as ca
import numpy as np


class MLP(CANNModule):
    @no_type_check
    def __init__(
        self,
        din: int,
        dout: int,
        hiddens: Sequence[int] = (),
        end_with_activation: bool = False,
        use_MX=True,  # use MX instead of SX, for faster computation
        fixed_batch_size: int = 0,
        batch_first: bool = False,
    ):
        super().__init__()
        self._din = din
        self._dout = dout

        hiddens = [din, *hiddens, dout]

        actv = ReLU()
        layers = []
        for i in range(len(hiddens) - 1):
            layers.append(Linear(hiddens[i], hiddens[i + 1], batch_first=batch_first))
            layers.append(actv)
        if not end_with_activation:
            layers.pop()
        self._net = Sequential(*layers)
        self.use_MX = use_MX
        self.fixed_batch_size = fixed_batch_size
        self.batch_first = batch_first

        init_weights(self)
        self._remake()

    @no_type_check
    def _remake(self):
        if self.fixed_batch_size > 0:
            _sym = ca.MX.sym if self.use_MX else ca.SX.sym
            if self.batch_first:
                sz = (self.fixed_batch_size, self._din)
            else:
                sz = (self._din, self.fixed_batch_size)
            X = _sym("X", sz)
            Y = self._net(X)
            self._forward = ca.Function("mlp", [X], [Y])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._remake()

    def forward(self, x: T_CaMatLike_co) -> T_CaMatLike_co:
        fix_bsz = self.fixed_batch_size
        if fix_bsz > 0:
            batch_first = self.batch_first
            axis = 0 if batch_first else 1
            assert x.shape[axis] == fix_bsz, (
                f"input batch size should be {fix_bsz}",
                x.shape[axis],
            )
            y = self._forward(x)
        else:
            y = self._net(x)
        return y  # type: ignore
