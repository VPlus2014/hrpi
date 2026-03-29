from __future__ import annotations

# 因为 casadi 只支持矩阵且广播机制只限于在 axis=1, 所以这里输入尺寸必须为 (d,batch_size)
from typing import Any, Iterator, Sequence, TypeVar, Union
import casadi as ca
from casadi import SX, MX, DM
import numpy as np

# import pprint
CaMatLike = Union[ca.DM, ca.SX, ca.MX]
T_CaMatLike_co = TypeVar("T_CaMatLike_co", bound=CaMatLike, covariant=True)

StateDictType = dict[str, "StateDictType"]


class CANNModule:

    IDX_BATCH = 1

    def __init__(self):
        pass

    def forward(self, x: T_CaMatLike_co, *args, **kwargs) -> T_CaMatLike_co | Any:
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() not implemented"
        )

    def __call__(self, x: T_CaMatLike_co, *args, **kwargs) -> T_CaMatLike_co | Any:
        return self.forward(x, *args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        # val = self.__dict__.get(name, None)
        # return val
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self.__dict__.get(name, None)

    def add_module(self, name: str, module: CANNModule):
        assert isinstance(module, CANNModule)
        assert name not in self.__dict__, f"'{name}' already exists"
        self.__dict__[name] = module

    def load_state_dict(self, state_dict: StateDictType):
        nms = dict(self.named_children())
        for name, sdi in state_dict.items():
            m = nms.get(name, None)
            if m is None:
                continue
            m.load_state_dict(sdi)

    def state_dict(self) -> StateDictType:
        d = {}
        for name, m in self.named_children():
            # clsname = m.__class__.__name__
            sdm = m.state_dict()
            if len(sdm):
                d[name] = sdm
        return d

    def named_children(self) -> Iterator[tuple[str, "CANNModule"]]:
        for name, m in self.__dict__.items():
            if isinstance(m, CANNModule):
                if m is self:
                    continue
                yield name, m

    def named_modules(self, recurse: bool = True) -> Iterator[tuple[str, "CANNModule"]]:
        kvs = []
        idx = 0
        kvs.append((f"{idx}.{self.__class__.__name__}", self))
        yield kvs[-1]
        idx += 1
        ids_tabu = {id(self)}
        for name, m in self.__dict__.items():
            if not isinstance(m, CANNModule):
                continue
            idm = id(m)
            if idm in ids_tabu:
                continue
            # ids_tabu.add(idm)
            prefix = f"{idx}"
            idx += 1
            di = m.named_modules(recurse=recurse)
            for ni, mi in di:
                idi = id(mi)
                if idi in ids_tabu:
                    continue
                ids_tabu.add(idi)
                kvs.append((f"{prefix}.{ni}", mi))
                yield kvs[-1]
                if not recurse and mi == m:
                    break
        # return iter(kvs)

    def __str__(self, indent: int = 0):
        # import pprint
        # return pprint.pformat(self, indent=indent)
        sep = "  "
        prefix = sep * indent
        prefix_attr = prefix + sep
        # ss = []
        # for name, attr in self.__dict__.items():
        #     if attr is self:
        #         continue
        #     if isinstance(attr, CANNModule):
        #         si = attr.__str__(indent + 1)
        #     else:
        #         si = str(attr)
        #     if len(si) > 10 or isinstance(attr, CANNModule):
        #         _pri = "\n" + prefix + sep
        #     else:
        #         _pri = ""

        #     ss.append(f"{_pri}{name}={si}")

        ps = []
        for name, attr in self.__dict__.items():
            if name.startswith("_") or not isinstance(attr, (int, float, str, bool)):
                continue
            ps.append(f"{name}={attr}")
        s1 = ", ".join(ps)

        lines4m = []
        if len(ps):
            lines4m.append(s1)
        for name, m in self.named_modules():
            if not isinstance(m, CANNModule) or m is self:
                continue
            if m not in self.__dict__.values():
                continue
            # si = pprint.pformat(m, indent=indent + 1,depth=1)
            si = m.__str__(indent + 1)
            lines4m.append(f"{name}={si}")

        s = ("\n" + prefix_attr).join(lines4m)
        if len(lines4m) > 1:
            s = ("\n" + prefix_attr) + s + ("\n" + prefix)

        s = f"{self.__class__.__name__}({s})"
        return s


class Linear(CANNModule):
    """see `torch.nn.Linear`"""

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, batch_first=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if batch_first:
            self.weight = DM(in_features, out_features)
        else:
            self.weight = DM(out_features, in_features)
        if bias:
            if batch_first:
                _bias = DM(1, out_features)
            else:
                _bias = DM(out_features, 1)
        else:
            _bias = None
        self.bias = _bias
        self.batch_first = batch_first

    def __repr__(self):
        return f"{self.__class__.__name__}(din={self.in_features}, dout={self.out_features}, bias={self.bias is not None})"

    def forward(self, x: Union[ca.DM, ca.SX]):
        batch_first = self.batch_first
        if batch_first:
            x = x @ self.weight
        else:
            x = self.weight @ x

        b = self.bias
        if b is not None:
            if batch_first:
                b = ca.repmat(b, x.shape[0], 1)
            x = x + b
        return x

    def state_dict(self) -> dict[str, np.ndarray]:
        """
        w: (dout,din)
        b: (dout,)
        """
        w = self.weight.full()
        if self.batch_first:
            w = w.T
        sd = {"weight": w}
        if self.bias is not None:
            sd["bias"] = np.ravel(self.bias.full())
        return sd

    def load_state_dict(self, state_dict: dict):
        w = state_dict["weight"]
        if self.batch_first:
            w = w.T
        self.weight[:, :] = w
        bias = state_dict.get("bias", None)
        if bias is not None and self.bias is not None:
            bias = np.reshape(bias, self.bias.shape)
            self.bias[:, :] = bias


class ReLU(CANNModule):
    def __init__(self):
        super().__init__()

    def forward(self, x: Union[ca.DM, ca.SX]):
        x = ca.fmax(x, 0)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Sequential(CANNModule):
    def __init__(self, *modules: CANNModule):
        super().__init__()
        ms: list[CANNModule] = []
        for i, m in enumerate(modules):
            self.__setattr__(f"{i}.{m.__class__.__name__}", m)
            ms.append(m)
        self.modules = ms

    def forward(self, x: Union[ca.DM, ca.SX, ca.MX]):
        for m in self.modules:
            x = m(x)
        return x

    def __repr__(self):
        ms = [repr(m) for n, m in self.__dict__.items() if isinstance(m, CANNModule)]
        return "{}(\n  {}\n)".format(self.__class__.__name__, ",\n  ".join(ms))


def init_weights(net: CANNModule, gain=1.0):
    for name, m in net.named_modules():
        if isinstance(m, Linear):
            din = m.weight.shape[1]
            # dout = m.weight.shape[0]
            m.weight[:, :] = np.random.randn(*m.weight.shape) * (gain / din)
            if m.bias is not None:
                m.bias[:, :] = 0
    return net


def mlp_maker(din: int, dout: int, hidden_size: Sequence[int] = (), batch_first=False):
    layers = []
    dims = [din, *hidden_size, dout]
    for i in range(len(dims) - 1):
        layers.append(Linear(dims[i], dims[i + 1], batch_first=batch_first))
        layers.append(ReLU())
    layers.pop()
    return Sequential(*layers)


def demo():
    from copy import deepcopy

    batch_size = 1024
    din = 2
    dout = 1
    layers = []

    model = Sequential(
        *[
            Linear(din, 1),
            Sequential(
                Linear(1, 1),
                ReLU(),
                mlp_maker(1, 1, (100, 100)),
                ReLU(),
                Sequential(
                    mlp_maker(1, 1, (10, 10)),
                    ReLU(),
                    mlp_maker(1, 2, (10, 10)),
                ),
                mlp_maker(2, dout, (10, 10)),
            ),
        ]
    )
    print(model)
    model = init_weights(model)
    x = DM.rand(din, batch_size)
    y = model(x)
    print(y.shape)

    for batch_first in [True, False]:
        print("-"*40)
        print("batch_first:", batch_first)
        model = mlp_maker(din, dout, (4,))

        sd = deepcopy(model.state_dict())
        print("model>>", sd)
        init_weights(model)
        print("model>>", model.state_dict())
        model.load_state_dict(sd)
        print("model>>", model.state_dict())


if __name__ == "__main__":
    demo()
