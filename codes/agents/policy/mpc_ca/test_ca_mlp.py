from __future__ import annotations
import itertools


def _setup():
    import os, sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[0]
    if str(_ROOT) not in sys.path:
        sys.path.append(str(_ROOT))
    return _ROOT


_ROOT = _setup()

from functools import partial
from timeit import timeit


def infer(model, x):
    y = model(x)
    y2 = model(x)
    return y2 - y


def main():
    # MIMO 效率:
    # MX>>DM>>SX, forall batch_size
    # batch size: fixed > variable
    # batch_first: False > True
    from ca_nn import MLP, CaMatLike
    import casadi as ca

    din = 10
    dout = 8
    ntst = 3
    for bsz in [128]:
        for batch_first, fix_bsz in itertools.product([False, True], [0, bsz]):
            params = {
                # "din": din,
                # "dout": dout,
                # "hiddens": [128, 128],
                "batch_size": bsz,
                "batch_first": batch_first,
                "fixed_batch_size": fix_bsz,
            }
            model = MLP(
                din,
                dout,
                hiddens=[128, 128],
                fixed_batch_size=fix_bsz,
                batch_first=batch_first,
            )

            sz_in = (bsz, din) if batch_first else (din, bsz)
            sz_out = (bsz, dout) if batch_first else (dout, bsz)

            for x in [
                ca.DM.rand(sz_in),
                ca.MX.sym("x", sz_in),
                ca.SX.sym("x", sz_in),
            ]:
                x: CaMatLike
                assert x.shape == sz_in, f"{x.shape} != {sz_in}"
                y = model(x)
                assert y.shape == sz_out, f"{y.shape} != {sz_out}"

                _tst_func = lambda: infer(model, x)
                t = timeit(_tst_func, number=ntst)
                fps = bsz * ntst / max(t, 1e-6)
                print(
                    params,
                    f"dt={t/ntst:.6f}s, fps={fps:.1f}, {type(x).__name__}->{type(y).__name__}",
                )


if __name__ == "__main__":
    main()
