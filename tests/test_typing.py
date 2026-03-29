import math
from typing import Iterable

import numpy as np
import pandas as pd


def isdtypeof(d, t):
    if isinstance(t, Iterable):
        return any(isdtypeof(d, _t) for _t in t)
    return np.issubdtype(d, t)


def _format_dtype(d):
    if isinstance(d, np.dtype):
        return repr(d)
    if isinstance(d, type):
        return d.__name__
    return str(d)


def main():
    rowheaders = [
        int,
        float,
        bool,
    ]
    for t in [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ]:
        rowheaders.append(t)
        # dt = np.dtype(t)
        # rowheaders.append(dt)
    colheaders = [int, float, bool, np.integer, np.floating, np.bool_]

    rows = []
    for t in rowheaders:
        row = []
        for tgt in colheaders:
            dt = np.dtype(t)
            yt1 = isdtypeof(t, tgt)
            yt2 = isdtypeof(dt, tgt)
            assert (
                yt1 == yt2,
                (
                    "answer of",
                    (t, tgt),
                    (dt, tgt),
                    "not equal",
                ),
            )
            row.append("True" if yt1 else "")
        rows.append(row)

    df = pd.DataFrame(
        rows,
        index=list(map(_format_dtype, rowheaders)),
        columns=list(map(_format_dtype, colheaders)),
    )
    pd.options.display.max_columns = None # type: ignore
    pd.options.display.max_rows = None # type: ignore
    print(df)


if __name__ == "__main__":
    main()
