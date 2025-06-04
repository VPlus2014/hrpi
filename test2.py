import hashlib
import heapq
import time
from timeit import timeit
import numpy as np

# import torch
# import torch.nn as nn


def get_nsmallest_idx(arr, m):
    use_np = False
    if use_np:
        idx = np.arange(len(arr))
    else:
        idx = list(range(len(arr)))
    return heapq.nsmallest(m, idx, key=arr.__getitem__)


# class AgentWithCache(nn.Module):
#     def __init__(self, input_size, output_size, buffer_size):
#         super(AgentWithCache, self).__init__()

#         xmin = torch.zeros(input_size)
#         xmax = torch.ones(input_size)
#         xspan = xmax - xmin

#         self.register_buffer("xmin", xmin)
#         self.xmin = self.xmin
#         self.xmax = xmax
#         self.xspan = xspan


def main():
    x = [
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
    ]

    ntest = int(100)
    a = np.asarray(x)
    np.random.shuffle(a)
    m = 4
    print(a[:m])

    def f():
        addrs = get_nsmallest_idx(a, m)

    t0 = time.time()
    dt = timeit(f, number=ntest)
    t1 = time.time()

    addrs = get_nsmallest_idx(a, m)
    print("dt", dt / ntest)
    print("idx", addrs)
    print("val", a[addrs])
    print("sorted", np.sort(a)[:m])
    # a = torch.arange(100)
    # print(a.shape)
    # x = a[None]
    # print(x.shape)


pass
if __name__ == "__main__":
    main()
