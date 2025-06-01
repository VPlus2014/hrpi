import torch


def main():

    q = torch.randn((8, 4), dtype=torch.float32)
    print(q)
    idxs = torch.arange(4)
    q[idxs, 0].fill_(1.0)
    print(q)
    return


pass
if __name__ == "__main__":
    main()
