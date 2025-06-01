import torch


def quat_split(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a quaternion into real and imaginary parts.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Re(q): 实部, shape is (..., 1).
        Im(q): 虚部, shape is (..., 3).
    """
    reQ, imQ = q.split([1, 3], dim=-1)
    return reQ, imQ




def main():
    q = torch.randn((2, 2, 22, 4), dtype=torch.float32)
    w, v = quat_split(q)
    print(w.shape, v.shape)
    return


pass
if __name__ == "__main__":
    main()
