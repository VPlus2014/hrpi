from __future__ import annotations


def _setup():  # 确保项目根节点在 sys.path 中
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


_setup()


def main():
    from envs_np.utils.math_np._test import main as test_math_np

    test_math_np()


if __name__ == "__main__":
    main()
