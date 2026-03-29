from __future__ import annotations


def _setup():  # 确保项目根节点在 sys.path 中
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


_ROOT = _setup()


def main():
    from codes.agents.policy.mpc_ca import plane_p6dof_opti

    plane_p6dof_opti.demo()


if __name__ == "__main__":
    main()
