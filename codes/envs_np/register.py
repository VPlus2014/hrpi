"""
警告: 因为本项目所有环境都为向量化环境, 内部返回值均为向量
与被 ```gym.register``` 封装后的接口约束不一致, 会疯狂报错;
```truncated``` 不是环境组的向量返回值, 而是 ```gym.make``` 封装的标量!!!
故不在 import 时注册, 而是提供注册接口, 用户自行操作
"""

from gymnasium.envs.registration import register


def register_nav_heading_env():
    from .nav_heading import NavHeadingEnv

    name = "Navigation-np-v2"
    register(
        id=name,
        entry_point=f"{NavHeadingEnv.__module__}:{NavHeadingEnv.__name__}",
    )
    return name


def register_evasion_env():
    from .evasion import EvasionEnv

    register(
        id="Evasion-v2", entry_point=f"{EvasionEnv.__module__}:{EvasionEnv.__name__}"
    )
