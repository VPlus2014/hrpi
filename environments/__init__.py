from environments.navigation import NavigationEnv
from environments.evasion import EvasionEnv
from gymnasium.envs.registration import register


register(
    id="Navigation-v1",
    entry_point=f"{NavigationEnv.__module__}:{NavigationEnv.__name__}",
)

register(id="Evasion-v1", entry_point=f"{EvasionEnv.__module__}:{EvasionEnv.__name__}")
