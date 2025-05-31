from environments.navigation import NavigationEnv
from environments.evasion import EvasionEnv
from gymnasium.envs.registration import register


register(
    id="Navigation-v1",
    entry_point="environments.navigation:NavigationEnv"
)

register(
    id="Evasion-v1",
    entry_point="environments.evasion:EvasionEnv"
)
