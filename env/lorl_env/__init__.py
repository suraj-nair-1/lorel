from gym.envs.registration import register

register(
    id='LorlEnv-v0',
    entry_point='lorl_env.lorltabletop:LorlTabletop',
)
