"""Flat-surface Unitree Go1 locomotion, intended for Hackathon task T3."""

import gymnasium as gym

gym.register(
    id="Unitree-Go1-T3-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:Go1FlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:Go1FlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:Go1FlatPPORunnerCfg",
    },
)
