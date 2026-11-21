"""Check real PhysX masses/inertias/friction before training, no checkpoint.

Run in env_isaaclab: python scripts/project/check_go1_physics.py --headless
The same validation also runs automatically at Go1 training startup.
"""
import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=42, help="Environment seed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.robots.go1.velocity_env_cfg import Go1FlatPlayEnvCfg


def main():
    cfg = Go1FlatPlayEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.commands.base_velocity.debug_vis = not args.headless
    print("[GO1 CHECK] creating environment", flush=True)
    env = gym.make("Unitree-Go1-T3-Flat", cfg=cfg)
    try:
        print("[GO1 CHECK] resetting environment", flush=True)
        env.reset()
        if not getattr(env.unwrapped, "_go1_mjcf_physics_report", None):
            raise RuntimeError("The Go1 MJCF startup check did not run; check the installed configuration")
        print("[GO1 PHYSICS CHECK COMPLETE] real PhysX properties verified; no policy loaded", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        # Kit shutdown can terminate Python before its uncaught-exception
        # printer runs. Emit the original traceback BEFORE app.close().
        print("[GO1 CHECK FAILED]", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
    raise SystemExit(exit_code)
