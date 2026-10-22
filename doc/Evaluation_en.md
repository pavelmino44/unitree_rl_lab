# Policy evaluation

Fixed, repeatable tests for a trained Go2 locomotion policy: a set of scenarios (flat ground, stairs, pits,
slopes, random blocks, sideways and backward walking, turning on the spot, sprint and stop), metrics per
episode, per scenario and per run, and a viewer with plots.

| File | Role |
|---|---|
| `scripts/rsl_rl/evaluate.py` | Runs the scenarios and writes a JSON report |
| `scripts/rsl_rl/eval_gui.py` | Settings window (`evaluate.py --gui`) |
| `scripts/rsl_rl/eval_report.py` | Report viewer (opens automatically, or on its own) |
| `.../robots/go2/eval_scenarios.py` | Scenario definitions |
| `.../robots/go2/velocity_env_cfg.py` | `RobotEvalEnvCfg`: evaluation settings shared by all scenarios |

Run everything from the `unitree_rl_lab` root, in the Isaac Lab conda environment. `eval_report.py` and the
GUI also need `matplotlib` and `tkinter`.

## Quick start

```bash
# settings window
python scripts/rsl_rl/evaluate.py --gui

# all scenarios, no simulator window
python scripts/rsl_rl/evaluate.py --headless --scenarios all --episodes 3

# open a saved report again
python scripts/rsl_rl/eval_report.py logs/rsl_rl/<experiment>/<run>/eval/eval_model_XXXX_<time>.json

# or use unitree_rl_lab.sh with the same arguments
./unitree_rl_lab.sh -e 
```

## evaluate.py

```bash
python scripts/rsl_rl/evaluate.py [--gui] [--task TASK] [--scenarios LIST] [--episodes N] [--eval_seed LIST]
                                  [--speed V] [--checkpoint PATH | --load_run RUN] [--output PATH]
                                  [--headless | --real-time] [--device DEV] [...]

./unitree_rl_lab.sh -e [--gui] [--task TASK] [--scenarios LIST] [--episodes N] [--eval_seed LIST]
                       [--speed V] [--checkpoint PATH | --load_run RUN] [--output PATH]
                       [--headless | --real-time] [--device DEV] [...]
```

| Option | Default | Meaning |
|---|---|---|
| `--gui` | | Open the settings window; all other options are then set there |
| `--task` | `Unitree-Go2-Velocity` | Gym task with an `eval_env_cfg_entry_point` |
| `--scenarios` | `all` | Comma-separated names or glob patterns; `rough` = `rough_*`, `all` = every scenario |
| `--episodes` | `5` | Episodes per scenario **and** seed |
| `--eval_seed` | `42` | Seed, or a comma-separated list (`42,43,44`); the episodes of all seeds are pooled |
| `--speed` | scenario default | Commanded speed in m/s for the walking scenarios (ignored by `flat_turn_360`, `flat_max_speed`) |
| `--checkpoint` | | Model file (`model_XXXX.pt`) |
| `--load_run` | latest run | Run folder under `logs/rsl_rl/<experiment>/`, used when no checkpoint is given |
| `--output` | `<checkpoint dir>/eval/eval_<model>_<time>.json` | Path of the JSON report |
| `--headless` | | No simulator window (fast) |
| `--real-time` | | Simulator window at real-time speed |
| `--no_contact_termination` | | A base contact does not end the episode (to see every contact) |
| `--record_dt` | `0.1` | Time step (s) of the time series stored in the report |
| `--no_report` | | Do not open the viewer at the end |
| `--report_save` | | Also save all figures as PNG into this folder |
| `--num_envs` | `1` | Parallel copies of the same scenario |
| `--device`, `--disable_fabric` | | Isaac Lab options |

Every (scenario, seed) pair runs in its own Isaac Sim process. The parent process collects the results,
prints the tables, writes one JSON report and opens the viewer. Use quotes around glob patterns:
`--scenarios "rough_pyramid_stairs_*cm,flat*"`.

**Seeds.** The policy is deterministic at inference, and so are flat ground, stairs and slopes: repeated
episodes on one seed are almost identical. The seed changes the layout of the random terrains
(`rough_boxes_*`, `rough_random_rough`) – use several seeds for them.

## GUI

`python scripts/rsl_rl/evaluate.py --gui`

- **Scenarios**: multi-select list, filter (substring or glob), quick filters (flat, stairs, slopes, boxes).
  Hover a name to see its description. *Refresh list* reloads the scenarios after `eval_scenarios.py` changes
  (it starts a short headless Isaac Sim; the list is cached in `~/.cache/unitree_rl_lab/eval_catalog.json`).
- **Run settings**: the same options as on the command line. The *Equivalent command* line shows the command
  that will run; *Copy* puts it in the clipboard.
- **Configurations**: *Save* / *Load* store all settings as JSON (default folder
  `scripts/rsl_rl/eval_configs/`). The last used settings are restored on the next start (`_last.json`).
- *Run* closes the window; the run continues in the terminal.

## Report viewer

```bash
python scripts/rsl_rl/eval_report.py REPORT.json [--scenarios LIST] [--no_overall] [--no_scenarios]
                                     [--no_episodes] [--max_windows N] [--save DIR] [--no_show] [--dpi D]
```

| Window | Content |
|---|---|
| Whole run | Outcomes, speed tracking and body contacts per scenario; success vs. step height / slope angle / block difficulty; summary and weakest scenarios |
| Scenario | Averages over its episodes along the course (or over time / rotation): speed vs. command, where the body touched the surface, outcome per episode, heading error and lateral deviation, tilt and power |
| Episode | Speed vs. time with the command, body-contact timeline (base / head / legs), progress, heading error / tilt / lateral deviation, all numbers |

At most `--max_windows` (40) windows open at once; `--save DIR --no_show` writes all figures as PNG. Without a
display the figures are saved next to the report. Windows are scaled to fit the screen (`--dpi` to override).

## Scenarios

All scenarios use one robot, no domain randomization, no observation noise and no curriculum. The robot
spawns on flat ground 0.7 m before the obstacle, its heading is held by a closed-loop yaw-rate command, and
the episode ends at the goal line. Pyramids and pits are crossed completely (up **and** down).
The `rough_*` scenarios take their terrain from `ROUGH_TERRAINS_CFG`, the training terrain set.

| Scenario | Task |
|---|---|
| `flat` | 6 m forward, 0.5 m/s |
| `flat_backward` | 6 m backward, 0.5 m/s |
| `flat_strafe_right`, `flat_strafe_left` | 6 m sideways, 0.3 m/s, facing forward |
| `flat_turn_360` | One full turn on the spot, 0.8 rad/s |
| `flat_max_speed` | Accelerate to the maximum speed of the training range, command 0 at 6 s; success = survive and stop |
| `rough_pyramid_stairs_<N>cm` | Pyramid stairs, step height N = 5, 10…20 cm |
| `rough_pyramid_stairs_inv_<N>cm` | Pit (inverted pyramid), same heights |
| `rough_pyramid_stairs[_inv]_d100` | Stairs / pit at the ROUGH difficulty 1.0 |
| `rough_boxes_d25/50/75/100` | Random blocks at 25…100 % difficulty |
| `rough_hf_pyramid_slope_<A>deg` | Slope pyramid, A = 10…45° |
| `rough_hf_pyramid_slope[_inv]_d50/_d100` | Slope / inverted slope at ROUGH difficulty |
| `rough_random_rough` | Uniform height noise |
| `slope_up_0p2`, `slope_up_0p4` | Hand-made slopes, rise/run 0.2 and 0.4 |

To add a scenario, add an entry to `EVAL_SCENARIOS` at the end of `eval_scenarios.py`.

## Metrics

Per-episode values are in the JSON report; the tables and the viewer show their means (and worst cases) per
scenario and per run.

### Task

| Metric | Meaning |
|---|---|
| `success` | Goal line crossed (far side of a pyramid/pit, full 360° turn, or for `flat_max_speed`: survived and stopped) |
| `midpoint` | Reached the top of a pyramid, the bottom of a pit or the centre of the blocks |
| `falls` | Ended by a termination (`bad_orientation`, `base_contact`) before the goal |
| `timeouts` | Neither goal nor fall before the time limit – usually stuck |
| `t_goal` | Time to the goal (successful episodes) |
| `progress` | Best distance along the course (m), or rotation (deg) for the turn |
| `end reasons` | Which termination ended each episode |

### Command tracking and path

| Metric | Meaning |
|---|---|
| `speed` | Mean speed along the commanded direction (positive when moving the right way) |
| `vel err xy` | Mean distance between commanded and measured planar velocity, m/s |
| `vel err yaw` | Mean yaw-rate error, rad/s |
| `yaw err` (heading error) | Largest deviation of the heading from the spawn heading, deg (not used for the turn) |
| `lateral` | Largest sideways deviation from the start line, m (turn: drift of the base from the start point) |

### Body contacts

Contacts of the robot's body (not the feet) with the surroundings, threshold = the force threshold of the
training termination (1 N):

| Part | Bodies | In training |
|---|---|---|
| base | `base` | ends the episode (`base_contact`) |
| head | `Head_.*` | penalized only (`undesired_contacts`) |
| legs | `*_hip`, `*_thigh`, `*_calf` | penalized only (`undesired_contacts`) |

**trunk** = base or head. For the trunk and the legs: events per episode (contacts closer than 0.1 s count
as one), share of the episodes with a contact, contact time, first contact, max force; per-part counts and a
timeline per episode.

### Sprint and stop (`flat_max_speed`)

| Metric | Meaning |
|---|---|
| `peak` | Highest speed before the stop command |
| `t to 90%` | Time to reach 90 % of the commanded speed |
| `at brake` | Speed when the stop command is given |
| `stop time`, `stop dist` | Time and distance until \|v\| < 0.1 m/s for 0.5 s |
| `lowest` | Lowest speed after the stop command (negative = the robot was thrown back) |
| `stopped` | Share of the episodes that came to a stop |

### Stability, effort, smoothness

| Metric | Meaning |
|---|---|
| `tilt` | Angle between the body's vertical axis and gravity (roll and pitch combined), mean and max, deg |
| `power` | Mean mechanical power Σ\|τ·q̇\|, W |
| `CoT` | Cost of transport E / (m·g·d) – energy per distance, lower is more efficient |
| `action rate` | Mean \|a_t − a_t−1\| – how jerky the actions are |
| `joint acc` | Mean joint acceleration norm |
| `reward`, `reward terms /s` | Training reward of the episode; each term per second, comparable to `Episode_Reward/<term>` in the training logs |

## Reading the results

1. **Sweeps** in the whole-run window: "passed up to N cm / deg" (success ≥ 80 % at every level up to N) is
   the most compact description of what the policy can do – compare it between checkpoints.
2. **Flat ground** (`flat*`): should be 100 % success and ~100 % speed tracking – a regression check.
3. **Weakest scenarios**: open their windows; the plots along the course show *where* it fails.
4. **Effort and smoothness** decide between checkpoints with the same capabilities.

| Pattern | Likely cause |
|---|---|
| `midpoint` high, `success` low | Climbs but falls on the way down |
| Head contacts on stairs | The robot leans on the next step with its nose (head contacts do not end training episodes) |
| Leg contacts at step edges | Feet not lifted high enough |
| Many timeouts, low progress | Stuck at the obstacle |
| Heading error grows on obstacles | Asymmetric gait; compare with `flat_strafe_*` and `flat_turn_360` |
| Negative `lowest` after the stop | Abrupt braking |

Use the same scenario set, seeds and episode count for every checkpoint you compare (save it as a GUI
configuration). Differences of one episode out of five are within noise.