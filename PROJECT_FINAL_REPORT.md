# G1 Upper-Body VLA + World Model Evidence Report

Date: 2026-07-26

## Status

The project has a reproducible modular evidence package. The strict end-to-end
language manipulation gate is not passed and must not be presented as passed.
The latest task-level result is the `G1FINAL-09` extension of the `G1FINAL-07`
hybrid: a learned language
router selects the requested object, then the verified classical controller
executes the pick-and-place sequence. The extended robustness run passes `9/9`
registered instructions in MuJoCo. This is the current practical demo and is
deliberately reported separately from end-to-end learned SmolVLA control.
The validated architecture is:

```text
language instruction
  -> G1LANG-34 target classifier
  -> target-specific SmolVLA specialist
  -> residual World Model + candidate MPC
  -> MuJoCo closed-loop control
  -> Isaac Sim assisted-grasp replay
```

## Validated Results

| Component | Evidence | Result |
| --- | --- | --- |
| Target specialists | `outputs/G1FINAL-01_target_specialists_audit.json` | red/yellow/green 5K checkpoints structurally and numerically valid |
| Language routing | `outputs/G1FINAL-01_target_router_smoke_3ep/summary.json` | `3/3` route accuracy |
| World Model | `outputs/G1FINAL-02_language_world_model_10000step/summary.json` | screening checks passed; best step `9500`; H20 target position RMSE `0.00750 m` |
| MPC integration | `outputs/G1FINAL-03_target_router_mpc_smoke_3ep/summary.json` | 108 replans; 82 non-baseline selections |
| Isaac Sim transfer | `/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-02/isaacsim_report.json` | 31-joint contract passed; final lift `0.16607 m`; transfer gate passed |
| Learned router + classical execution | `outputs/G1FINAL-09_router_classical_9ep/summary.json` | router `9/9`; controller `9/9`; hybrid strict success `9/9` |

## Failed Gates

The closed-loop baseline and MPC both selected the wrong object in two of three
episodes and completed no strict language task:

```text
router accuracy:             3/3
object selection accuracy:   0/3
strict success:              0/3
wrong-object grasp rate:     2/3
```

The video/trajectory audit is in
`outputs/G1FINAL-05_video_audit.json`. It confirms that actions were applied,
but the requested object was not selected by the learned specialist/MPC path.
The learned end-to-end source for G1SIM-03 remains unavailable; G1FINAL-07
supplies a separate strict hybrid source, whose Isaac replay is reported below.

## Videos

Working MuJoCo videos (the latest hybrid demo):

```text
outputs/G1FINAL-09_router_classical_9ep/episode_0000/language_pick_place.mp4
outputs/G1FINAL-09_router_classical_9ep/episode_0001/language_pick_place.mp4
outputs/G1FINAL-09_router_classical_9ep/episode_0002/language_pick_place.mp4
```

Open one from WSL2 with:

```bash
explorer.exe "$(wslpath -w /home/ubuntu/G1-UpperBody/outputs/G1FINAL-09_router_classical_9ep/episode_0000/language_pick_place.mp4)"
```

## Reproducibility

The primary Python environment is:

```text
/home/ubuntu/miniconda3/envs/lerobot/bin/python
```

The evaluation import contract needs Genesis MuJoCo `3.2.5` first on
`PYTHONPATH`, plus LeRobot's `datasets` and `transformers` packages. The
machine-readable project audit is
`outputs/G1FINAL-04_project_audit.json`.

## Known Limitation

The target specialist dataset contract is correct, but the 5K specialists do
not provide reliable visual target grounding. Further SmolVLA training on the
current WSL2/DXG stack repeatedly fails in CUDA attention/backward kernels, even
with an evaluation-only or full-FP32 fallback. A fresh supported CUDA context or
different training stack is required before another training attempt.

## G1FINAL-07 Hybrid Execution Evidence

After the `G1FINAL-03` video audit showed that the learned specialist/MPC
controller moved the wrong object, a separate hybrid experiment was run. The
trained `G1LANG-34` language router was evaluated on the initial rendered scene
and its selected target was passed to the already verified classical
IK/interpolation/assisted-grasp controller.

The result was `3/3` router accuracy, `3/3` controller success, and `3/3`
hybrid strict success, with mean joint-limit violation fraction `0.000340`.
The preferred multi-scene rerun is recorded in
`outputs/G1FINAL-09_router_classical_9ep/summary.json`. It covers three
object-slot permutations and fresh position jitter: router `9/9`, controller
`9/9`, hybrid strict success `9/9`, mean joint-limit violation fraction
`0.000441`, and minimum router top-1 probability `0.999798`.
The evidence is in:

```text
outputs/G1FINAL-09_router_classical_9ep/summary.json
```

The nine videos are under the corresponding episode directories. The
preferred launcher is `scripts/run_g1_final09.sh`; it refuses to overwrite an
existing result directory. This is a
learned-language-routing plus privileged/classical-control upper bound. It is
valid evidence that the language route can select a target for a reliable
controller, but it must not be reported as end-to-end learned SmolVLA success.
The end-to-end learned gate from `G1FINAL-03` remains `0/3`.

## G1FINAL-08 Hybrid Sim2Sim Attempt

The successful `G1FINAL-07` red-triangle episode was converted to the
`G1SIM-03` offline replay contract. The conversion passed the 31-action joint
contract and produced `source_passed=true` under:

```text
/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-03/manifest.json
/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-03/data/source_summary.json
```

The Isaac Sim 6.0.1 replay did not produce `isaacsim_report.json`. After the
MJCF importer stage, RTX reported `GPU pagefault`, `VkResult: ERROR_DEVICE_LOST`,
and exited with code `3221225477`. Removing the non-target assisted-grasp
constraints and replacing the inline triangle mesh with a box primitive did not
change the failure. This is recorded as an Isaac/RTX environment failure, not
as a task-success result; the MuJoCo hybrid videos and metrics remain valid.
