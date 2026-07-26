# G1 Upper-Body VLA + World Model Evidence Report

Date: 2026-07-26

## Status

The project has a reproducible modular evidence package. The strict end-to-end
language manipulation gate is not passed and must not be presented as passed.
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
but the requested object was not selected. G1SIM-03 is blocked because it
requires a strict-success language episode as its source.

## Videos

Working MuJoCo videos:

```text
outputs/G1FINAL-01_target_router_smoke_3ep/episode_0000/closed_loop.mp4
outputs/G1FINAL-03_target_router_mpc_smoke_3ep/episode_0000/closed_loop.mp4
```

Open one from WSL2 with:

```bash
explorer.exe "$(wslpath -w /home/ubuntu/G1-UpperBody/outputs/G1FINAL-03_target_router_mpc_smoke_3ep/episode_0000/closed_loop.mp4)"
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
