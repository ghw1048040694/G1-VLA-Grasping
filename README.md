# G1 Upper-Body Control

Independent project for high-dynamic Unitree G1 upper-body skills with active
whole-body balance.

## System Boundary

- Isaac Gym: source simulator for GPU-parallel reinforcement learning.
- ASAP/HumanoidVerse: G1 motion tracking and PPO training framework.
- MuJoCo: target simulator for Sim2Sim evaluation with unchanged policy weights.
- World model: later predicts balance, contact, tracking failure, and fall risk.
- VLA: later maps visual and language commands to high-level skills or action chunks.
- LeRobot: optional VLA component only; it is not the low-level control framework.

## Layout

```text
G1-UpperBody/
|-- README.md
|-- scripts/       # Reproducible setup and experiment entry points
|-- outputs/       # Metrics, logs, checkpoints, and videos
`-- third_party/   # Links to external research code
```

## Current Experiment

`G1UB-01-IsaacGym-capacity-audit` found that Isaac Gym Preview 4 can create 64
GPU environments and begin PPO, but its own tensor example also crashes under
WSL2. This engine is not used for further WSL experiments.

`G1UB-01B-Genesis-capacity-audit` keeps the G1 task and PPO configuration fixed,
changes only the source simulator to Genesis, and probes 4, 8, 16, 32, and 64
parallel environments. The smaller range reflects the 8 GB laptop GPU and the
first observed failure at 64 environments.

Run from any directory:

```bash
bash /home/ubuntu/G1-UpperBody/scripts/run_g1ub_01b.sh 2>&1 | tee /home/ubuntu/G1-UpperBody/outputs/G1UB-01B.log
```

## Next Experiment

`G1UB-02-Genesis-CR7-baseline` uses the largest stable capacity found by
G1UB-01B: 32 environments. It trains for 200 PPO iterations and saves a
checkpoint every 50 iterations.

```bash
bash /home/ubuntu/G1-UpperBody/scripts/run_g1ub_02.sh 2>&1 | tee /home/ubuntu/G1-UpperBody/outputs/G1UB-02.log
```
