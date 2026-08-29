# G1 Grasp / Warehouse VLA Project

This repository keeps the paused G1 manipulation project code, scripts, assets,
and reproducibility entry points. It is not the canonical experiment journal.

## Scope

- Robot/task: Unitree G1 upper-body language-conditioned pick/place and warehouse manipulation.
- Main technical line: VLA action proposal + world model / MPC screening + MuJoCo contact-only evaluation.
- Current status: paused. The project did not reach the final end-to-end VLA / world-model grasp success target.
- Historical positive results include modular RGB/contact-controlled baselines and language-routing demos; these are not claimed as end-to-end SmolVLA success.

## Canonical record

All project thinking, experiment logs, result summaries, failure analysis, and
claim boundaries are maintained in the single workspace journal:

```text
/home/ubuntu/VLA-Learning/notes/Lerobot.md
```

Do not add a second G1 experiment log in this repository. If a future G1
experiment resumes, record the experiment number, command, metrics, videos,
conclusion, and next step in `notes/Lerobot.md`.

## Repository role

This repository should contain:

- source code and reusable scripts;
- lightweight configuration files;
- small reproducibility manifests;
- concise project-level README information.

This repository should not contain:

- duplicated learning notes;
- separate final reports that can drift from `notes/Lerobot.md`;
- runtime logs such as `MUJOCO_LOG.TXT`;
- large generated outputs, videos, datasets, or checkpoints.

## Important paths

```text
scripts/       Experiment, training, evaluation, and audit scripts
resources/     Lightweight project resources
sim2sim/       Sim2Sim preparation / replay helpers
outputs/       Ignored generated outputs, when present locally
```

## Current reproducibility note

Many historical commands in old logs used the former path
`/home/ubuntu/G1-UpperBody`. The current workspace consolidation moved the
project under:

```text
/home/ubuntu/VLA-Learning/projects/G1-UpperBody
```

Before rerunning historical commands, update paths accordingly and verify that
the referenced assets/checkpoints still exist.

## Unified lifecycle entry point

The project now has one lightweight CLI for workspace checks and historical
workflow compatibility. It does not launch a simulator unless you explicitly
run one of the mapped workflows:

```bash
conda activate g1ub_genesis
python -m pip install -e . --no-deps
g1-vla doctor
g1-vla prepare
g1-vla smoke
g1-vla collect --help
g1-vla train-vla --help
g1-vla train-world-model --help
g1-vla eval --help
g1-vla play --help
g1-vla report
```

Every non-dry run is recorded under `.runtime/runs/G1-*/`. The old Python
scripts remain compatibility implementations; no G1 training or evaluation
was started by this restructuring.

## Current G1 selector line

The one-step geometry world model is still available for backward-compatible
inspection, but the preferred selector path is now the full-chunk task-value
world model:

```bash
/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python scripts/train_g1_task_value_world_model.py \
  --cache-dir <branch_cache_dir> \
  --output-dir <task_value_output_dir>

/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python scripts/evaluate_g1_policy_intervention_wm_mpc.py \
  --asset <asset_path> \
  --train-root <train_root> \
  --vla-checkpoint <vla.pt> \
  --world-model-checkpoint <one_step_world.pt> \
  --task-value-world-model-checkpoint <task_value_world.pt> \
  --output-dir <eval_dir>
```

This selector is only valid once the branch cache contains real stable-grasp
positives. If the data gate fails, collect better candidate branches first.
