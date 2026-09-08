# G1 VLA Grasping

**G1 VLA Grasping** is a compact MuJoCo research codebase for language-conditioned object grasping and pick-and-place with the Unitree G1 upper body.

The project focuses on turning a natural-language instruction such as *“put the yellow rod into the blue box”* into a grounded, contact-aware bimanual manipulation episode.

> This is an intentionally small public snapshot. It contains reusable source code and a lightweight visual preview, but no checkpoints, datasets, experiment logs, private robot assets, or machine-specific paths.

## What it demonstrates

- **Language grounding** — map an object description to a scene object.
- **Visual target localization** — use RGB observations to estimate the target in the tabletop scene.
- **Bimanual reach and grasp** — coordinate arms, palms, fingers, contact geometry, and object-specific closure profiles.
- **Pick-and-place** — approach, close, transport, and place the selected object into the blue receptacle.
- **Reproducible episodes** — emit local summaries and optional rendered artifacts for inspection.
- **Policy integration points** — provide adapters and evaluation entry points without bundling a trained policy.

## Pipeline

```text
language instruction
        │
        ▼
object selection / RGB target localization
        │
        ▼
reachability and contact-aware target generation
        │
        ▼
 bimanual arm + finger actuation
        │
        ▼
 grasp → lift → transport → place
        │
        ▼
 episode summary / visual inspection
```

The public implementation keeps target generation and actuation explicit so that failures can be inspected in the simulator. Object-specific offsets, hand closure scales, grasp yaw, and collision dimensions are part of the scene contract rather than hidden in a checkpoint.

## Public demo

A small MuJoCo reference preview is included for a quick visual overview. It is an illustrative rollout, not a claim of end-to-end VLA benchmark performance or real-robot success.

![G1 grasping rollout](media/g1-grasping-rollout.jpg)

## Requirements

Install the minimal public runtime dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The core simulation uses Python 3.10+, MuJoCo 3.3.x, NumPy, SciPy, ImageIO, and Pillow. Policy evaluation and LeRobot integration require an additional compatible PyTorch/LeRobot environment.

The G1 XML model and mesh assets are not redistributed here. Provide a compatible local scene asset through `--asset`; its mesh paths must resolve on the local machine.

## Quick start

```bash
python scripts/run_g1_language_pick_place.py \
  --asset /path/to/g1_scene.xml \
  --target-object yellow_rod \
  --output-dir /tmp/g1-language-demo
```

The reference scene supports these target identifiers:

```text
red_triangle
yellow_rod
green_cube
```

Generated files are runtime data and should remain outside the repository.

## Validation and integration entry points

```bash
python scripts/validate_g1_bimanual_actuation.py \
  --asset /path/to/g1_scene.xml \
  --output-dir /tmp/g1-actuation-check

python scripts/play_g1_language_episode.py --help
python scripts/collect_g1_language_pick_place_dataset.py --help
python scripts/evaluate_g1_language_pick_place.py --help
```

The validation script checks simulator-side actuation, contact, joint-limit, and saturation behavior. Passing it does not establish grasp robustness, policy generalization, or real-hardware safety.

## Repository layout

| Path | Role |
| --- | --- |
| `scripts/run_g1_language_pick_place.py` | Reference language-conditioned pick-and-place episode generator |
| `scripts/validate_g1_bimanual_actuation.py` | Bimanual actuation and contact checks |
| `scripts/g1_finger_contact_geometry.py` | Finger/contact-pad geometry helpers |
| `scripts/g1_language_action_adapter.py` | Optional language/action adapter boundary |
| `scripts/collect_g1_language_pick_place_dataset.py` | Reproducible local collection wrapper |
| `scripts/evaluate_g1_language_pick_place.py` | Optional closed-loop evaluation entry point |
| `scripts/play_g1_language_episode.py` | Local episode viewer |
| `requirements.txt` | Minimal public Python dependencies |
| `media/` | Small public preview assets only |

## Design principles

1. **No privileged target pose in the intended policy path.** Use permitted visual/state observations instead of simulator-only object coordinates.
2. **Contact is observable and inspectable.** Grasp quality should be checked through contact and object motion, not inferred from a hidden equality constraint.
3. **Actuation remains explicit.** Gains, targets, limits, and hand closure profiles are inspectable in source.
4. **Evaluation is separate from training.** Offline loss is not a pick-and-place success metric; use held-out layouts and per-episode outcomes.
5. **Public code stays small.** Models, datasets, logs, and generated outputs belong in private storage.

## Scope and limitations

This repository covers the G1 upper-body grasping direction. It is not a complete Unitree hardware driver, whole-body locomotion stack, or production deployment package. The reference path is simulator-oriented and depends on a compatible local G1 asset.

Results depend on the exact XML model, mesh versions, actuator parameters, camera calibration, object layout, contact solver settings, and evaluation seed. Do not interpret the demo image or the presence of an evaluation script as a performance claim.

## Reproducibility checklist

For each private experiment, preserve the code/asset revision, object layout, instruction, dynamics and contact settings, controller profile, random seed, per-episode success, grasp retention, placement error, and failure reason. Keep these records in private experiment storage rather than committing them here.
