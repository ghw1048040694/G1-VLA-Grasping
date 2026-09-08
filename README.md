# G1 VLA Grasping

A compact research codebase for language-conditioned object grasping and
pick-and-place with the Unitree G1 upper body in MuJoCo.

## Scope

The public snapshot focuses on the reusable grasping path:

- language-grounded object selection;
- RGB-conditioned pick-and-place episode generation;
- contact-aware bimanual actuation and finger geometry;
- dataset collection and optional policy evaluation.

This repository contains source code only. Checkpoints, datasets, videos,
experiment logs, and machine-specific assets are intentionally excluded.

## Requirements

Install the dependencies from `requirements.txt`. Policy evaluation additionally
requires a compatible PyTorch/LeRobot environment.

The G1 MuJoCo model and mesh assets are not redistributed here. Provide a
compatible G1 XML model through `--asset`; its mesh paths must be valid on the
local machine.

## Example

Generate one contact-aware language-conditioned episode:

```bash
python scripts/run_g1_language_pick_place.py \\
  --asset /path/to/g1_scene.xml \\
  --target-object green_cube \\
  --output-dir outputs/example
```

The output directory is local runtime data and should not be committed.

## Layout

- `scripts/run_g1_language_pick_place.py`: core MuJoCo grasp/pick-and-place episode.
- `scripts/validate_g1_bimanual_actuation.py`: actuation and contact checks.
- `scripts/g1_finger_contact_geometry.py`: finger-contact geometry helpers.
- `scripts/collect_g1_language_pick_place_dataset.py`: reproducible collection wrapper.
- `scripts/evaluate_g1_language_pick_place.py`: optional closed-loop policy evaluation.
- `scripts/g1_language_action_adapter.py`: optional language/action adapter.
- `scripts/play_g1_language_episode.py`: local episode viewer.

Results are not implied by the presence of the scripts; evaluate on held-out
layouts and report success metrics with the exact model and environment used.

## Demo preview

A small MuJoCo preview is included for a quick visual overview of the
manipulation scene. It is an illustrative rollout/reference trajectory, not a
claim of end-to-end VLA benchmark performance.

![G1 grasping rollout](media/g1-grasping-rollout.jpg)

[Download the G1 grasping rollout video](media/g1-grasping-rollout.mp4)
