# G1 VLA Grasping

G1 is being rebuilt as a fixed-stance, strict-contact manipulation system. The preparation revision has one active configuration and one package entry point. Training and world-model optimization are disabled.

## Workspace

The development directory is `/home/ubuntu/VLA-Learning/projects/G1-VLA-Grasping`. Robot meshes remain in the workspace's `frameworks/unitree_lerobot` repository. The exact 31-joint order is read from `resources/g1_action31_train_root_stub/meta/info.json`; this file is a protocol stub, not training data.

All experiment history and new implementation records belong in `../../notes/Lerobot.md`. This README describes operation and boundaries, not a second experiment journal.

## Preparation commands

```bash
conda activate g1ub_genesis
python -m pip install -e . --no-deps --no-build-isolation
g1-vla experiment --config configs/experiment.json --stage plan
g1-vla experiment --config configs/experiment.json --stage inventory
g1-vla experiment --config configs/experiment.json --stage registry
g1-vla experiment --config configs/experiment.json --stage audit
g1-vla experiment --config configs/experiment.json --stage report
```

`plan` and `--dry-run` are read-only. Inventory and registry create exclusive evidence files; they never delete, move or deserialize training artifacts. The current `train` stage records a refusal without creating an optimizer. Editing the configuration cannot enable optimization in this preparation revision. Collection, dataset build, replay and physical policy evaluation are not yet admitted runners; selecting them returns an explicit blocked result and never invokes historical scripts.

## Contracts and evidence

- `configs/experiment.json` is the only active configuration; unknown or duplicate JSON keys are rejected.
- Actions are eight future joint targets relative to one planning state, with four commands executed per plan. Joint ordering, normalization and vocabulary are checkpoint compatibility requirements.
- The historical controller called its rate 15 Hz but advanced 33 physics steps of 0.002 s per command. The configuration records the actual `500/33` Hz cadence separately. No timing change is hidden in the migration.
- The new adjacent-command speed limit has not been physically audited and remains null. The execution gate refuses incomplete limits. This is not a hardware-calibrated controller.
- Raw successful and failed episodes, source rollouts, journal and research baselines are protected. Historical caches are not automatically accepted as new training data.
- Old evaluations become regression evidence. A new independent holdout must be sealed before future model selection.
- Event metrics separate physical contact, sustained lift, stable released placement and strict safety. An in-box flag or text classification alone cannot establish task success.

## Historical code

The original source is preserved by Git tag `codex/g1-before-rebuild-20261014` and under `legacy/`. The 154 experiment scripts and former script-forwarding lifecycle are no longer active package entry points. Historical checkpoint 672 remains a research baseline, explicitly separate from a new-contract deployment checkpoint.

The latest archived matched evaluations 688/690/691 each had zero real contact and zero complete success in six episodes. Historical hybrid successes used classical control and assisted grasp; they are not end-to-end VLA results.

See `docs/ARCHITECTURE.md` for package responsibilities and the current preparation boundary.
