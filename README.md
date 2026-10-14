# G1 VLA Grasping

G1 is being rebuilt as a fixed-stance, strict-contact manipulation system. There is one active configuration and one package entry point. Training is conditionally authorized and defaults off until an audited backend and dataset split are admitted; world-model optimization remains disabled.

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
g1-vla experiment --config configs/experiment.json --stage limits
g1-vla experiment --config configs/experiment.json --stage pairing
MUJOCO_GL=egl g1-vla experiment --config configs/experiment.json --stage teacher-smoke
MUJOCO_GL=egl g1-vla experiment --config configs/experiment.json --stage teacher
g1-vla experiment --config configs/experiment.json --stage report
```

External-data and original-policy probes use the isolated audit environment so the physical environment remains stable:

```bash
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage external-fetch
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage external-audit
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage weights-audit
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage offline-fetch
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage offline-infer
```

The official GraspSquare sample uses pinned episodes 0/150/300. Downloads verify exact revisions and LFS hashes. Video probes fetch bounded HTTP ranges and preserve source PTS; range hashes do not claim full-video integrity. Data keeps its native 28 dimensions and missing waist values remain missing. Model-source joint geometry supports an identity mapping, while acquisition clocks, calibration and current-task conversion need separate evidence.

The compact ACT probe uses its actual original pizza-box dataset, saved normalization, a complete strict checkpoint load and FP32 CUDA inference. Its 29-state/20-action protocol includes remote axes and binary hand-open commands. The measured output is a compute baseline, without language conditioning, simulated rollout or adaptation to the current 31-joint policy. Large VLA metadata estimates and gated dependencies are reported separately; estimates never count as successful inference.

`plan` and `--dry-run` are read-only. Inventory and registry create exclusive evidence files; they never delete, move or deserialize training artifacts. `train.enabled` is a conditional request, and the current training stage records missing prerequisites without constructing an optimizer. No stage automatically starts training. Collection, dataset build, replay and physical policy evaluation are not yet admitted runners; selecting them returns an explicit blocked result and never invokes historical scripts. Public source-native imitation is a separate candidate pipeline: failure of the current green/right teacher does not invalidate official demonstrations.

## Contracts and evidence

- `configs/experiment.json` is the only active configuration; unknown or duplicate JSON keys are rejected.
- Actions are eight future joint targets relative to one planning state, with four commands executed per plan. Joint ordering, normalization and vocabulary are checkpoint compatibility requirements.
- The historical controller called its rate 15 Hz but advanced 33 physics steps of 0.002 s per command. The configuration records the actual `500/33` Hz cadence separately. No timing change is hidden in the migration.
- The 31 adjacent-command speed limits come from the observed successful green/right grasp-lift command envelope, rounded upward to 0.01 rad/s. Zero holds the command fixed. This is a simulation command bound, not hardware calibration; `limits` preserves the source audit.
- `pairing` audits raw observation/command timing without repairing, resampling or clipping historical labels. Old sampled commands cannot prove next actual control-command pairing.
- `teacher-smoke` checks the historical source position separately; `teacher` runs the predeclared 20 unique layouts in `configs/teacher_layouts.json`. The formal gate requires at least 18/20 strict contact lifts of 0.1 m held for 0.5 s, complete RGB pairs, finite physics and declared joint safety. This local grid does not cover the full workspace.
- Teacher commands use the same anchored codec, velocity limiter and PD executor. Post-step `mj_forward` synchronizes sensors/FK/contact without advancing time. Raw evidence stays diagnostic with training disabled even when a physical episode passes. Inconsistent eight-step chunks are reported and never clipped into compatible labels.
- Raw successful and failed episodes, source rollouts, journal and research baselines are protected. Historical caches are not automatically accepted as new training data.
- Old evaluations become regression evidence. A new independent holdout must be sealed before future model selection.
- Event metrics separate physical contact, sustained lift, stable released placement and strict safety. An in-box flag or text classification alone cannot establish task success.

## Historical code

The original source is preserved by Git tag `codex/g1-before-rebuild-20261014` and under `legacy/`. The 154 experiment scripts and former script-forwarding lifecycle are no longer active package entry points. Historical checkpoint 672 remains a research baseline, explicitly separate from a new-contract deployment checkpoint.

The latest archived matched evaluations 688/690/691 each had zero real contact and zero complete success in six episodes. Historical hybrid successes used classical control and assisted grasp; they are not end-to-end VLA results.

See `docs/ARCHITECTURE.md` for package responsibilities and the current preparation boundary.
