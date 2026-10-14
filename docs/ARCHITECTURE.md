# G1 rebuild architecture

## Single entry

`g1-vla experiment --config configs/experiment.json --stage <stage>` uses validated configuration and direct package functions. There is no arbitrary argument forwarding or shell-script fallback. Preparation is explicit; no stage chains into training.

## Package responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Strict single JSON schema, resolved paths and no-training preparation settings |
| `contracts.py` | Joint order, anchored action codec, sensor boundary and checkpoint compatibility |
| `control.py` | Shared dynamics/PD profile and bounded physical-command execution |
| `control_limits.py` | Source-hashed green/right command-envelope audit |
| `data_pairing.py` | Exact pre-command/post-command pairing, historical rejection and eight-step chunk consistency |
| `expert.py`, `teacher_acceptance.py` | Privileged green/right physical teacher and frozen-layout acceptance; no policy fallback |
| `observation.py`, `sim/scene.py` | Sensor-only observation and strict scene preparation |
| `policy.py` | Self-contained continuous model and explicit historical-checkpoint loading |
| `training.py` | Aligned dataset/statistics/loss/resume primitives and disabled optimizer gate |
| `evaluation.py` | Contact/lift/placement events and strict success metrics |
| `artifacts.py` | Read-only inventory and protection classification |
| `data_registry.py`, `data_ingest.py` | Raw provenance, historical quarantine and connected source/layout splits |
| `runtime/experiment.py` | Immutable config, provenance and result records |

## Data boundary

The policy receives synchronized RGB, joint position/velocity, robot forward-kinematic finger samples, sensor-equivalent tactile, language and causal history. Simulator objects, target coordinates, object IDs and phase truth are not passed to it. Teacher labels and evaluation truth stay outside policy inputs.

Historical raw episodes enter a diagnostic/regression registry first. A reported success without the new sustained-hold evidence is not silently relabeled as a new positive. Direct corrections and safe-waypoint recoveries have distinct regimes. Layout and original-rollout connectivity determines split membership; language augmentation and multiple takeovers do not create independent groups.

## Immutable run artifacts

Each explicit preparation stage writes `effective_config.json`, `provenance.json`, `started.json`, `finished.json`, `summary.json` and optional manifests/reports into a unique run directory. Existing evidence files and run IDs cannot be overwritten. The configuration records both nominal and measured execution cadence. Artifact inventory excludes its own generated run directory.

## Current boundary

No optimizer runner is installed in this revision. Limits/pairing and physical teacher stages are available through the single entry. The teacher uses the shared executor, captures RGB and exact command intervals, and evaluates a frozen local layout suite. Command bounds, measured joint tracking, torque saturation, substep contact, sustained lift and chunk consistency are recorded separately. A passing format audit does not establish physical grasp success; a passing teacher does not establish a learned policy result. Collection, training-cache build, replay and physical policy evaluation remain blocked, without historical fallback. All new raw evidence remains diagnostic pending future dataset admission.

World model, MPC, left-hand expansion and Sim2Sim remain deferred. Every implementation step and its validation belongs in the existing `notes/Lerobot.md` journal.
