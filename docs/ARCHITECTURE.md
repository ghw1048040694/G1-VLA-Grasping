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
| `teacher_evidence.py`, `teacher_video.py` | Actual sensor videos, separate diagnostic renders, phase scores and frame/time provenance, including failures |
| `observation.py`, `sim/scene.py` | Sensor-only observation and strict scene preparation |
| `policy.py` | Self-contained continuous model and explicit historical-checkpoint loading |
| `training.py` | Aligned dataset/statistics/loss/resume primitives and disabled optimizer gate |
| `evaluation.py` | Contact/lift/placement events and strict success metrics |
| `artifacts.py` | Read-only inventory and protection classification |
| `data_registry.py`, `data_ingest.py` | Raw provenance, historical quarantine and connected source/layout splits |
| `external_sources.py`, `external_data.py`, `video_audit.py` | Bounded revision-pinned sources, native row/joint/camera/time audits and partial-video PTS evidence |
| `weight_audit.py`, `offline_inference.py` | Public checkpoint metadata and measured original-protocol offline compute; no controller handoff |
| `runtime/experiment.py` | Immutable config, provenance and result records |

## Data boundary

The policy receives synchronized RGB, joint position/velocity, robot forward-kinematic finger samples, sensor-equivalent tactile, language and causal history. Simulator objects, target coordinates, object IDs and phase truth are not passed to it. Teacher labels and evaluation truth stay outside policy inputs.

Historical raw episodes enter a diagnostic/regression registry first. A reported success without the new sustained-hold evidence is not silently relabeled as a new positive. Direct corrections and safe-waypoint recoveries have distinct regimes. Layout and original-rollout connectivity determines split membership; language augmentation and multiple takeovers do not create independent groups.

## Immutable run artifacts

Each explicit preparation stage writes `effective_config.json`, `provenance.json`, `started.json`, `finished.json`, `summary.json` and optional manifests/reports into a unique run directory. Existing evidence files and run IDs cannot be overwritten. The configuration records both nominal and measured execution cadence. Artifact inventory excludes its own generated run directory.

## Current boundary

Training authorization is conditional; the current action31 builder/optimizer runner remains unadmitted. Limits/pairing, physical teacher, external source audits and original-policy offline inference are available through the single entry. The teacher uses the shared executor, captures RGB and exact command intervals, and evaluates a frozen local layout suite. Command bounds, measured joint tracking, torque saturation, substep contact, sustained lift and chunk consistency are recorded separately. A passing format audit does not establish physical grasp success; a passing teacher or original-policy compute probe does not establish current learned-policy success. Collection, training-cache build, replay and physical policy evaluation remain blocked, without historical fallback.

External source rows and current physical interval pairs have different contracts. GraspSquare keeps its native action28, four real views and 30Hz row cadence; three waist values and physical acquisition times are unavailable. Native external imitation can be validated independently rather than being rejected by the local teacher gate. Admission to the current green/right task still requires an explicit sensor/action/time bridge, source grouping and closed-loop evaluation. No missing observations or successful labels are fabricated during an audit.

World model, MPC, left-hand expansion and Sim2Sim remain deferred. Every implementation step and its validation belongs in the existing `notes/Lerobot.md` journal.
