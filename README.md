# G1 VLA Grasping

G1 is being rebuilt as a fixed-stance, strict-contact manipulation system. There is one active configuration and one package entry point. A bounded source-native ACT trial is conditionally enabled, with verified data, independent splits and measured training memory required before optimization. World-model optimization remains disabled.

## Workspace

The development directory is `/home/ubuntu/VLA-Learning/projects/G1-VLA-Grasping`. Robot meshes remain in the workspace's `frameworks/unitree_lerobot` repository. The exact 31-joint order is read from `resources/g1_action31_train_root_stub/meta/info.json`; this file is a protocol stub, not training data.

All experiment history and new implementation records belong in `../../notes/Lerobot.md`. This README describes operation and boundaries, not a second experiment journal.

## Preparation commands

Physical teacher acceptance requires `train.enabled=false`; it preserves the existing no-optimization acceptance boundary. The native ACT trial uses its own explicit backend and stages below. Switching stages does not invoke another stage automatically.

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
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage external-reconcile
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage weights-audit
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage offline-fetch
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage offline-infer
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage source-native-fetch
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage build
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage grasp-memory
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage train
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage eval
```

The official GraspSquare sample uses pinned episodes 0/150/300. Downloads verify exact revisions and LFS hashes. Video probes fetch bounded HTTP ranges and preserve source PTS; range hashes do not claim full-video integrity. When transport fails, `external-reconcile` verifies configured immutable reports against the same full source hashes, sample plan and actual PNG pixels. It requires all 36 planned frames and agrees on duplicate successful frames; it never hides failed attempts or treats partial probes as complete. Data keeps its native 28 dimensions and missing waist values remain missing. Model-source joint geometry supports an identity mapping, while acquisition clocks, calibration and current-task conversion need separate evidence.

The compact ACT probe uses its actual original pizza-box dataset, saved normalization, a complete strict checkpoint load and FP32 CUDA inference. Its 29-state/20-action protocol includes remote axes and binary hand-open commands. The measured output is a compute baseline, without language conditioning, simulated rollout or adaptation to the current 31-joint policy. Large VLA metadata estimates and gated dependencies are reported separately; estimates never count as successful inference.

`plan` and `--dry-run` are read-only. Inventory and registry create exclusive evidence files; they never delete, move or deserialize training artifacts. No stage automatically starts training. `train.backend=grasp_square_act` selects the separate source-native build, bounded optimizer and offline evaluation; `current_action31` remains unadmitted for optimization. Collection, replay and physical policy evaluation remain blocked and never invoke historical scripts. Failure of the current green/right teacher does not invalidate official demonstrations.

The native trial predeclares twelve new episodes: six train, three validation and three test, excluding the prior discovery episodes 0/150/300. All four original camera file sets are disjoint between splits; acquisition session and layout independence are unknown. Only the original left head eye is a model input. Complete pinned videos supply every row at 30 Hz, explicitly resized from 640×480 to 320×240, with original 28D float32 state/actions and complete 16-step future windows. Statistics come only from the six training episodes. The recorded task text `camera packaging` is preserved despite its disagreement with the data card.

The small ACT starts from scratch, including its ResNet18 backbone. It does not reuse or pad the pizza-box checkpoint's incompatible 20D action head. `grasp-memory` performs one temporary real forward/backward/AdamW step and the configured evaluation batch, discards its parameters, and binds the observed memory to the exact data/configuration/model code. `train` requires that report and completes at most 200 steps and 1800 seconds with a 7 GiB PyTorch allocator cap. Validation recorded-action MAE selects the checkpoint every 50 steps. `eval` strictly reloads that complete checkpoint and consumes the test split once; state-hold and training-mean baselines use the same targets. These errors do not establish grasp success, language conditioning, real-time control or a 31-joint adapter. Trial checkpoints are new safetensors artifacts and do not provide resume state; existing run directories and checkpoints cannot be overwritten.

ACT failure diagnosis and controlled ablations use the same configuration and entry point:

```bash
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage diagnose
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage act-diagnose
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage act-ablate
```

`diagnose` independently compares only the nine train/validation episodes with their pinned original rows, metadata, decoded PTS and actual dataset indexing. It measures state/action lag and future-horizon baselines without shifting labels, separates observed joint motion groups, and replays the original sample coverage and L1/KL losses. Hardware acquisition and applied-command clocks remain unknown; a preferred statistical lag never certifies a label correction.

`act-diagnose` verifies the original checkpoint against its saved configuration and runtime, then evaluates validation observations with real images, constant training-channel means, a nonidentity within-episode image shift and an image donor from another validation episode. State and targets stay fixed. `act-ablate` compares RGB, constant-image, genuinely state-only and no-CVAE models using identical compatible initial parameters and the same sampled anchors. The state-only adapter creates no visual or environment-state feature, token or projection; an alias of the same observed joints only supplies the original ACT library's batch-size lookup. Removing CVAE tests the whole variational mechanism, and does not isolate KL weight alone. Each variant has a discarded real memory probe and immutable validation-selected checkpoints. The RGB trajectory must reproduce the original first 200 steps before any longer budget is interpreted. Loss, gradient groups, coverage and label-free prior inference are reported separately. No diagnostic stage reads test arrays/images or performs test inference; extending the budget or trying pretrained initialization requires a new declared run and output directory.

Visual backbone initialization is compared through the same configuration and entry point:

```sh
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage vision-fetch
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage vision-ablate
```

`visual_pretraining` declares four models, all without VAE: RGB from scratch, RGB with TorchVision ResNet18 ImageNet1K V1 initialization, constant training-channel mean images, and state-only. State-hold uses the same evaluation labels and denominator. All models keep the existing native train/validation split, shared compatible head initialization, seed, anchor schedule, optimizer and 2000-step budget. The scratch RGB trajectory must reproduce the preceding no-VAE run before its comparison is accepted. Test payloads and previous test metrics are excluded.

The weight download is limited to the official TorchVision HTTPS URL and its published SHA-256 filename prefix. `vision-fetch` records the full file SHA-256; that full digest must be pinned in the same configuration before training. Loading is restricted to tensors, and every one of the 100 FrozenBatchNorm backbone state keys must match. The official archive has 102 keys, including two ImageNet classifier keys which are excluded. It stores no BatchNorm batch counters; a complete modern 122-key dictionary is also accepted with all 20 counters explicitly excluded, while partial counter sets are rejected. All other initial model tensors remain identical. This replaces convolution weights and FrozenBatchNorm parameters/statistics together, and does not import a public robot policy head.

Both visual models retain the same 320×240 source resize and train-derived RGB normalization. Neither receives a separate ImageNet crop or processor. Official ImageNet preprocessing differs, so this comparison measures initialization under the current recorded-data protocol; it is not a claim that all pretrained preprocessing choices or robot generalization have been evaluated. Each model receives a discarded real optimizer/evaluation memory probe before formal training. Runs, reports, selected checkpoints and predictions are saved in new directories without overwriting earlier evidence.

State-relative action prediction and trajectory coverage use the same entry point:

```sh
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage coverage-audit
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage residual-check
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage residual-train
```

`state_residual` fixes two no-VAE models: state-only and RGB from scratch. Both use the previous data split, compatible shared initial tensors, anchor schedule and 2000-step optimizer budget. Their final action head starts at zero. The label for every future row is `(raw_action[t+h] - raw_state[t]) / train_action_std`; there is no residual mean or changed scale. Reconstruction uses the original float32 source anchor, `raw_state[t] + train_action_std * prediction`. Zero residual explicitly copies the raw anchor, preserving exact state-hold values including signed zero. The anchor is never reconstructed from normalized model input. Training applies L1 directly to scaled residuals, so the exact-zero inference branch does not suppress the head gradient.

`residual-check` verifies zero identity and performs one discarded real optimizer/evaluation memory probe per model. `residual-train` requires that report with matching configuration, data, source, runtime and initial-model bindings, then starts fresh formal models without duplicating the admission probes. The first zero-head backward has zero upstream gradients by construction; AdamW weight decay may still update those parameters. Initial step 0 is a declared checkpoint candidate. Best-trained and last-trained results are reported separately, so selecting hold cannot be presented as learned improvement. Validation includes per-episode, horizon and anchor-motion comparisons using the same hold denominator. Test payloads and previous test scores are excluded.

`coverage-audit` separates sampled-anchor coverage from future-label-row coverage and audits only existing train payloads plus pinned source metadata. Its next-batch proposal is metadata-only: it does not download candidates, expand the active dataset or alter the split. File grouping is storage evidence; missing collection-session, scene and operator identities cannot be inferred from different episode numbers or MP4 shards. Candidate coverage expansion and genuinely independent session acquisition must be reported separately.

The next stage audits the eight predeclared same-source coverage candidates before expanding train. Use the same CLI and configuration:

```sh
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage expansion-audit
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage expansion-build
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage expanded-check
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage expanded-train
```

`train_expansion` binds the previous coverage/residual reports and a pinned session-source evidence file. The official data card describes multiple collection sessions, but provides no episode-to-session mapping for these eight candidates. Successful data admission therefore adds same-source coverage only. Audit checks original 28D labels, row and all-four-camera file identities, actual left-eye RGB/PTS pairing and unchanged parent validation payloads. Build writes a separate train/validation-only manifest, copies six original train and three original validation episodes without changing their bytes, adds eight admitted train episodes, and recomputes statistics from the fourteen train episodes only. Old test payloads are excluded.

`expanded_residual.data_manifest_sha256` must be set to the actual immutable build hash before admission or training. A zero placeholder is not training admission. Both residual models restart from the original compatible initialization and exact zero head; each retains 2000 optimizer steps, batch two, the same seed/LR, validation identities and checkpoint criterion. They share a new fixed sampler over the expanded anchor population. The larger dataset and its new train-only normalization are part of the intervention; old-model scores retain their original normalization. Reports separate original-train and added-train evaluations, total coverage, step-zero hold, best-trained and final results. No warm-start, extra training budget or test-based selection is implicit.

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

Normalization attribution and a train-only constant residual use the same entry point:

```sh
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage constant-residual --run-id G1-REBUILD-20261015-constant-residual-01
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage normalization-check --run-id G1-REBUILD-20261015-normalization-check-01
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage normalization-train --run-id G1-REBUILD-20261015-normalization-train-01
```

`normalization_attribution` trains two fresh zero-head, no-VAE residual models on the admitted fourteen train episodes while explicitly using the original six train episodes' statistics. Dataset statistics retain their real fourteen-episode origin; checkpoints separately bind the six-episode model normalizer. Both models replay the expanded reference's exact initialization, RNG and sampled anchors, with 2000 steps per arm and unchanged validation. The existing six-train/old-statistics and fourteen-train/new-statistics results are immutable references, without new training or inference. This follows one path through data membership and a joint normalizer change; it does not identify their interaction or separate state, image and residual-scale effects. Only raw radian metrics are compared across normalizers.

`constant_residual` fits a fixed [16,28] raw residual on every complete train anchor, once for the original six episodes and once for all fourteen. Source FP32 labels and anchors are subtracted in FP64, the pooled coordinate median uses the midpoint for even populations, and parameters are saved in FP32. Inference adds the saved constant to raw current joints, preserving raw anchor bits for exact zero parameters. It uses no RGB, validation fitting, scaling or checkpoint selection. This is a closed-form train fit with zero optimizer steps; using every train anchor does not match the neural training compute budget. Original six, added eight, all fourteen and the frozen validation episodes share the same evaluation stride, with fit membership recorded explicitly. Test payloads remain excluded.

Frozen state conditioning uses the same configuration and entry point:

```sh
/home/ubuntu/VLA-Learning/.envs/g1-audit/bin/python -m g1_vla.cli experiment --config configs/experiment.json --stage state-conditioning --run-id G1-REBUILD-20261015-state-conditioning-01
```

`state_conditioning` fixes the existing fourteen-train/new-statistics state-only and RGB selected checkpoints at step 1650. It creates no optimizer and performs no gradient updates or memory probes. The four predeclared network-state inputs are real, normalized-zero training mean, a half-length cyclic shift within each episode's evaluation anchors, and a donor from the next sorted episode at the same relative rank. Donors come only from the same split, using identities rather than labels or errors. Raw decoder anchors, labels and RGB observations always remain those of the original sample. The real condition must reproduce saved predictions; state-only donor outputs must match the donor's real residual, and constant state inputs must produce constant residuals within fixed numerical tolerance.

Both models retain state-hold and the existing fourteen-train label-median residual control. A separate diagnostic collapses each model's real outputs at all 468 fixed train evaluation anchors into one coordinate-median raw residual, using FP32 scale multiplication, FP64 median fitting and FP32 saved parameters. Its fit cannot receive validation observations or labels. The frozen 85 validation anchors are scored with paired errors and episode, joint, horizon and motion summaries; there is no new checkpoint or hyperparameter selection. This closed-form output collapse is distinct from the earlier all-train-label median and is not a compute-matched neural training arm. Mean and mismatched-state interventions may be out of distribution, so changes support evidence about state dependence and correspondence, without certifying dynamics, causality, independent-session generalization or physical success.
