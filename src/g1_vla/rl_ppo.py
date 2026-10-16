"""Bounded native Isaac Lab / RSL-RL 5 execution behind the project CLI.

Importing this module neither imports Isaac nor creates an optimizer.  Simulator
startup is confined to :func:`run_native_stage`.  PPO uses the installed RSL-RL
algorithm unchanged; the small loop here adds deadlines and auditable optimizer
and environment records around its public methods.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NATIVE_STAGES = frozenset(("rl-audit", "rl-smoke", "rl-train", "rl-eval"))
RUNNER_VERSION = "g1_rsl5_bounded_ppo_v1"


class BudgetExpired(RuntimeError):
    """The predeclared monotonic wall-clock budget has expired."""


class NumericalFailure(RuntimeError):
    """A nonfinite state, loss, gradient, or parameter was observed."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NumericalFailure("Nonfinite value cannot enter the evidence stream")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def _write_json_new(path: Path, value: Any) -> None:
    encoded = json.dumps(_json_safe(value), indent=2, ensure_ascii=False,
                         allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


class EvidenceStream:
    """Append records to a new file; pre-existing evidence is never replaced."""

    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("x", encoding="utf-8", newline="\n")
        self.count = 0

    def write(self, row: dict) -> None:
        self.stream.write(json.dumps(_json_safe(row), ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")) + "\n")
        self.stream.flush()
        self.count += 1

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


@dataclass
class Deadline:
    seconds: float
    clock: Any = time.monotonic

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or not 0 < self.seconds <= 3600:
            raise ValueError("Native wall-clock budget must be in (0, 3600] seconds")
        self.started = self.clock()

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started

    def check(self) -> None:
        if self.elapsed >= self.seconds:
            raise BudgetExpired(f"Native wall-clock budget {self.seconds:g}s expired")


def build_rsl_config(settings: dict) -> dict:
    """Construct the explicit installed RSL-RL 5 actor/critic configuration."""
    ppo = settings["ppo"]
    hidden = list(ppo["hidden_dims"])
    if hidden != [128, 128]:
        raise ValueError("The first RL teacher budget pins actor/critic to [128, 128]")
    rollout = int(ppo["rollout_steps"])
    epochs = int(ppo["epochs"])
    if rollout != 32 or epochs != 4:
        raise ValueError("The initial PPO protocol pins rollout_steps=32 and epochs=4")
    return {
        "seed": int(settings["train_seed"]),
        "device": settings["device"],
        "num_steps_per_env": rollout,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "check_for_nan": True,
        "save_interval": int(ppo["validation_interval"]),
        "logger": "tensorboard",
        "actor": {
            "class_name": "rsl_rl.models:MLPModel", "hidden_dims": hidden,
            "activation": "elu", "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": float(ppo["initial_action_std"]), "std_type": "log",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models:MLPModel", "hidden_dims": hidden,
            "activation": "elu", "obs_normalization": True,
            "distribution_cfg": None,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:PPO",
            "num_learning_epochs": epochs,
            "num_mini_batches": int(ppo["minibatches"]),
            "clip_param": float(ppo["clip_param"]), "gamma": float(ppo["gamma"]),
            "lam": float(ppo["lam"]),
            "value_loss_coef": 1.0, "entropy_coef": float(ppo["entropy_coef"]),
            "learning_rate": float(ppo["learning_rate"]), "max_grad_norm": float(ppo["max_grad_norm"]),
            "optimizer": "adam", "use_clipped_value_loss": True,
            "schedule": "adaptive", "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None, "symmetry_cfg": None,
        },
    }


def normalize_settings(flat: dict) -> dict:
    """Map the unified JSON configuration to runner-only names without mutation."""
    return {
        "train_seed": flat["seed"], "eval_seed": flat["evaluation_seed"],
        "smoke_seed": flat["smoke_seed"], "device": flat["device"],
        "num_envs": flat["num_envs"],
        "runtime": {"git_executable": flat["git_executable"], "asset_path": flat["asset_path"]},
        "smoke": {"num_envs": list(flat["smoke_env_counts"]), "control_steps": flat["smoke_steps"]},
        "evaluation": {"episodes": flat["eval_episodes"], "num_envs": flat["eval_num_envs"],
                       "max_control_steps": math.ceil(flat["episode_length_s"] /
                           (flat["physics_dt_s"] * flat["decimation"])) *
                           math.ceil(flat["eval_episodes"] / flat["eval_num_envs"]) * 2,
                       "checkpoint_path": flat.get("checkpoint_path"),
                       "checkpoint_sha256": flat.get("checkpoint_sha256")},
        "ppo": {"hidden_dims": list(flat["hidden_dims"]), "rollout_steps": flat["steps_per_env"],
                "epochs": flat["num_learning_epochs"], "minibatches": flat["num_mini_batches"],
                "max_updates": flat["max_iterations"], "validation_interval": flat["eval_interval"],
                "initial_action_std": flat["init_noise_std"], "entropy_coef": flat["entropy_coef"],
                "learning_rate": flat["learning_rate"], "wall_time_seconds": flat["max_wall_time_s"],
                "gamma": flat["gamma"], "lam": flat["lam"], "clip_param": flat["clip_param"],
                "max_grad_norm": flat["max_grad_norm"]},
    }


def policy_training_identity(config: dict) -> str:
    """Ignore execution locations and checkpoint selection, retain task semantics."""
    host_fields = frozenset(("runtime_windows_root", "isaac_python_bat", "isaaclab_path",
                             "git_executable", "asset_path", "seed_state_path", "asset_manifest_path",
                             "checkpoint_path", "checkpoint_sha256", "num_envs", "smoke_env_counts",
                             "smoke_steps", "max_iterations", "max_wall_time_s", "eval_interval",
                             "eval_episodes", "eval_num_envs", "save_interval"))
    rl = {key: value for key, value in config["rl_teacher"].items() if key not in host_fields}
    action = copy.deepcopy(config.get("action"))
    if isinstance(action, dict):
        metadata_path = action.pop("joint_metadata_path", None)
        if metadata_path is not None:
            action["joint_metadata_sha256"] = file_sha256(metadata_path)
    value = {"rl_teacher": rl, "action": action, "control": config.get("control"),
             "expert": config.get("expert"), "runner_version": RUNNER_VERSION}
    return canonical_sha256(value)


def source_tree_identity() -> str:
    root = Path(__file__).resolve().parent
    return canonical_sha256({str(path.relative_to(root)): file_sha256(path)
                             for path in sorted(root.rglob("*.py"))})


def _validate_settings(settings: dict) -> None:
    if not isinstance(settings, dict):
        raise ValueError("rl_teacher must be an object")
    ppo = settings["ppo"]
    if type(ppo["max_updates"]) is not int or not 1 <= ppo["max_updates"] <= 400:
        raise ValueError("Initial teacher budget is at most 400 PPO updates")
    if ppo["validation_interval"] != 50:
        raise ValueError("Validation interval must remain 50 PPO updates")
    if type(settings["num_envs"]) is not int or not 1 <= settings["num_envs"] <= 32:
        raise ValueError("The first task admits between 1 and 32 parallel environments")
    if list(settings["smoke"]["num_envs"]) != [1, 8, 32]:
        raise ValueError("Smoke parallelism is predeclared as [1, 8, 32]")
    for key in ("train_seed", "eval_seed", "smoke_seed"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if settings["train_seed"] == settings["eval_seed"]:
        raise ValueError("Training and evaluation seeds must differ")
    if settings["smoke"]["control_steps"] < 1 or settings["evaluation"]["episodes"] < 1:
        raise ValueError("Smoke and evaluation require positive declared budgets")
    for key in ("learning_rate", "initial_action_std", "max_grad_norm"):
        if not math.isfinite(ppo[key]) or ppo[key] <= 0:
            raise ValueError(f"PPO {key} must be finite and positive")
    Deadline(float(ppo["wall_time_seconds"]))
    build_rsl_config(settings)


def summarize_outcomes(outcomes: list[dict]) -> dict:
    """Summarize only explicit independent outcome booleans, never reward."""
    valid = []
    unavailable = []
    seen = set()
    gates: dict[str, dict[str, int]] = {}
    reasons: dict[str, int] = {}
    for row in outcomes:
        identity = (row.get("env_id"), row.get("episode_id"))
        if None in identity or identity in seen:
            raise ValueError("Missing or duplicate environment episode identity")
        seen.add(identity)
        success = row.get("success")
        if type(success) is bool:
            valid.append(success)
        else:
            unavailable.append({"env_id": identity[0], "episode_id": identity[1],
                                "reason": row.get("unavailable_reason", "success is unavailable")})
        for name, passed in row.get("gates", {}).items():
            count = gates.setdefault(name, {"passed": 0, "failed": 0, "unavailable": 0})
            count["passed" if passed is True else "failed" if passed is False else "unavailable"] += 1
        for reason in row.get("failure_reasons", []):
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "completed_episodes": len(outcomes), "scorable_episodes": len(valid),
        "successes": sum(valid), "failures": len(valid) - sum(valid),
        "success_rate": (sum(valid) / len(valid)) if valid and not unavailable else None,
        "score_status": "available" if valid and not unavailable else "unavailable",
        "unavailable_episodes": unavailable, "gates": gates, "failure_reasons": reasons,
        "reward_used_as_success": False,
    }


class EvaluationQuota:
    """A fixed per-environment denominator, unaffected by episode duration."""

    def __init__(self, scored_envs: int, target_episodes: int):
        if not 0 < scored_envs <= target_episodes:
            raise ValueError("Every scored environment needs at least one episode")
        self.quotas = {index: target_episodes // scored_envs + (index < target_episodes % scored_envs)
                       for index in range(scored_envs)}
        self.counts = dict.fromkeys(self.quotas, 0)
        self.rows: list[dict] = []
        self.excluded: list[list[int]] = []
        self.seen = set()

    @property
    def complete(self) -> bool:
        return self.counts == self.quotas

    def add(self, rows: list[dict]) -> None:
        for row in rows:
            identity = (row.get("env_id"), row.get("episode_id"))
            if None in identity or identity in self.seen:
                raise ValueError("Missing or duplicate evaluation episode identity")
            self.seen.add(identity)
            env = identity[0]
            if env not in self.quotas or self.counts[env] >= self.quotas[env]:
                self.excluded.append(list(identity))
            else:
                self.rows.append(row)
                self.counts[env] += 1

    def report(self) -> dict:
        summary = summarize_outcomes(self.rows)
        summary.update({"target_completed": self.complete,
                        "per_env_quota": dict(self.quotas), "per_env_completed": dict(self.counts),
                        "scored_episode_ids": [[row["env_id"], row["episode_id"]] for row in self.rows],
                        "excluded_episode_ids": self.excluded})
        if not self.complete:
            summary["score_status"] = "incomplete"
            summary["success_rate"] = None
        return summary


def _assert_finite(value: Any, torch: Any, label: str) -> None:
    if isinstance(value, dict) or hasattr(value, "items"):
        for name, item in value.items():
            _assert_finite(item, torch, f"{label}.{name}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite(item, torch, f"{label}[{index}]")
    elif torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise NumericalFailure(f"Nonfinite {label}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise NumericalFailure(f"Nonfinite {label}")


def _parameters(algorithm: Any) -> list:
    return list(algorithm.actor.parameters()) + list(algorithm.critic.parameters())


def parameter_snapshot(module: Any, torch: Any) -> Any:
    values = [value.detach().to(device="cpu", dtype=torch.float64).flatten().clone()
              for value in module.parameters()]
    if not values:
        raise NumericalFailure("Policy module has no trainable parameters")
    vector = torch.cat(values)
    _assert_finite(vector, torch, "parameters")
    return vector


def parameter_change(before: Any, after: Any, torch: Any) -> dict:
    if before.shape != after.shape:
        raise NumericalFailure("Parameter shape changed during an update")
    delta = after - before
    _assert_finite(delta, torch, "parameter delta")
    return {"l2": float(torch.linalg.vector_norm(delta)),
            "max_abs": float(delta.abs().max()), "changed_elements": int((delta != 0).sum()),
            "before_sha256": hashlib.sha256(before.numpy().tobytes()).hexdigest(),
            "after_sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest()}


def module_state_sha256(module: Any, torch: Any) -> str:
    """Hash trainable parameters and normalization buffers without pickle."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        _assert_finite(tensor, torch, f"state.{name}")
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class OptimizerAudit:
    """Observe real backward/optimizer calls and reject nonfinite gradients first."""

    def __init__(self, algorithm: Any, torch: Any, deadline: Deadline, record: Any = None):
        self.algorithm, self.torch, self.deadline = algorithm, torch, deadline
        self.optimizer_steps = 0
        self.records: list[dict] = []
        self.pending: dict[int, tuple[Any, Any]] = {}
        self.handles = []
        self.record = record
        for index, parameter in enumerate(_parameters(algorithm)):
            if parameter.requires_grad:
                def observe(gradient, key=index):
                    # These scalars are taken before RSL-RL's gradient clipping.
                    self.pending[key] = (gradient.detach().double().square().sum(),
                                         self.torch.isfinite(gradient).all())
                    return gradient
                self.handles.append(parameter.register_hook(observe))
        self.handles.append(algorithm.optimizer.register_step_pre_hook(self._before_step))
        self.handles.append(algorithm.optimizer.register_step_post_hook(self._after_step))

    def _before_step(self, optimizer: Any, args: tuple, kwargs: dict) -> None:
        self.deadline.check()
        if not self.pending:
            raise NumericalFailure("Optimizer step without observed backward gradients")
        torch = self.torch
        raw_squared = torch.stack([row[0] for row in self.pending.values()]).sum()
        raw_finite = torch.stack([row[1] for row in self.pending.values()]).all()
        gradients = [p.grad.detach().double().square().sum()
                     for p in _parameters(self.algorithm) if p.grad is not None]
        if not bool(raw_finite) or not bool(torch.isfinite(raw_squared)):
            raise NumericalFailure("Nonfinite raw gradient; optimizer step rejected")
        if not gradients:
            raise NumericalFailure("Optimizer step has no gradients")
        clipped_squared = torch.stack(gradients).sum()
        if not bool(torch.isfinite(clipped_squared)):
            raise NumericalFailure("Nonfinite clipped gradient; optimizer step rejected")
        self.current = {"optimizer_step": self.optimizer_steps + 1,
                        "raw_gradient_l2": float(raw_squared.sqrt()),
                        "clipped_gradient_l2": float(clipped_squared.sqrt()),
                        "gradient_parameter_count": len(self.pending),
                        "raw_gradients_finite": True}

    def _after_step(self, optimizer: Any, args: tuple, kwargs: dict) -> None:
        self.optimizer_steps += 1
        try:
            _assert_finite(_parameters(self.algorithm), self.torch, "post optimizer parameters")
        except BaseException:
            self.current["post_parameters_finite"] = False
            self._record(self.current)
            raise
        self.current["post_parameters_finite"] = True
        self._record(self.current)
        self.pending.clear()

    def _record(self, row: dict) -> None:
        self.records.append(row)
        if self.record is not None:
            self.record(row)

    def drain(self) -> list[dict]:
        rows, self.records = self.records, []
        return rows

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _gpu_memory(torch: Any, device: str) -> dict:
    if not str(device).startswith("cuda"):
        return {"device": device, "cuda": False}
    free, total = torch.cuda.mem_get_info(device)
    result = {"device": device, "cuda": True, "free_bytes": free, "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_scope": "torch allocator only; free/total are device-wide observations"}
    try:
        import psutil
        memory = psutil.Process().memory_info()
        result.update(host_process_rss_bytes=memory.rss, host_available_ram_bytes=psutil.virtual_memory().available)
    except ImportError:
        result["host_memory_unavailable"] = "psutil is unavailable"
    return result


def _seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    # Isaac Lab requires NumPy; keep it lazy like torch.
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment_audit(environment: Any) -> dict:
    result = _json_safe(environment.unwrapped.runtime_audit())
    contract = result.get("policy_contract")
    if not isinstance(contract, dict) or not contract:
        raise ValueError("The environment must expose an explicit policy_contract")
    result["policy_contract_sha256"] = canonical_sha256(contract)
    return result


def verify_step_timing(previous: dict, current: dict) -> dict:
    """Check the source 33 x 2 ms cadence and explicitly mark clock resets."""
    for key in ("physics_samples", "reset_count", "sim_time_s"):
        if key not in previous or key not in current:
            raise ValueError(f"Missing environment time/reset evidence: {key}")
    delta = current["physics_samples"] - previous["physics_samples"]
    time_delta = current["sim_time_s"] - previous["sim_time_s"]
    if delta != 33 or not math.isclose(time_delta, 0.066, rel_tol=0, abs_tol=1e-8):
        raise NumericalFailure(f"Actual curriculum sample cadence changed: {delta} samples/{time_delta:g}s")
    if current["reset_count"] < previous["reset_count"]:
        raise NumericalFailure("Environment reset counter moved backwards")
    result = {"physics_sample_delta": delta, "sample_time_delta_s": time_delta,
              "reset_delta": current["reset_count"] - previous["reset_count"]}
    for key in ("actual_physics_dt_s", "physics_dt_s"):
        if key in current and not math.isclose(current[key], 0.002, rel_tol=0, abs_tol=1e-12):
            raise NumericalFailure("Native physics time step changed")
    return result


def _record_step(environment: Any, extras: dict, streams: dict,
                 phase: str, step: int, update: int | None, torch: Any) -> list[dict]:
    audit = _json_safe(environment.unwrapped.step_audit())
    if audit.get("finite") is False or audit.get("numerically_safe") is False or audit.get("nonfinite_state") is True:
        raise NumericalFailure("Environment step audit reports an unsafe numerical state")
    previous = getattr(environment, "_g1_last_step_audit", None)
    timing = verify_step_timing(previous, audit) if previous is not None else None
    environment._g1_last_step_audit = audit
    streams["steps"].write({"phase": phase, "control_step": step, "update": update,
                           "environment": audit, "verified_cadence": timing})
    outcomes = environment.unwrapped.pop_episode_outcomes()
    for row in outcomes:
        streams["episodes"].write({"phase": phase, "update": update, **row})
    return outcomes


def _new_streams(run_dir: Path) -> dict[str, EvidenceStream]:
    result = {}
    try:
        for key, filename in (("steps", "steps.jsonl"), ("episodes", "episode_outcomes.jsonl"),
                              ("updates", "updates.jsonl"), ("optimizers", "optimizer_steps.jsonl"),
                              ("validations", "validations.jsonl")):
            result[key] = EvidenceStream(run_dir / filename)
    except BaseException:
        for stream in result.values():
            stream.close()
        raise
    return result


def _make_runner(environment: Any, settings: dict, run_dir: Path, torch: Any) -> tuple[Any, dict]:
    from rsl_rl.runners import OnPolicyRunner
    cfg = build_rsl_config(settings)
    _write_json_new(run_dir / "rsl_config.json", cfg)
    # Own JSONL evidence replaces the optional external logger. No upstream code
    # or global environment configuration is changed.
    runner = OnPolicyRunner(environment, copy.deepcopy(cfg), log_dir=None, device=settings["device"])
    runner.logger.init_logging_writer()
    return runner, cfg


def _save_checkpoint(runner: Any, run_dir: Path, label: str, info: dict) -> dict:
    path = run_dir / "checkpoints" / f"checkpoint_{label}.pt"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    runner.current_learning_iteration = int(info["completed_updates"])
    runner.save(str(path), infos=_json_safe(info))
    # Flush the checkpoint itself before recording its digest.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path),
            "completed_updates": info["completed_updates"], "optimizer_steps": info["optimizer_steps"]}


def _collect_evaluation(environment: Any, runner: Any, settings: dict, streams: dict,
                        deadline: Deadline, torch: Any, *, update: int, phase: str,
                        state_hold: bool = False) -> dict:
    """Evaluate deterministic actions with no optimizer or normalizer update."""
    import numpy as np
    before_actor = parameter_snapshot(runner.alg.actor, torch)
    before_critic = parameter_snapshot(runner.alg.critic, torch)
    before_actor_state = module_state_sha256(runner.alg.actor, torch)
    before_critic_state = module_state_sha256(runner.alg.critic, torch)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    steps = 0
    try:
        environment.unwrapped.set_evaluation_mode(True, seed=int(settings["eval_seed"]))
        environment.reset()
        obs = environment.get_observations().to(settings["device"])
        runner.alg.eval_mode()
        target = int(settings["evaluation"]["episodes"])
        scored_envs = min(environment.num_envs, int(settings["evaluation"]["num_envs"]))
        quota = EvaluationQuota(scored_envs, target)
        exhausted = False
        max_steps = int(settings["evaluation"]["max_control_steps"])
        with torch.inference_mode():
            while not quota.complete and steps < max_steps:
                try:
                    deadline.check()
                except BudgetExpired:
                    exhausted = True
                    break
                actions = (environment.unwrapped.state_hold_actions()
                           if state_hold else runner.alg.actor(obs))
                _assert_finite(actions, torch, "evaluation actions")
                obs, rewards, dones, extras = environment.step(actions.to(environment.device))
                _assert_finite((obs, rewards), torch, "evaluation transition")
                obs = obs.to(settings["device"])
                steps += 1
                rows = _record_step(environment, extras, streams, phase, steps, update, torch)
                quota.add(rows)
        summary = quota.report()
        summary.update({"phase": phase, "update": update, "control_steps": steps,
                        "target_episodes": target, "budget_exhausted": exhausted,
                        "optimizer_steps": 0, "deterministic_actions": True,
                        "normalizer_updates": 0, "seed": settings["eval_seed"],
                        "action_policy": "measured-state-hold requests through shared velocity/anchor/mechanical limits and probe shield"
                            if state_hold else "checkpoint deterministic actor"})
        summary.update({"simulated_num_envs": environment.num_envs, "scored_num_envs": scored_envs,
                        "actor_state_sha256_before": before_actor_state,
                        "actor_state_sha256_after": module_state_sha256(runner.alg.actor, torch),
                        "critic_state_sha256_before": before_critic_state,
                        "critic_state_sha256_after": module_state_sha256(runner.alg.critic, torch)})
        # Normalizer buffers and trainable weights must both remain unchanged.
        summary["actor_trainable_change"] = parameter_change(before_actor,
                                                             parameter_snapshot(runner.alg.actor, torch), torch)
        summary["critic_trainable_change"] = parameter_change(before_critic,
                                                              parameter_snapshot(runner.alg.critic, torch), torch)
        if summary["actor_trainable_change"]["changed_elements"] or summary["critic_trainable_change"]["changed_elements"]:
            raise NumericalFailure("Evaluation changed trainable policy parameters")
        if (summary["actor_state_sha256_after"] != before_actor_state or
                summary["critic_state_sha256_after"] != before_critic_state):
            raise NumericalFailure("Evaluation changed policy or normalizer state")
        streams["validations"].write(summary)
        return summary
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        environment.unwrapped.set_evaluation_mode(False)


def _refresh_training_progress(progress: dict, context: dict, torch: Any) -> None:
    """Recover actual optimization and weight changes even after a partial update."""
    guard = context.get("guard")
    if guard is not None:
        progress["optimizer_steps"] = guard.optimizer_steps
    progress["training_started"] = progress.get("optimizer_steps", 0) > 0
    expected_per_update = context.get("optimizer_steps_per_update", 0)
    if expected_per_update:
        progress["partial_update_optimizer_steps"] = progress.get("optimizer_steps", 0) - (
            progress.get("completed_updates", 0) * expected_per_update)
    runner = context.get("runner")
    if runner is not None:
        for name in ("actor", "critic"):
            initial = context.get("initial_" + name)
            if initial is None:
                continue
            try:
                progress[name + "_total_change"] = parameter_change(initial,
                    parameter_snapshot(getattr(runner.alg, name), torch), torch)
            except BaseException as exc:
                progress[name + "_total_change"] = None
                progress[name + "_change_unavailable"] = {"type": type(exc).__name__, "message": str(exc)}


def _train(environment: Any, settings: dict, run_dir: Path, streams: dict,
           deadline: Deadline, torch: Any, audit: dict, progress: dict | None = None) -> dict:
    """Keep authoritative live progress outside all fallible training operations."""
    progress = {} if progress is None else progress
    context: dict = {}
    try:
        return _train_impl(environment, settings, run_dir, streams, deadline, torch, audit, progress, context)
    except BaseException as exc:
        _refresh_training_progress(progress, context, torch)
        progress["failure_type"] = type(exc).__name__
        progress["failure_message"] = str(exc)
        try:
            _write_json_new(run_dir / "training_partial_failure.json", progress)
        except BaseException as persistence_error:
            # The native owner still receives the same live dictionary if even
            # checkpoint/evidence persistence fails (for example a full disk).
            progress["partial_report_persistence_error"] = {
                "type": type(persistence_error).__name__, "message": str(persistence_error)}
        raise
    finally:
        _refresh_training_progress(progress, context, torch)
        if context.get("guard") is not None:
            context["guard"].close()


def _train_impl(environment: Any, settings: dict, run_dir: Path, streams: dict,
                deadline: Deadline, torch: Any, audit: dict, progress: dict, context: dict) -> dict:
    runner, runner_cfg = _make_runner(environment, settings, run_dir, torch)
    context["runner"] = runner
    result = progress
    result.update({"completed_updates": 0, "optimizer_steps": 0, "checkpoints": [],
                   "validations": [], "training_control_steps": 0,
                   "training_environment_transitions": 0, "budget_exhausted": False,
                   "success_claim_scope": "local curriculum only; no original focus/20 acceptance"})
    audit_update = {"index": 0}
    def record_optimizer(row):
        result["optimizer_steps"] = row["optimizer_step"]
        result["training_started"] = True
        result["last_optimizer_update"] = audit_update["index"]
        streams["optimizers"].write({"update": audit_update["index"], **row})
    guard = OptimizerAudit(runner.alg, torch, deadline, record=record_optimizer)
    context["guard"] = guard
    context["optimizer_steps_per_update"] = int(settings["ppo"]["epochs"]) * int(settings["ppo"]["minibatches"])
    base_info = {"runner_version": RUNNER_VERSION,
                 "policy_contract": audit["policy_contract"],
                 "policy_contract_sha256": audit["policy_contract_sha256"],
                 "rsl_config_sha256": canonical_sha256(runner_cfg),
                 "training_config_sha256": settings["training_config_sha256"],
                 "source_tree_sha256": settings["source_tree_sha256"],
                 "seed": settings["train_seed"]}
    initial_actor = parameter_snapshot(runner.alg.actor, torch)
    initial_critic = parameter_snapshot(runner.alg.critic, torch)
    context["initial_actor"], context["initial_critic"] = initial_actor, initial_critic
    result["checkpoints"].append(_save_checkpoint(runner, run_dir, "update_0000_initial",
                                                  {**base_info, "completed_updates": 0, "optimizer_steps": 0}))
    try:
        result["state_hold_validation"] = _collect_evaluation(environment, runner, settings, streams,
            deadline, torch, update=0, phase="state_hold_baseline", state_hold=True)
        result["validations"].append(_collect_evaluation(environment, runner, settings, streams,
                                                          deadline, torch, update=0, phase="validation_initial"))
        environment.reset()
        obs = environment.get_observations().to(settings["device"])
        runner.alg.train_mode()
        for update in range(1, int(settings["ppo"]["max_updates"]) + 1):
            audit_update["index"] = update
            deadline.check()
            actor_before = parameter_snapshot(runner.alg.actor, torch)
            critic_before = parameter_snapshot(runner.alg.critic, torch)
            optimizer_before = guard.optimizer_steps
            started = time.monotonic()
            with torch.inference_mode():
                for _ in range(int(settings["ppo"]["rollout_steps"])):
                    deadline.check()
                    actions = runner.alg.act(obs)
                    _assert_finite(actions, torch, "training actions")
                    obs, rewards, dones, extras = environment.step(actions.to(environment.device))
                    _assert_finite((obs, rewards), torch, "training transition")
                    obs, rewards, dones = (item.to(settings["device"]) for item in (obs, rewards, dones))
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    result["training_control_steps"] += 1
                    result["training_environment_transitions"] += environment.num_envs
                    _record_step(environment, extras, streams, "training", result["training_control_steps"], update, torch)
                runner.alg.compute_returns(obs)
                _assert_finite((runner.alg.storage.returns, runner.alg.storage.advantages), torch, "PPO targets")
            rollout_seconds = time.monotonic() - started
            deadline.check()
            started_update = time.monotonic()
            loss = runner.alg.update()
            _assert_finite(loss, torch, "PPO loss")
            gradient_rows = guard.drain()
            actor_delta = parameter_change(actor_before, parameter_snapshot(runner.alg.actor, torch), torch)
            critic_delta = parameter_change(critic_before, parameter_snapshot(runner.alg.critic, torch), torch)
            expected = int(settings["ppo"]["epochs"]) * int(settings["ppo"]["minibatches"])
            actual = guard.optimizer_steps - optimizer_before
            if actual != expected or actor_delta["changed_elements"] == 0 or critic_delta["changed_elements"] == 0:
                raise NumericalFailure("PPO update did not perform the declared real optimizer/parameter changes")
            elapsed_update = time.monotonic() - started_update
            row = {"update": update, "optimizer_steps_this_update": actual,
                   "optimizer_steps_total": guard.optimizer_steps, "loss": loss,
                   "actor_parameter_change": actor_delta, "critic_parameter_change": critic_delta,
                   "raw_gradient_l2_min": min(item["raw_gradient_l2"] for item in gradient_rows),
                   "raw_gradient_l2_max": max(item["raw_gradient_l2"] for item in gradient_rows),
                   "rollout_seconds": rollout_seconds, "optimization_seconds": elapsed_update,
                   "rollout_transitions": environment.num_envs * int(settings["ppo"]["rollout_steps"]),
                   "transitions_per_second": (environment.num_envs * int(settings["ppo"]["rollout_steps"])) /
                                              (rollout_seconds + elapsed_update),
                   "learning_rate": float(runner.alg.learning_rate), "gpu_memory": _gpu_memory(torch, settings["device"]),
                   "wall_elapsed_seconds": deadline.elapsed}
            streams["updates"].write(row)
            print("G1_RL_UPDATE=" + json.dumps(_json_safe(row), separators=(",", ":")), flush=True)
            result["completed_updates"] = update
            result["optimizer_steps"] = guard.optimizer_steps
            if update % int(settings["ppo"]["validation_interval"]) == 0:
                result["checkpoints"].append(_save_checkpoint(runner, run_dir, f"update_{update:04d}_interval",
                    {**base_info, "completed_updates": update, "optimizer_steps": guard.optimizer_steps}))
                result["validations"].append(_collect_evaluation(environment, runner, settings, streams,
                    deadline, torch, update=update, phase="validation_interval"))
                if result["validations"][-1]["budget_exhausted"]:
                    raise BudgetExpired("Wall-clock budget expired during validation")
                environment.reset()
                obs = environment.get_observations().to(settings["device"])
                runner.alg.train_mode()
        if (result["completed_updates"] % int(settings["ppo"]["validation_interval"]) and
                not result["budget_exhausted"]):
            result["validations"].append(_collect_evaluation(environment, runner, settings, streams,
                deadline, torch, update=result["completed_updates"], phase="validation_final"))
            if result["validations"][-1]["budget_exhausted"]:
                raise BudgetExpired("Wall-clock budget expired during final validation")
    except BudgetExpired:
        result["budget_exhausted"] = True
        result["stop_reason"] = "predeclared_wall_clock_budget"
    except BaseException:
        result["optimizer_steps"] = guard.optimizer_steps
        result["failure_during_update"] = result["completed_updates"] + 1
        raise
    finally:
        guard.drain()  # Each actual optimizer call has already reached the JSONL sink.
        guard.close()
    result["optimizer_steps"] = guard.optimizer_steps
    result["checkpoints"].append(_save_checkpoint(runner, run_dir,
        f"update_{result['completed_updates']:04d}_final",
        {**base_info, "completed_updates": result["completed_updates"],
         "optimizer_steps": guard.optimizer_steps, "budget_exhausted": result["budget_exhausted"],
         "partial_rollout_discarded": result["training_control_steps"] >
             result["completed_updates"] * int(settings["ppo"]["rollout_steps"]),
         "partial_update_optimizer_steps": guard.optimizer_steps - result["completed_updates"] *
             int(settings["ppo"]["epochs"]) * int(settings["ppo"]["minibatches"])}))
    result["actor_total_change"] = parameter_change(initial_actor, parameter_snapshot(runner.alg.actor, torch), torch)
    result["critic_total_change"] = parameter_change(initial_critic, parameter_snapshot(runner.alg.critic, torch), torch)
    if result["completed_updates"] == 0:
        raise BudgetExpired("No complete PPO update fit within the wall-clock budget")
    return result


def _smoke(environment: Any, settings: dict, streams: dict, deadline: Deadline,
           torch: Any, profile: int) -> dict:
    outcomes = []
    obs = environment.get_observations()
    _assert_finite(obs, torch, "smoke initial observation")
    steps = int(settings["smoke"]["control_steps"])
    initial = _json_safe(environment.unwrapped.step_audit())
    start = time.monotonic()
    with torch.inference_mode():
        for step in range(steps):
            deadline.check()
            if step < steps // 2:
                actions = torch.zeros((environment.num_envs, environment.num_actions), device=environment.device)
            else:
                actions = 2 * torch.rand((environment.num_envs, environment.num_actions), device=environment.device) - 1
            obs, rewards, dones, extras = environment.step(actions)
            _assert_finite((obs, rewards), torch, "smoke transition")
            outcomes.extend(_record_step(environment, extras, streams, f"smoke_{profile}", step + 1, None, torch))
    elapsed = time.monotonic() - start
    final = _json_safe(environment.unwrapped.step_audit())
    reset_envs = sorted({row["env_id"] for row in outcomes})
    if reset_envs != list(range(profile)) or final["reset_count"] <= initial["reset_count"]:
        raise NumericalFailure("Smoke did not observe a real terminal reset for every parallel environment")
    return {"num_envs": profile, "control_steps": steps, "environment_transitions": steps * profile,
            "wall_seconds": elapsed, "transitions_per_second": steps * profile / max(elapsed, 1e-9),
            "optimizer_steps": 0, "actions": "first half zero target-hold, second half seeded bounded uniform",
            "initial_step_audit": initial, "final_step_audit": final,
            "terminal_reset_observed_for_env_ids": reset_envs,
            "outcomes": summarize_outcomes(outcomes), "gpu_memory": _gpu_memory(torch, settings["device"]),
            "success_claim_scope": "smoke execution and numerical checks only"}


def _evaluation(environment: Any, settings: dict, run_dir: Path, streams: dict,
                deadline: Deadline, torch: Any, audit: dict) -> dict:
    selection = settings["evaluation"]["checkpoint_path"]
    if not isinstance(selection, str) or not selection:
        raise ValueError("Evaluation requires an explicit checkpoint path")
    checkpoint = Path(selection)
    expected = settings["evaluation"]["checkpoint_sha256"]
    if not checkpoint.is_file() or not expected or file_sha256(checkpoint) != expected:
        raise ValueError("Evaluation requires an existing explicitly hashed checkpoint")
    runner, _ = _make_runner(environment, settings, run_dir, torch)
    saved = torch.load(str(checkpoint), map_location=settings["device"], weights_only=False)
    info = saved.get("infos", {})
    if info.get("policy_contract_sha256") != audit["policy_contract_sha256"]:
        raise ValueError("Checkpoint policy contract does not match this environment")
    for key in ("training_config_sha256", "source_tree_sha256"):
        if info.get(key) != settings[key]:
            raise ValueError(f"Checkpoint {key} does not match the frozen evaluation identity")
    runner.alg.load(saved, {"actor": True, "critic": True, "optimizer": False, "iteration": False}, strict=True)
    summary = _collect_evaluation(environment, runner, settings, streams, deadline, torch,
                                  update=int(info.get("completed_updates", 0)), phase="evaluation")
    summary.update({"checkpoint": str(checkpoint), "checkpoint_sha256": expected,
                    "checkpoint_training_optimizer_steps": info.get("optimizer_steps"),
                    "optimizer_steps": 0, "success_claim_scope": "local curriculum only; no original focus/20 acceptance"})
    return summary


def run_native_stage(stage: str, config: dict, run_dir: str | Path) -> dict:
    """Run one frozen native stage and persist its report before Kit shutdown.

    Isaac Sim's normal fast shutdown may terminate Python inside ``app.close``.
    The caller must therefore consume ``native_report.json`` as the result and
    independently record the child process exit code.
    """
    if stage not in NATIVE_STAGES:
        raise ValueError(f"Unsupported native RL stage: {stage}")
    if config["rl_teacher"].get("enabled") is not True:
        raise ValueError("RL teacher execution requires rl_teacher.enabled=true")
    settings = normalize_settings(config["rl_teacher"])
    _validate_settings(settings)
    settings["training_config_sha256"] = policy_training_identity(config)
    settings["source_tree_sha256"] = source_tree_identity()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "native_report.json").exists():
        raise FileExistsError("This native run already has a report")
    streams = _new_streams(run_dir)
    deadline = Deadline(float(settings["ppo"]["wall_time_seconds"]))
    result: dict = {"runner_version": RUNNER_VERSION, "stage": stage, "passed": False,
                   "started_utc": datetime.now(timezone.utc).isoformat(), "optimizer_steps": 0,
                   "training_requested": stage == "rl-train", "training_started": False,
                   "reward_used_as_success": False,
                   "original_focus_or_20_acceptance": False, "run_dir": str(run_dir)}
    app = None
    environment = None
    torch = None
    error = None
    training_progress = None
    try:
        # This is process-local and precedes GitPython's import from rsl_rl.
        git_path = Path(settings["runtime"]["git_executable"])
        if not git_path.is_file():
            raise FileNotFoundError(f"Native Git executable not found: {git_path}")
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = str(git_path)
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, device=settings["device"], enable_cameras=stage == "rl-eval")
        app = launcher.app
        deadline.check()
        import torch as torch_module
        torch = torch_module
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from .rl_environment import build_environment
        _seed(int(settings["smoke_seed"] if stage == "rl-smoke" else settings["train_seed"]), torch)
        if str(settings["device"]).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(settings["device"])
        result["runtime_versions"] = {name: importlib.metadata.version(name)
                                      for name in ("torch", "isaaclab", "isaaclab_rl", "rsl-rl-lib")}
        if result["runtime_versions"]["rsl-rl-lib"] != "5.0.1":
            raise ValueError("This runner is pinned to the audited RSL-RL 5.0.1 API")
        result["gpu_memory_before"] = _gpu_memory(torch, settings["device"])
        asset_path = Path(settings["runtime"]["asset_path"])
        if not asset_path.is_file():
            raise FileNotFoundError(f"Frozen robot asset not found: {asset_path}")
        result["input_asset_sha256"] = file_sha256(asset_path)
        profiles = list(settings["smoke"]["num_envs"]) if stage == "rl-smoke" else [
            1 if stage == "rl-audit" else int(settings["evaluation"]["num_envs"])
            if stage == "rl-eval" else int(settings["num_envs"])]
        result["profiles"] = []
        for count in profiles:
            deadline.check()
            env_out = run_dir / f"environment_{count}"
            env_out.mkdir(exist_ok=False)
            native_env = build_environment(config, num_envs=count, device=settings["device"],
                asset_path=str(asset_path), train_seed=int(settings["train_seed"]),
                eval_mode=stage == "rl-eval", output_dir=str(env_out))
            environment = RslRlVecEnvWrapper(native_env, clip_actions=1.0)
            environment._g1_last_step_audit = _json_safe(environment.unwrapped.step_audit())
            audit = _environment_audit(environment)
            _write_json_new(env_out / "environment_audit.json", audit)
            _assert_finite(environment.get_observations(), torch, "initial observation")
            if stage == "rl-audit":
                profile = {"num_envs": count, "optimizer_steps": 0,
                           "environment_audit": audit, "success_claim_scope": "none; audit only"}
            elif stage == "rl-smoke":
                profile = _smoke(environment, settings, streams, deadline, torch, count)
            elif stage == "rl-train":
                training_progress = {}
                profile = _train(environment, settings, run_dir, streams, deadline, torch, audit, training_progress)
                result["optimizer_steps"] = profile["optimizer_steps"]
                result["training_started"] = profile["optimizer_steps"] > 0
            else:
                profile = _evaluation(environment, settings, run_dir, streams, deadline, torch, audit)
            result["profiles"].append(profile)
            environment.close()
            environment = None
        result["passed"] = True
        result["status"] = "completed"
    except BaseException as exc:
        error = exc
        result["status"] = "failed"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc(),
                             "out_of_memory": "out of memory" in str(exc).lower()}
        partial = run_dir / "training_partial_failure.json"
        if training_progress is not None:
            result["partial_training"] = _json_safe(training_progress)
            result["optimizer_steps"] = training_progress.get("optimizer_steps", 0)
            result["training_started"] = result["optimizer_steps"] > 0
        elif partial.is_file():
            result["partial_training"] = json.loads(partial.read_text(encoding="utf-8"))
            result["optimizer_steps"] = result["partial_training"]["optimizer_steps"]
            result["training_started"] = result["optimizer_steps"] > 0
    finally:
        if environment is not None:
            try:
                environment.close()
            except BaseException as exc:
                result["environment_close_error"] = {"type": type(exc).__name__, "message": str(exc)}
                result["passed"] = False
                result["status"] = "failed"
        if torch is not None:
            try:
                result["gpu_memory_final"] = _gpu_memory(torch, settings["device"])
            except BaseException as exc:
                result["gpu_memory_unavailable"] = str(exc)
        for stream in streams.values():
            stream.close()
        result["evidence"] = [{"path": str(stream.path), "records": stream.count,
                               "bytes": stream.path.stat().st_size, "sha256": file_sha256(stream.path)}
                              for stream in streams.values()]
        result["wall_elapsed_seconds"] = deadline.elapsed
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["application_close_requested"] = app is not None
        result["application_close_confirmation"] = "parent must verify native process exit"
        _write_json_new(run_dir / "native_report.json", result)
        print("G1_RL_NATIVE_REPORT=" + json.dumps({"path": str(run_dir / "native_report.json"),
                                                   "passed": result["passed"], "stage": stage}), flush=True)
        if app is not None:
            app.close(exit_code=0 if result["passed"] else 2)
    if error is not None:
        raise error
    return result
