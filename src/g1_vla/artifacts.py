"""Read-only artifact inventory for the G1 rebuild.

An inventory is evidence, never a deletion plan.  The module does not move,
unlink, execute, or deserialize model/data artifacts.  In particular, a summary
claiming that a cache is reproducible cannot make it eligible for deletion.
"""

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


ARCHIVE_EXPERIMENTS = frozenset(("625", "665", "686b"))
EXCLUDED_EXPERIMENTS = frozenset(("624", "631", "632", "633", "634", "647", "648"))
CACHE_CANDIDATE_EXPERIMENTS = frozenset(("619", "624", "632", "636", "642", "648", "650"))
SOURCE_ROLLOUT_EXPERIMENTS = frozenset(("639", "662", "673", "679", "682"))
BASELINE_EXPERIMENTS = frozenset(("672", "687", "689", "688", "690", "691"))
DEFAULT_SCAN_PATHS = ("outputs", "resources", "runtime", "notes")
DEFAULT_EXCLUDE_PATHS = ("outputs/rebuild", ".g1-rebuild")
DEFAULT_LARGE_FILE_THRESHOLD = 8 * 1024 * 1024
MAX_SUMMARY_BYTES = 2 * 1024 * 1024
_EXPERIMENT_PATTERN = re.compile(r"(?:g1lang)[_-]?(\d+[a-z]?)(?=[^a-z0-9]|$)", re.I)
_SOURCE_KEYS = frozenset(("parent_root", "parent_rollout", "parent_rollout_path",
                          "source_rollout", "source_rollout_path", "source_npz",
                          "raw_npz", "parent_npz", "input_npz", "source_paths",
                          "parent_sources", "raw_sources", "source_cache", "base_cache",
                          "cache_dirs", "pose_cache", "recovery_cache", "rollout_root",
                          "source_summary", "episode_dir", "episode_dirs", "image_path_train",
                          "image_path_validation"))
_RECIPE_KEYS = frozenset(("rebuild_recipe", "reconstruction_recipe", "recipe_path"))
_CHECKPOINT_EXTENSIONS = frozenset((".pt", ".pth", ".ckpt", ".safetensors"))


def sha256_file(path: Union[str, Path]) -> str:
    """Hash in bounded memory; no pickle/torch loading occurs."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_ids(path: Union[str, Path]) -> List[str]:
    """Recognise only explicit G1LANG names, avoiding unrelated numeric paths."""
    return sorted(set(match.lower() for match in _EXPERIMENT_PATTERN.findall(str(path))))


def _is_summary(path: Path) -> bool:
    name = path.name.lower()
    return "summary" in name and path.suffix.lower() in (".json", ".jsonl", ".md", ".txt")


def _is_recovery_spec(path: Path) -> bool:
    name = path.name.lower()
    return ("recovery" in name or "takeover" in name) and (
        "spec" in name or "manifest" in name
    )


def _is_checkpoint(path: Path) -> bool:
    if path.suffix.lower() not in _CHECKPOINT_EXTENSIONS:
        return False
    words = str(path).lower()
    return any(word in words for word in ("checkpoint", "weights", "model.pt", "model.pth", "policy.pt"))


def classify_artifact(path: Union[str, Path], size_bytes: int = 0,
                      large_file_threshold: int = DEFAULT_LARGE_FILE_THRESHOLD) -> Dict[str, Any]:
    """Classify preservation and training policy separately.

    Raw failures stay protected even when their experiment is excluded from
    training.  Only the specified historical .pt/.npy products become cleanup
    candidates; being a candidate never authorises a mutation.
    """
    item = Path(path)
    ids = experiment_ids(item)
    experiment_set = set(ids)
    excluded = bool(experiment_set & EXCLUDED_EXPERIMENTS)
    historical = bool(experiment_set & ARCHIVE_EXPERIMENTS)
    raw = item.suffix.lower() in (".npz", ".xml")
    summary = _is_summary(item)
    recovery = _is_recovery_spec(item)
    journal = item.name.lower() == "lerobot.md" or (
        "notes" in [part.lower() for part in item.parts] and item.suffix.lower() in (".md", ".txt")
    )
    source = bool(experiment_set & SOURCE_ROLLOUT_EXPERIMENTS)
    baseline = bool(experiment_set & BASELINE_EXPERIMENTS)
    small_checkpoint = _is_checkpoint(item) and size_bytes <= large_file_threshold
    protected = raw or summary or recovery or journal or source or baseline or small_checkpoint
    reasons = []
    if raw:
        reasons.append("Preserve raw NPZ/XML evidence, including failed rollouts.")
    if summary:
        reasons.append("Preserve experiment summary and failure evidence.")
    if recovery:
        reasons.append("Preserve recovery/takeover specification and visited-state provenance.")
    if journal:
        reasons.append("Preserve the shared experiment journal and development notes.")
    if source:
        reasons.append("Preserve parent policy rollout for takeover provenance.")
    if baseline:
        reasons.append("Preserve research baseline or its matched evaluation.")
    if small_checkpoint:
        reasons.append("Preserve small complete checkpoint.")
    candidate = (bool(experiment_set & CACHE_CANDIDATE_EXPERIMENTS)
                 and item.suffix.lower() in (".pt", ".npy") and not protected)
    if candidate:
        action = "cleanup_candidate"
        reasons.append("Historical derived cache; source chain and reconstruction still require verification.")
    elif protected:
        action = "keep"
    elif historical:
        action = "archive"
        reasons.append("Freeze historical cache for reproduction; no movement is performed.")
    elif excluded:
        action = "exclude_training"
    else:
        action = "review"
        reasons.append("Unclassified provenance; preserve pending review.")
    if excluded:
        reasons.append("Historical contamination/grouping issue: isolate from new training.")
    if historical:
        reasons.append("Historical frozen experiment; do not use directly for the new training set.")
    return {
        "experiment_ids": ids,
        "action": action,
        "protected": protected,
        "training_eligible": False if excluded or historical else None,
        "cleanup_eligible": False,
        "cleanup_blockers": (["Parent source chain has not been independently verified.",
                              "Reconstruction recipe and rebuilt content have not been independently verified."]
                             if candidate else []),
        "reasons": reasons,
    }


def _extract_provenance(value: Any) -> Dict[str, Any]:
    # Deliberately extract a small allowlist, never echo arbitrary configuration,
    # environment, authentication fields, or full summary text.
    parents = set()
    recipes = set()

    def visit(node: Any, depth: int = 0, context: str = "") -> None:
        if depth > 16:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                normalized = str(key).lower()
                source_key = normalized in _SOURCE_KEYS or (
                    normalized == "source" and context in ("initial_state_recovery", "recovery", "start_pair_audit")
                )
                destination = parents if source_key else recipes if normalized in _RECIPE_KEYS else None
                if destination is not None:
                    values = child if isinstance(child, list) else [child]
                    for item in values:
                        if isinstance(item, str) and item.strip():
                            destination.add(item.strip())
                if isinstance(child, (dict, list)):
                    visit(child, depth + 1, normalized)
        elif isinstance(node, list):
            for child in node:
                visit(child, depth + 1, context)

    visit(value)
    return {"parent_sources": sorted(parents), "recipe_paths": sorted(recipes)}


def _read_provenance(path: Path) -> Dict[str, Any]:
    if path.stat().st_size > MAX_SUMMARY_BYTES:
        raise ValueError("Summary exceeds the bounded metadata read limit.")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    elif path.suffix.lower() == ".jsonl":
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = {}
        for line in text.splitlines():
            match = re.match(r"\s*[-*]?\s*([a-zA-Z_]+)\s*[:=]\s*(.+?)\s*$", line)
            if match and match.group(1).lower() in _SOURCE_KEYS | _RECIPE_KEYS:
                payload[match.group(1).lower()] = match.group(2).strip("`\"'")
    return _extract_provenance(payload)


def inventory_artifacts(root: Union[str, Path], scan_paths: Optional[Iterable[Union[str, Path]]] = None,
                        hash_large: bool = False,
                        large_file_threshold: int = DEFAULT_LARGE_FILE_THRESHOLD,
                        exclude_paths: Optional[Iterable[Union[str, Path]]] = None) -> Dict[str, Any]:
    """Inventory explicitly selected roots without modifying scanned files.

    Relative paths resolve under root.  External journals must be explicitly
    supplied via scan_paths.  Symlinks are reported but never traversed.  Every
    readable protected raw file is streamed through SHA-256, including large
    NPZs; other large files need hash_large=True.  Errors are collected so one
    bad summary or unreadable artifact does not abort the report.
    """
    if large_file_threshold < 0:
        raise ValueError("large_file_threshold must be non-negative")
    # Normalize '..' lexically without resolving or traversing symlink targets.
    base = Path(os.path.abspath(str(Path(root).expanduser())))
    choices = list(DEFAULT_SCAN_PATHS if scan_paths is None else scan_paths)
    selected = []
    for choice in choices:
        path = Path(choice).expanduser()
        selected.append(Path(os.path.abspath(str(path if path.is_absolute() else base / path))))
    exclusions = []
    for choice in DEFAULT_EXCLUDE_PATHS if exclude_paths is None else exclude_paths:
        path = Path(choice).expanduser()
        exclusions.append(Path(os.path.abspath(str(path if path.is_absolute() else base / path))))

    def excluded(path: Path) -> bool:
        for exclusion in exclusions:
            try:
                path.relative_to(exclusion)
                return True
            except ValueError:
                pass
        return False

    errors = []
    paths = {}

    def add_error(path: Path, operation: str, error: Exception) -> None:
        errors.append({"path": str(path), "operation": operation, "error_type": type(error).__name__,
                       "message": str(error)})

    for scan in selected:
        if excluded(scan):
            continue
        if not os.path.lexists(str(scan)):
            errors.append({"path": str(scan), "operation": "scan", "error_type": "MissingPath",
                           "message": "Explicit scan path does not exist."})
            continue
        if scan.is_symlink() or scan.is_file():
            paths[str(scan.absolute())] = scan
            continue
        if not scan.is_dir():
            errors.append({"path": str(scan), "operation": "scan", "error_type": "UnsupportedPath",
                           "message": "Path is not a regular file or directory."})
            continue

        def walk_error(error: OSError) -> None:
            add_error(Path(error.filename or scan), "scan", error)

        for directory, dirnames, filenames in os.walk(str(scan), followlinks=False, onerror=walk_error):
            directory_path = Path(directory)
            for name in list(dirnames):
                item = directory_path / name
                if excluded(item):
                    dirnames.remove(name)
                    continue
                if item.is_symlink():
                    paths[str(item.absolute())] = item
                    dirnames.remove(name)
            for name in filenames:
                item = directory_path / name
                if excluded(item):
                    continue
                paths[str(item.absolute())] = item

    summaries = {}
    for item in paths.values():
        if _is_summary(item) and not item.is_symlink():
            try:
                summaries[str(item)] = _read_provenance(item)
            except (OSError, UnicodeError, ValueError, RecursionError) as error:
                add_error(item, "read_summary", error)

    summaries_by_directory = {}
    summaries_by_experiment = {}
    for summary_path in summaries:
        summaries_by_directory.setdefault(str(Path(summary_path).parent), set()).add(summary_path)
        for experiment in experiment_ids(summary_path):
            summaries_by_experiment.setdefault(experiment, set()).add(summary_path)

    records = []
    for item in sorted(paths.values(), key=lambda value: str(value)):
        try:
            relative = str(item.relative_to(base))
        except ValueError:
            relative = None
        row = {"path": str(item), "relative_path": relative, "external": relative is None,
               "sha256": None, "hash_status": "not_attempted", "parent_sources": [],
               "recipe_paths": [], "provenance_summaries": []}
        try:
            stat = item.lstat()
            row.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "kind": "symlink" if item.is_symlink() else "file"})
            row.update(classify_artifact(item, stat.st_size, large_file_threshold))
            if row["external"]:
                row["protected"] = True
                row["action"] = "keep"
                row["reasons"].append("Preserve explicitly selected external evidence.")
            if item.is_symlink():
                row["hash_status"] = "skipped_symlink"
                row["protected"] = True
                row["action"] = "keep"
                row["reasons"].append("Preserve symlink without following its target.")
            else:
                raw = item.suffix.lower() in (".npz", ".xml")
                should_hash = stat.st_size <= large_file_threshold or hash_large or raw
                if should_hash:
                    try:
                        row["sha256"] = sha256_file(item)
                        after = item.stat()
                        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                            row["sha256"] = None
                            row["hash_status"] = "changed_during_hash"
                            row["reasons"].append("Artifact changed during hashing; repeat when the writer is idle.")
                        else:
                            row["hash_status"] = "complete"
                    except OSError as error:
                        row["hash_status"] = "error"
                        add_error(item, "hash", error)
                else:
                    row["hash_status"] = "skipped_large"
                # Nearby summaries may contain the origin of a derived cache.
                related_summaries = set(summaries_by_directory.get(str(item.parent), ()))
                for experiment in row["experiment_ids"]:
                    related_summaries.update(summaries_by_experiment.get(experiment, ()))
                for summary_path in related_summaries:
                    evidence = summaries[summary_path]
                    row["parent_sources"].extend(evidence["parent_sources"])
                    row["recipe_paths"].extend(evidence["recipe_paths"])
                    row["provenance_summaries"].append(summary_path)
                for key in ("parent_sources", "recipe_paths", "provenance_summaries"):
                    row[key] = sorted(set(row[key]))
        except OSError as error:
            row.update({"size_bytes": None, "kind": "unreadable", "action": "keep", "protected": True,
                        "cleanup_eligible": False, "training_eligible": None,
                        "experiment_ids": experiment_ids(item), "reasons": ["Unreadable artifact; preserve pending review."],
                        "cleanup_blockers": [], "hash_status": "error"})
            add_error(item, "stat", error)
        records.append(row)

    counts = Counter(row["action"] for row in records)
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "root": str(base), "scan_paths": [str(path) for path in selected],
            "exclude_paths": [str(path) for path in exclusions],
            "hash_large": hash_large, "large_file_threshold": large_file_threshold,
            "read_only": True, "records": records, "errors": errors,
            "summary": {"artifact_count": len(records), "total_size_bytes": sum(row["size_bytes"] or 0 for row in records),
                        "actions": dict(sorted(counts.items())), "error_count": len(errors),
                        "cleanup_eligible_count": 0},
            "cleanup_policy": "Candidates require independently verified complete sources and reconstruction. This inventory grants no deletion eligibility."}


def write_manifest(manifest: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Write a new JSON or JSONL manifest, refusing any existing destination."""
    destination = Path(path)
    with destination.open("x", encoding="utf-8") as stream:
        if destination.suffix.lower() == ".jsonl":
            metadata = {key: value for key, value in manifest.items() if key != "records"}
            stream.write(json.dumps({"record_type": "inventory_metadata", **metadata}, ensure_ascii=False) + "\n")
            for row in manifest.get("records", []):
                stream.write(json.dumps({"record_type": "artifact", **row}, ensure_ascii=False) + "\n")
        else:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return destination
