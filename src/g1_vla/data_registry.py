"""Provenance validation and grouped dataset splits without loading training data.

A positive strict-grasp example requires contact, a 100 mm lift, at least
0.5 seconds of holding, and explicitly disabled assisted grasping.  An in-box
flag alone is not physical success.  Language rewrites and repeated takeovers
are joined through their original rollout and layout before assigning splits.
"""

import hashlib
import json
import math
import posixpath
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


REQUIRED_FIELDS = ("parent_root", "source_rollout", "layout_fingerprint", "takeover_frame", "regime", "physical_gate")
VALID_SPLITS = frozenset(("train", "val", "test", "excluded", "historical_regression", "diagnostic"))


class RegistryValidationError(ValueError):
    def __init__(self, report: Dict[str, Any]):
        self.report = report
        super().__init__("Invalid data registry: " + "; ".join(error["message"] for error in report["errors"][:5]))


def _normalized(value: str) -> str:
    return posixpath.normpath(value.strip().replace("\\", "/"))


def source_group_key(record: Dict[str, Any]) -> str:
    """Identify an original source even when its metadata lives in other shards."""
    root = _normalized(str(record.get("parent_root", "")))
    source = _normalized(str(record.get("source_rollout", "")))
    return source if posixpath.isabs(source) else posixpath.normpath(posixpath.join(root, source))


def group_key(record: Dict[str, Any]) -> str:
    """Stable initial parent-rollout/layout key; split assignment also joins shared sources/layouts."""
    value = [source_group_key(record), str(record.get("layout_fingerprint", "")).strip()]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_positive(record: Dict[str, Any]) -> bool:
    return record.get("is_positive") is True or record.get("label") in ("positive", "success")


def strict_physical_success(record: Dict[str, Any]) -> bool:
    gate = record.get("physical_gate")
    if not isinstance(gate, dict) or record.get("expert_success") is False or gate.get("expert_success") is False:
        return False

    def adequate(name: str, threshold: float) -> bool:
        value = gate.get(name)
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= threshold

    success = (gate.get("contact") is True and adequate("lift_height_m", 0.1)
               and adequate("hold_seconds", 0.5) and gate.get("assisted") is False)
    if record.get("task") in ("pick_place", "place_in_box"):
        success = success and gate.get("in_box") is True and gate.get("released") is True
    return bool(success)


def detect_split_leakage(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check global original-source and layout identities, not just local shard groups."""
    identities = {}
    for index, record in enumerate(records):
        split = record.get("split")
        if split not in ("train", "val", "test"):
            continue
        keys = (("source", source_group_key(record)),
                ("layout", str(record.get("layout_fingerprint", "")).strip()))
        for kind, identity in keys:
            if not identity or identity == ".":
                continue
            entry = identities.setdefault((kind, identity), {"splits": set(), "indices": []})
            entry["splits"].add(split)
            entry["indices"].append(index)
    return [{"code": "split_leakage", "identity_kind": kind, "identity": identity,
             "splits": sorted(value["splits"]), "indices": value["indices"],
             "message": "The same original {} appears in multiple splits: {}.".format(kind, ", ".join(sorted(value["splits"])))}
            for (kind, identity), value in sorted(identities.items()) if len(value["splits"]) > 1]


def validate_registry(records: Iterable[Dict[str, Any]], check_splits: bool = True) -> Dict[str, Any]:
    rows = list(records)
    errors = []
    warnings = []
    valid_rows = []
    valid_indices = []
    for index, record in enumerate(rows):
        def error(code: str, message: str) -> None:
            errors.append({"index": index, "code": code, "message": message})

        if not isinstance(record, dict):
            error("invalid_record", "Registry row {} is not an object.".format(index))
            continue
        historical = record.get("split") in ("historical_regression", "diagnostic") and record.get("training_eligible") is False
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            error("missing_fields", "Registry row {} is missing {}.".format(index, ", ".join(missing)))
        for field in ("parent_root", "source_rollout", "layout_fingerprint", "regime"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                message = "Registry row {} requires a nonempty {}.".format(index, field)
                if historical:
                    warnings.append({"index": index, "code": "historical_missing_provenance", "message": message})
                else:
                    error("missing_provenance", message)
        frame = record.get("takeover_frame")
        if frame is not None and (not isinstance(frame, int) or isinstance(frame, bool) or frame < 0):
            error("invalid_takeover_frame", "Registry row {} takeover_frame must be a non-negative integer or null.".format(index))
        if record.get("regime") in ("direct_takeover", "waypoint_recovery", "direct_continuation", "safe_waypoint") and frame is None:
            message = "Registry row {} takeover regime requires the visited frame.".format(index)
            if historical:
                warnings.append({"index": index, "code": "historical_missing_takeover_frame", "message": message})
            else:
                error("missing_takeover_frame", message)
        if not isinstance(record.get("physical_gate"), dict):
            error("invalid_physical_gate", "Registry row {} physical_gate must contain measured evidence.".format(index))
        if _is_positive(record) and not strict_physical_success(record):
            error("invalid_positive", "Registry row {} positive label lacks strict physical success or reports expert failure.".format(index))
        if "is_positive" in record and not isinstance(record["is_positive"], bool):
            error("invalid_label", "Registry row {} is_positive must be a boolean.".format(index))
        split = record.get("split")
        if check_splits and split is not None and (not isinstance(split, str) or split not in VALID_SPLITS):
            error("invalid_split", "Registry row {} has an unsupported split.".format(index))
        if check_splits and record.get("training_eligible") is False and split in ("train", "val", "test"):
            error("excluded_training_source", "Registry row {} is excluded from the new model dataset.".format(index))
        if "split" not in record:
            warnings.append({"index": index, "code": "unassigned_split", "message": "Split has not been assigned."})
        if record.get("regime") in ("waypoint_recovery", "safe_waypoint"):
            warnings.append({"index": index, "code": "waypoint_recovery", "message": "Preserve waypoint recovery as a separate regime; do not relabel as direct takeover."})
        valid_rows.append(record)
        valid_indices.append(index)
    if check_splits:
        for problem in detect_split_leakage(valid_rows):
            problem["indices"] = [valid_indices[index] for index in problem["indices"]]
            errors.append(problem)
    split_counts = Counter(str(row.get("split", "unassigned")) for row in valid_rows)
    return {"schema_version": 1, "ok": not errors, "record_count": len(rows),
            "group_count": len(set(group_key(row) for row in valid_rows if all(
                isinstance(row.get(field), str) and row[field].strip() for field in ("parent_root", "source_rollout", "layout_fingerprint")))),
            "split_counts": dict(sorted(split_counts.items(), key=lambda pair: str(pair[0]))),
            "errors": errors, "warnings": warnings}


def assign_group_splits(records: Iterable[Dict[str, Any]], train_fraction: float = 0.8,
                        val_fraction: float = 0.1, seed: Union[str, int] = 0) -> List[Dict[str, Any]]:
    """Return copied rows; keep shared parent rollouts/layouts in one split.

    Connected components join repeated takeovers, language augmentations, and
    copies across input shards.  Fractions apply to components, not language-row
    counts.  Excluded artifacts receive split='excluded'.
    """
    if (not math.isfinite(train_fraction) or not math.isfinite(val_fraction)
            or train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction > 1):
        raise ValueError("Split fractions must be finite, non-negative, and sum to at most one.")
    rows = [dict(row) if isinstance(row, dict) else row for row in records]
    report = validate_registry(rows, check_splits=False)
    if not report["ok"]:
        raise RegistryValidationError(report)
    eligible = [index for index, row in enumerate(rows) if row.get("training_eligible") is not False]
    parents = {index: index for index in eligible}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    seen = {}
    for index in eligible:
        row = rows[index]
        for identity in (("source", source_group_key(row)), ("layout", row["layout_fingerprint"].strip())):
            if identity in seen:
                union(index, seen[identity])
            else:
                seen[identity] = index
    components = {}
    for index in eligible:
        components.setdefault(find(index), []).append(index)
    keyed = []
    for indices in components.values():
        identities = sorted(set(group_key(rows[index]) for index in indices))
        component_key = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode("utf-8")).hexdigest()
        order = hashlib.sha256((str(seed) + ":" + component_key).encode("utf-8")).hexdigest()
        keyed.append((order, component_key, indices))
    keyed.sort()
    fractions = (train_fraction, val_fraction, 1.0 - train_fraction - val_fraction)
    allocations = [len(keyed) * fraction for fraction in fractions]
    counts = [int(math.floor(value)) for value in allocations]
    remainder_order = sorted(range(3), key=lambda index: (-(allocations[index] - counts[index]), index))
    for index in remainder_order[:len(keyed) - sum(counts)]:
        counts[index] += 1
    names = [name for name, count in zip(("train", "val", "test"), counts) for _ in range(count)]
    for (_, component_key, indices), name in zip(keyed, names):
        for index in indices:
            rows[index]["split"] = name
            rows[index]["group_id"] = component_key
    for index, row in enumerate(rows):
        if row.get("training_eligible") is False:
            if row.get("split") in ("historical_regression", "diagnostic"):
                # Unknown historical lineage stays diagnostic, rather than
                # acquiring a fabricated group or being coerced into model data.
                row.setdefault("group_id", None)
                continue
            row["split"] = "excluded"
            row["group_id"] = group_key(row)
    final = validate_registry(rows)
    if not final["ok"]:
        raise RegistryValidationError(final)
    return rows


def read_registry(path: Union[str, Path]) -> List[Dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        if source.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        payload = json.load(stream)
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Registry must be a list, a records object, or JSONL rows.")
    return rows


def write_registry(records: Iterable[Dict[str, Any]], path: Union[str, Path]) -> Path:
    """Create an immutable registration snapshot; never overwrite an earlier one."""
    rows = list(records)
    report = validate_registry(rows)
    if not report["ok"]:
        raise RegistryValidationError(report)
    destination = Path(path)
    with destination.open("x", encoding="utf-8") as stream:
        if destination.suffix.lower() == ".jsonl":
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump({"schema_version": 1, "records": rows, "validation": report}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return destination
