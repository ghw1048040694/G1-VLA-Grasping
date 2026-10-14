"""Bounded, revision-pinned public source downloads; no training imports."""
from pathlib import Path
import hashlib
import json
import time

from .runtime.experiment import write_new_json


class SourceError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_public_files(repo_id, revision, repo_type, filenames, destination, max_bytes):
    """Anonymous exact-file allowlist. Verify Hub LFS SHA after bounded download."""
    from huggingface_hub import HfApi, hf_hub_download
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise SourceError("Public source revision must be pinned")
    api = HfApi(token=False)
    if not filenames or len(filenames) != len(set(filenames)):
        raise SourceError("Source allowlist must be nonempty and contain no duplicates")
    retries = []
    def retry_network(operation, scope):
        for attempt in range(1, 4):
            try:
                return operation()
            except OSError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                retries.append({"scope": scope, "attempt": attempt, "error": type(exc).__name__ + ": " + str(exc)})
                if attempt == 3:
                    raise
                time.sleep(attempt)
    info = retry_network(lambda: api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True), "pinned_metadata")
    if info.sha != revision or info.private or info.gated:
        raise SourceError("Expected anonymous, ungated public revision")
    indexed = {row.rfilename: row for row in info.siblings}
    absent = set(filenames) - set(indexed)
    if absent:
        raise SourceError("Missing pinned source files: " + ", ".join(sorted(absent)))
    if any(not isinstance(indexed[name].size, int) or indexed[name].size <= 0 for name in filenames):
        raise SourceError("Every selected source file must have a bounded positive metadata size")
    planned = sum(indexed[name].size for name in filenames)
    if planned <= 0 or planned > max_bytes:
        raise SourceError("Explicit source file list exceeds download budget")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    provenance_path = destination / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous.get("repo_id") != repo_id or previous.get("revision") != revision:
            raise SourceError("Source cache belongs to a different immutable revision")
    records = []
    for name in filenames:
        expected = indexed[name]
        path = destination / name
        if not path.resolve().is_relative_to(destination) or path.is_symlink():
            raise SourceError("Source destination must stay inside its cache")
        path = Path(retry_network(lambda: hf_hub_download(repo_id, name, repo_type=repo_type, revision=revision,
                                   local_dir=destination, token=False), name))
        digest = file_sha256(path)
        lfs = expected.lfs
        expected_sha = lfs.sha256 if lfs else None
        if path.stat().st_size != expected.size or expected_sha and digest != expected_sha:
            raise SourceError("Downloaded source differs from pinned size/LFS SHA: " + name)
        records.append({"relative_path": name, "path": str(path), "size_bytes": path.stat().st_size,
                        "sha256": digest, "expected_lfs_sha256": expected_sha,
                        "full_file_verified": True})
    result = {"repo_id": repo_id, "revision": revision, "repo_type": repo_type,
            "anonymous": True, "files": records, "total_bytes": planned,
            "training_started": False, "training_eligible": False, "network_retries": retries}
    if not provenance_path.exists():
        write_new_json(provenance_path, result)
    return result


DATA_FILES = ["README.md", "meta/info.json", "meta/stats.json", "meta/tasks.parquet",
              "meta/episodes/chunk-000/file-000.parquet", "data/chunk-000/file-000.parquet"]


def verify_download_receipt(directory, repo_id, revision, required_files):
    """Bind an independent audit to the complete previously verified source files."""
    directory = Path(directory).resolve()
    receipt_path = directory / "provenance.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise SourceError("Verified source download provenance is missing")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("repo_id"), receipt.get("revision")) != (repo_id, revision) or receipt.get("anonymous") is not True:
        raise SourceError("Source provenance differs from configured public revision")
    indexed = {}
    for item in receipt.get("files", []):
        name = item.get("relative_path")
        if not isinstance(name, str) or name in indexed:
            raise SourceError("Source provenance contains missing/duplicate names")
        indexed[name] = item
    verified = {}
    for name in required_files:
        if name not in indexed:
            raise SourceError("Required file is absent from pinned source provenance: " + name)
        item = indexed[name]
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.is_symlink() or not path.is_file():
            raise SourceError("Source file is not regular evidence within its declared cache")
        if item.get("path") is not None and Path(item["path"]).resolve() != path.resolve():
            raise SourceError("Source provenance path disagrees with its relative identity")
        if item.get("full_file_verified") is not True or path.stat().st_size != item.get("size_bytes"):
            raise SourceError("Source file size/full-verification evidence differs")
        digest = file_sha256(path)
        if digest != item.get("sha256") or item.get("expected_lfs_sha256") and digest != item["expected_lfs_sha256"]:
            raise SourceError("Source file hash differs from the verified pinned download")
        verified[name] = {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
    return verified


def fetch_official_sample(config, run_dir):
    settings = config["external_data"]
    result = download_public_files(settings["repo_id"], settings["revision"], "dataset", DATA_FILES,
                                   settings["local_dir"], settings["max_download_bytes"])
    result["selected_episode_ids"] = settings["episode_ids"]
    result["scope"] = "Shared tabular shard downloaded; only predeclared episodes enter the audit. Video uses bounded range probes separately."
    write_new_json(Path(run_dir) / "reports" / "external_fetch.json", result)
    return result


def fetch_native_policy(config, run_dir):
    """Fetch one explicitly configured small policy and its actual source data."""
    settings = config["offline_policy"]
    model_files = ["README.md", "config.json", "model.safetensors", "policy_preprocessor.json",
                   "policy_postprocessor.json", "policy_preprocessor_step_3_normalizer_processor.safetensors",
                   "policy_postprocessor_step_0_unnormalizer_processor.safetensors", "train_config.json"]
    model = download_public_files(settings["repo_id"], settings["revision"], "model", model_files,
                                  settings["model_dir"], settings["max_download_bytes"])
    data_files = DATA_FILES + ["videos/observation.images.global_view/chunk-000/file-000.mp4"]
    remaining = settings["max_download_bytes"] - model["total_bytes"]
    data = download_public_files(settings["dataset_repo_id"], settings["dataset_revision"], "dataset",
                                 data_files, settings["dataset_dir"], remaining)
    result = {"model": model, "native_source_dataset": data,
              "total_bytes": model["total_bytes"] + data["total_bytes"],
              "training_started": False, "project_adapter_ready": False,
              "scope": "Original policy input/data only; no GraspSquare-to-native fabrication or closed-loop deployment"}
    write_new_json(Path(run_dir) / "reports" / "offline_fetch.json", result)
    return result


def fetch_source_native_videos(config, run_dir):
    """Fetch complete selected source-camera shards after a frozen split preflight."""
    import pyarrow.parquet as pq
    settings, original = config["source_native"], config["external_data"]
    verified = verify_download_receipt(original["local_dir"], original["repo_id"], original["revision"], DATA_FILES)
    source = Path(original["local_dir"])
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != settings["source_fps"]:
        raise SourceError("Native source layout/cadence differs from the declared protocol")
    rows = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    indexed = {row["episode_index"]: row for row in rows}
    if len(indexed) != len(rows) or any(item not in indexed for item in settings["episode_ids"]):
        raise SourceError("Native episode identities are missing or duplicated")
    cameras = sorted(key for key, feature in info["features"].items() if feature["dtype"] == "video")
    if len(cameras) != 4 or settings["camera_key"] not in cameras:
        raise SourceError("The pinned source must retain its original four camera identities")
    paths_by_split, episodes, selected_files = {}, [], set()
    for split, identifiers in settings["split_episode_ids"].items():
        paths_by_split[split] = set()
        for identifier in identifiers:
            row = indexed[identifier]
            paths = {camera: info["video_path"].format(video_key=camera,
                        chunk_index=row[f"videos/{camera}/chunk_index"],
                        file_index=row[f"videos/{camera}/file_index"]) for camera in cameras}
            paths_by_split[split].update(paths.values())
            selected_files.add(paths[settings["camera_key"]])
            if not settings["chunk_steps"] <= row["length"] <= settings["max_episode_rows"]:
                raise SourceError("Native episode length is outside the declared admission bounds")
            episodes.append({"episode_id": identifier, "split": split, "rows": row["length"],
                             "recorded_tasks": row["tasks"], "original_video_paths": paths})
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if paths_by_split[left] & paths_by_split[right]:
            raise SourceError("Native split shares an original camera video file")
    if sum(row["rows"] for row in episodes) > settings["max_total_rows"]:
        raise SourceError("Native selected rows exceed the declared bound")
    plan = {"repo_id": original["repo_id"], "revision": original["revision"],
            "episode_plan": episodes, "camera_key_used": settings["camera_key"],
            "complete_video_allowlist": sorted(selected_files), "source_files": verified,
            "four_camera_file_sets_disjoint": True, "session_independence_proven": False,
            "max_video_download_bytes": settings["max_video_download_bytes"],
            "source_metadata_read_only": True, "test_images_decoded": False, "training_started": False}
    write_new_json(Path(run_dir) / "reports/source_native_plan.json", plan)
    fetched = download_public_files(original["repo_id"], original["revision"], "dataset", sorted(selected_files),
                                    settings["video_dir"], settings["max_video_download_bytes"])
    result = {"status": "complete_verified_source_videos", "plan": plan, "download": fetched,
              "training_started": False, "current_adapter_ready": False,
              "scope": "Complete pinned source-camera files, with only declared native episodes admitted to subsequent build."}
    write_new_json(Path(run_dir) / "reports/source_native_fetch.json", result)
    return result
