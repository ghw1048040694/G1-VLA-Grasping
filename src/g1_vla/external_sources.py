"""Bounded, revision-pinned public source downloads; no training imports."""
from pathlib import Path
import hashlib
import json

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
    info = api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    if info.sha != revision or info.private or info.gated:
        raise SourceError("Expected anonymous, ungated public revision")
    indexed = {row.rfilename: row for row in info.siblings}
    absent = set(filenames) - set(indexed)
    if absent:
        raise SourceError("Missing pinned source files: " + ", ".join(sorted(absent)))
    planned = sum(indexed[name].size or 0 for name in filenames)
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
        path = Path(hf_hub_download(repo_id, name, repo_type=repo_type, revision=revision,
                                   local_dir=destination, token=False))
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
            "training_started": False, "training_eligible": False}
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
