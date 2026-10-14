import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from g1_vla.external_sources import SourceError, download_public_files


class ExternalSourceTests(unittest.TestCase):
    def fixture(self, content=b"source", downloaded=None):
        digest = hashlib.sha256(content).hexdigest()
        calls = []
        info = SimpleNamespace(sha="a" * 40, private=False, gated=False,
            siblings=[SimpleNamespace(rfilename="file", size=len(content), lfs=SimpleNamespace(sha256=digest))])
        def download(repo, name, **kwargs):
            calls.append(kwargs)
            path = Path(kwargs["local_dir"]) / name
            path.write_bytes(content if downloaded is None else downloaded)
            return str(path)
        module = SimpleNamespace(HfApi=lambda **kwargs: SimpleNamespace(repo_info=lambda *args, **kw: info),
                                 hf_hub_download=download)
        return module, calls, info

    def test_budget_refusal_precedes_payload_and_pin_mismatch_is_rejected(self):
        module, calls, info = self.fixture()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaisesRegex(SourceError, "budget"):
                download_public_files("owner/repo", "a" * 40, "dataset", ["file"], directory, 1)
            self.assertEqual(calls, [])
            info.sha = "b" * 40
            with self.assertRaisesRegex(SourceError, "revision"):
                download_public_files("owner/repo", "a" * 40, "dataset", ["file"], directory, 100)

    def test_lfs_integrity_and_anonymous_provenance(self):
        module, calls, info = self.fixture()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": module}):
            report = download_public_files("owner/repo", "a" * 40, "dataset", ["file"], directory, 100)
            self.assertTrue(report["files"][0]["full_file_verified"])
            self.assertIs(calls[0]["token"], False)
            self.assertTrue((Path(directory) / "provenance.json").is_file())
        module, calls, info = self.fixture(downloaded=b"wrong!")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaisesRegex(SourceError, "LFS SHA"):
                download_public_files("owner/repo", "a" * 40, "dataset", ["file"], directory, 100)
            self.assertFalse((Path(directory) / "provenance.json").exists())

    def test_transient_metadata_errors_have_bounded_recorded_retries(self):
        module, calls, info = self.fixture()
        attempts = []
        def metadata(*args, **kwargs):
            attempts.append(kwargs)
            if len(attempts) <= 2:
                raise OSError("transient TLS EOF")
            return info
        module.HfApi = lambda **kwargs: SimpleNamespace(repo_info=metadata)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": module}), patch("g1_vla.external_sources.time.sleep"):
            report = download_public_files("owner/repo", "a" * 40, "dataset", ["file"], directory, 100)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(len(report["network_retries"]), 2)
            self.assertEqual(len(calls), 1)
            self.assertTrue(all(row["revision"] == "a" * 40 for row in attempts))

    def test_unbounded_size_or_duplicate_allowlist_precedes_payload(self):
        module, calls, info = self.fixture()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaisesRegex(SourceError, "duplicates"):
                download_public_files("owner/repo", "a" * 40, "dataset", ["file", "file"], directory, 100)
            info.siblings[0].size = None
            with self.assertRaisesRegex(SourceError, "bounded"):
                download_public_files("owner/repo", "a" * 40, "dataset", ["file"], directory, 100)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
