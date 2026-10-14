"""CPU-only historical continuous model regression; never runs training.

The checkpoint is optional on clean checkouts. It remains outside Git. The
historical module is imported only by this regression test, never production.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path

from g1_vla.policy import CheckpointContractError, load_checkpoint, torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path("outputs/G1LANG-672_yellow_factorized_arm_hand_dagger/best.pt")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_parity(project_root=ROOT):
    if torch is None:
        raise unittest.SkipTest("PyTorch unavailable")
    root = Path(project_root)
    checkpoint_path = root / FIXTURE
    legacy = root / "legacy/scripts"
    if not checkpoint_path.is_file() or not (legacy / "g1_continuous_dagger_vla.py").is_file():
        raise unittest.SkipTest("Historical G1LANG-672 fixture or legacy source unavailable")
    # CPU is explicit for every tensor and checkpoint; no optimizer is imported
    # or created. Disable CUDA externally as well when invoking the regression.
    torch.set_num_threads(2)
    before = _sha256(checkpoint_path)
    try:
        load_checkpoint(checkpoint_path, "cpu")
    except CheckpointContractError:
        default_loader_rejected = True
    else:
        raise AssertionError("Historical checkpoint unexpectedly passed the default rebuild loader")
    model, vocab, stats, value = load_checkpoint(checkpoint_path, "cpu", historical_baseline=True)
    if value.get("load_role") != "historical_regression_baseline":
        raise AssertionError("Historical load role was lost")
    spec = importlib.util.spec_from_file_location("g1_historical_continuous_parity", legacy / "g1_continuous_dagger_vla.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(legacy))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    old = module.ContinuousDaggerVLA(**value["config"]).to("cpu")
    old.load_state_dict(value["model"], strict=True)
    old.eval()
    model.eval()
    old_state, new_state = old.state_dict(), model.state_dict()
    if list(old_state) != list(new_state) or any(old_state[key].shape != new_state[key].shape for key in old_state):
        raise AssertionError("Historical and package parameter names/shapes differ")
    if any(not torch.equal(old_state[key], new_state[key]) for key in old_state):
        raise AssertionError("Strict loading changed historical parameter values")
    generator = torch.Generator(device="cpu").manual_seed(20261014)
    steps = model.history_frames
    batch = 2
    inputs = (
        torch.rand(batch, steps, 3, 3, 64, 96, generator=generator, device="cpu"),
        torch.randn(batch, steps, 31, generator=generator, device="cpu"),
        torch.randn(batch, steps, 31, generator=generator, device="cpu"),
        torch.randn(batch, steps, model.contact_site_dim, generator=generator, device="cpu"),
        torch.randint(0, 2, (batch, steps, 1), generator=generator, device="cpu").float(),
        torch.randint(0, len(vocab), (batch, 16), generator=generator, device="cpu"),
        torch.ones(batch, 16, dtype=torch.bool, device="cpu"),
    )
    with torch.no_grad():
        expected, actual = old(*inputs), model(*inputs)
    if set(expected) != set(actual):
        raise AssertionError("Historical output names differ")
    parity = {key: bool(torch.equal(expected[key], actual[key])) for key in expected}
    if not all(parity.values()):
        raise AssertionError("Historical forward changed: " + str(parity))
    after = _sha256(checkpoint_path)
    if before != after:
        raise AssertionError("Historical checkpoint content changed during verification")
    return {"schema_version": 1, "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256_before": before, "checkpoint_sha256_after": after,
            "parameter_key_count": len(new_state), "parameter_names_and_shapes_identical": True,
            "parameter_values_identical": True, "outputs_bitwise_identical": parity,
            "default_loader_rejected_historical_checkpoint": default_loader_rejected,
            "historical_load_role": value["load_role"], "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
            "training_started": False, "optimizer_created": False, "physics_steps": 0,
            "weights_updated": False}


class HistoricalModelParityTest(unittest.TestCase):
    def test_historical_672_strict_cpu_forward_and_unchanged_checkpoint(self):
        report = verify_model_parity()
        self.assertTrue(all(report["outputs_bitwise_identical"].values()))
        self.assertEqual(report["checkpoint_sha256_before"], report["checkpoint_sha256_after"])
        self.assertFalse(report["optimizer_created"])


if __name__ == "__main__":
    unittest.main()
