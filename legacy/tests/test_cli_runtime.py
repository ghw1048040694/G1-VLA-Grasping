from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from g1_vla.cli import build_parser, main
from g1_vla.runtime import RunContext


class G1CliRuntimeTest(unittest.TestCase):
    def test_parser_registers_workflows_and_forwards_native_args(self):
        parser = build_parser()
        args, unknown = parser.parse_known_args(["collect", "--dry-run", "--episodes", "1"])
        self.assertEqual(args.command, "collect")
        self.assertTrue(args.dry_run)
        self.assertIn("--episodes", unknown)
        self.assertIn("train-vla", parser._subparsers._group_actions[0].choices)

    def test_dry_run_does_not_create_runtime_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = main(["smoke", "--dry-run", "--runtime-root", tmp])
            self.assertEqual(result, 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_run_context_records_snapshot_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext.create("test", ["g1-vla", "smoke"], runtime_root=Path(tmp), run_id="G1-test", config={"paused": True})
            context.write_summary({"passed": True})
            context.finish(True)
            self.assertTrue(json.loads((context.path / "config.json").read_text())["config"]["paused"])
            self.assertEqual(json.loads((context.path / "summary.json").read_text())["run_id"], "G1-test")
            self.assertEqual(json.loads((context.path / "status.json").read_text())["status"], "success")


if __name__ == "__main__":
    unittest.main()
