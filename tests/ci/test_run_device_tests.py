# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the shared device sweep through its task-submit boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


RUNNER = Path(__file__).resolve().parents[2] / ".github/scripts/run_device_tests.sh"
ENTRIES = (
    "examples/single.py",
    "models/deepseek_v4_flash_lowlat/decode_fwd.py",
    "models/deepseek_v4_flash_lowlat/decode_sparse_attn_swa.py",
    "models/deepseek_v4_flash_lowlat/decode_compressor_ratio128.py",
)


class DeviceSweepTests(unittest.TestCase):
    def run_sweep(self, platform="a2a3", entries=ENTRIES, fail_entry=""):
        with tempfile.TemporaryDirectory(prefix="device sweep ") as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (root / "activate.sh").write_text("export PYPTO_BENCH=1 PYPTO_LOG_LEVEL=debug\n")
            scheduler = bin_dir / "task-submit"
            scheduler.write_text(f"#!{sys.executable}\n" + textwrap.dedent(r"""
                import json, os, subprocess, sys
                from pathlib import Path
                args = sys.argv[1:]
                with Path('submissions.jsonl').open('a') as output:
                    output.write(json.dumps(args) + '\n')
                env = dict(os.environ, TASK_DEVICE='3,4' if '--device-num' in args else '5')
                raise SystemExit(subprocess.run(['bash', '-c', args[args.index('--run') + 1]], env=env).returncode)
                """))
            scheduler.chmod(0o755)
            entry_source = textwrap.dedent(r"""
                import json, os, sys
                from pathlib import Path
                names = ('PYPTO_BENCH', 'PYPTO_LOG_LEVEL', 'PYPTO_WARNING_LEVEL', 'PYPTO_RUNTIME_LOG', 'PYTHONPATH')
                record = {'argv': sys.argv, 'env': {name: os.environ.get(name) for name in names}}
                with Path('executions.jsonl').open('a') as output:
                    output.write(json.dumps(record) + '\n')
                raise SystemExit(7 if sys.argv[0] == os.environ.get('FAIL_ENTRY') else 0)
                """)
            for index, name in enumerate(entries):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(("# ci: devices=2\n" if index == 1 else "") + entry_source)
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", PLATFORM=platform,
                       FILES=" ".join(entries), FAIL_ENTRY=fail_entry, PYPTO_BENCH="1")
            result = subprocess.run(["bash", str(RUNNER)], cwd=root, env=env, text=True, capture_output=True)
            def records(name):
                path = root / name
                return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            results = (root / "results.tsv").read_text() if (root / "results.tsv").exists() else ""
            return result, records("submissions.jsonl"), records("executions.jsonl"), results, str(root)

    def test_platforms_share_workload_environment_and_deadlines(self):
        for platform in ("a2a3", "a5"):
            with self.subTest(platform=platform):
                result, submissions, executions, results, root = self.run_sweep(platform)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(len(submissions), len(ENTRIES))
                self.assertEqual(len(executions), len(ENTRIES))
                for index, (submission, execution) in enumerate(zip(submissions, executions)):
                    self.assertEqual(submission[:2], ["--device", "auto"])
                    self.assertEqual(submission[submission.index("--timeout") + 1], "3600")
                    self.assertEqual(submission[submission.index("--max-time") + 1], "900")
                    devices = "3,4" if index == 1 else "5"
                    self.assertEqual(execution["argv"], [ENTRIES[index], "-p", platform, "-d", devices])
                    self.assertEqual(execution["env"], {
                        "PYPTO_BENCH": "0", "PYPTO_LOG_LEVEL": "error", "PYPTO_WARNING_LEVEL": "none",
                        "PYPTO_RUNTIME_LOG": "error", "PYTHONPATH": root,
                    })
                    self.assertEqual("--device-num" in submission, index == 1)
                    self.assertEqual("--ignore-whitelist" in submission, index == 1)
                self.assertEqual(results, "".join(f"{entry}\tpass\n" for entry in ENTRIES))

    def test_failure_is_recorded_and_remaining_entries_run(self):
        result, submissions, executions, results, _ = self.run_sweep(fail_entry=ENTRIES[1])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(len(submissions), len(ENTRIES))
        self.assertEqual(len(executions), len(ENTRIES))
        self.assertIn(f"{ENTRIES[1]}\tfail\n", results)
        self.assertTrue(results.endswith(f"{ENTRIES[-1]}\tpass\n"))

    def test_empty_selection_submits_nothing(self):
        result, submissions, executions, results, _ = self.run_sweep(entries=())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((submissions, executions, results), ([], [], ""))

    def test_unknown_platform_submits_nothing(self):
        result, submissions, executions, _, _ = self.run_sweep(platform="unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((submissions, executions), ([], []))
