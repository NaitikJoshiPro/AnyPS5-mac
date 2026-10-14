"""Tests for tools/mac/mac_progress.py on throwaway trees, plus a check that this tree's own data parses."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOL = HERE / "mac_progress.py"
REPO = HERE.parents[1]

PROGRESS = """# Progress
## 2026-10-12: M0 complete (native arm64 build)
- done
## 2026-10-13 ~20:00 — M3: title meets every criterion
- done
"""

MILESTONES = """
[[milestone]]
id = "M0"
title = "Build"
[[milestone.criterion]]
text = "a"
status = "verified"
progress = "2026-10-12: M0 complete"
[[milestone.criterion]]
text = "b"
status = "verified"
progress = "2026-10-12: M0"

[[milestone]]
id = "M3"
title = "Titles"
[[milestone.criterion]]
text = "c"
status = "verified"
progress = "2026-10-13 ~20:00 — M3"
[[milestone.criterion]]
text = "d"
status = "partial"
[[milestone.criterion]]
text = "e"
status = "open"
[[milestone.criterion]]
text = "f"
status = "open"
"""

COMPATIBILITY = """# Compatibility

## Titles

| Title | ID | Version | Status | Top blocker |
|---|---|---|---|---|
| Doom | PPSA00001 | 1 | **Playable** (soak passed) | none |
| Showcase | PPSA00002 | 1 | **Menu** (scene renders) | x |
| Kart | PPSA00003 | 1 | **Loads; radeonsi's GL draws run** | y |
| Big | PPSA00004 | 1 | **Nothing** (relinks) | z |
| Racer | PPSA00005 | 1 | **In game** | w |

## DOOM: criteria

| Criterion | Status | Evidence |
|---|---|---|
| Boots | PASS | x |
"""


def gpu_run(logs, name, finished, selectors=("all",), stopped=None, skip_errors=()):
    run = logs / name
    run.mkdir(parents=True)
    results = [{"test": "t_pass_a", "outcome": "passed", "reason": ""},
               {"test": "t_pass_b", "outcome": "passed", "reason": ""},
               {"test": "t_fail", "outcome": "failed", "reason": ""},
               {"test": "t_skip_ok", "outcome": "skipped", "reason": "skipped, Metal has none"},
               {"test": "t_skip_bad", "outcome": "skipped", "reason": "skipped, who knows"}]
    data = {"meta": {"started": finished, "finished": finished, "stopped": stopped, "not_run": [],
                     "git": {"head": "abc123456", "branch": "mac-native", "modified_files": 0},
                     "environment": {"MTL_SHADER_VALIDATION": "1"}},
            "selectors": list(selectors), "held": ["t_held"], "results": results,
            "skip_errors": [f"t_skip_bad: skip reason is not allowlisted"] if not skip_errors else list(skip_errors),
            "skip_warnings": []}
    (run / "summary.json").write_text(json.dumps(data), encoding="utf-8")
    return run


class MacProgressTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "tree"
        docs = self.root / "docs" / "mac"
        docs.mkdir(parents=True)
        (docs / "PROGRESS.md").write_text(PROGRESS, encoding="utf-8")
        (docs / "milestones.toml").write_text(MILESTONES, encoding="utf-8")
        (docs / "COMPATIBILITY.md").write_text(COMPATIBILITY, encoding="utf-8")
        self.logs = Path(self.directory.name) / "logs"
        gpu_run(self.logs, "older-full", "2026-10-13T10:00:00")
        gpu_run(self.logs, "newest-partial", "2026-10-14T12:00:00", selectors=("G1",))
        gpu_run(self.logs, "newest-stopped", "2026-10-14T13:00:00", stopped="canary failed")
        gpu_run(self.logs, "latest-full", "2026-10-14T11:00:00")

    def tearDown(self):
        self.directory.cleanup()

    def run_tool(self, *arguments):
        return subprocess.run([sys.executable, "-I", str(TOOL), "--root", str(self.root), *arguments],
                              capture_output=True, text=True)

    def snapshot(self):
        result = self.run_tool("gpu-snapshot", "--logs", str(self.logs))
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((self.root / "docs/mac/gpu-status.json").read_text(encoding="utf-8"))

    def test_snapshot_takes_the_latest_complete_full_run(self):
        data = self.snapshot()
        self.assertEqual(data["run"], "latest-full")
        self.assertEqual(data["registered"], 6)
        self.assertEqual((data["passed"], data["failed"], data["skipped"], data["skipped_allowlisted"]), (2, 1, 2, 1))
        self.assertEqual(data["held"], ["t_held"])
        self.assertEqual(data["percent"], 33.33)
        self.assertEqual(data["failures"], ["t_fail"])
        self.assertTrue(data["validation"])

    def test_snapshot_refuses_a_partial_run(self):
        result = self.run_tool("gpu-snapshot", "--run", str(self.logs / "newest-partial"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not a full run", result.stderr)

    def test_site_numbers(self):
        self.snapshot()
        out = Path(self.directory.name) / "site"
        result = self.run_tool("site", str(out), "--no-upstream")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads((out / "mac-progress.json").read_text(encoding="utf-8"))
        # M0 2/2 = 100, M3 1/4 = 25 (partial counts 0): (100 + 25) / 2
        self.assertEqual(data["milestones"]["percent"], 62.5)
        self.assertEqual([m["verified"] for m in data["milestones"]["milestones"]], [2, 1])
        # 100 + 60 + 20 + 0 + 80 over 5 titles; the criteria table below the Titles section is ignored
        self.assertEqual(data["titles"]["percent"], 52.0)
        self.assertEqual([t["stage"] for t in data["titles"]["titles"]], ["Playable", "Menu", "Loads", "Nothing", "In game"])
        self.assertEqual(data["gpu"]["percent"], 33.33)
        for name, label in (("badge-milestones.svg", "port milestones: 62.5%"), ("badge-titles.svg", "titles: 52.0%"),
                            ("badge-metal-tests.svg", "Metal GPU tests: 33.33%")):
            self.assertIn(f'aria-label="{label}"', (out / name).read_text(encoding="utf-8"))
        index = (out / "index.html").read_text(encoding="utf-8")
        self.assertIn("2 of 6 registered GPU tests passed", index)
        self.assertIn("t_skip_bad</code> (not allowlisted)", index)
        self.assertNotIn("Upstream metrics", index)
        self.assertIn("port milestones 62.5% (M0 2/2, M3 1/4)", result.stdout)

    def test_verified_needs_a_progress_entry(self):
        path = self.root / "docs/mac/milestones.toml"
        path.write_text(MILESTONES.replace('progress = "2026-10-12: M0"\n', ""), encoding="utf-8")
        self.snapshot()
        result = self.run_tool("show")
        self.assertEqual(result.returncode, 1)
        self.assertIn("names no PROGRESS.md entry", result.stderr)

    def test_unknown_progress_entry_fails(self):
        path = self.root / "docs/mac/milestones.toml"
        path.write_text(MILESTONES.replace('"2026-10-12: M0"', '"2026-10-12: M9"'), encoding="utf-8")
        self.snapshot()
        result = self.run_tool("show")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no '## ' heading starts with it", result.stderr)

    def test_unknown_status_and_stage_fail(self):
        self.snapshot()
        path = self.root / "docs/mac/milestones.toml"
        path.write_text(MILESTONES.replace('status = "open"', 'status = "done"', 1), encoding="utf-8")
        self.assertIn("has status 'done'", self.run_tool("show").stderr)
        path.write_text(MILESTONES, encoding="utf-8")
        compatibility = self.root / "docs/mac/COMPATIBILITY.md"
        compatibility.write_text(COMPATIBILITY.replace("**Menu**", "**Mostly works**"), encoding="utf-8")
        result = self.run_tool("show")
        self.assertEqual(result.returncode, 1)
        self.assertIn("known stage", result.stderr)

    def test_missing_snapshot_fails(self):
        result = self.run_tool("show")
        self.assertEqual(result.returncode, 1)
        self.assertIn("gpu-status.json is missing", result.stderr)

    def test_retarget_points_upstream_pages_at_the_branch(self):
        sys.path.insert(0, str(HERE))
        try:
            import mac_progress
        finally:
            sys.path.pop(0)
        page = '<a href="https://github.com/x/y/blob/main/tools/progress.py">x</a> every push to <code>main</code>'
        self.assertEqual(mac_progress.retarget(page, "mac-native"),
                         '<a href="https://github.com/x/y/blob/mac-native/tools/progress.py">x</a> every push to <code>mac-native</code>')

    def test_this_tree_parses(self):
        result = subprocess.run([sys.executable, "-I", str(TOOL), "--root", str(REPO), "show"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("port milestones", result.stdout)


if __name__ == "__main__":
    unittest.main()
