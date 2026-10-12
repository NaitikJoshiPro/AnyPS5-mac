"""Tests for tools/mac/port_ledger.py on a throwaway git repository."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOL = HERE / "port_ledger.py"
DRIVER = "core/libs/prx/libSceAgcDriver"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def write(repo, path, text):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


class PortLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.directory.name)
        git(self.repo, "init", "-q", "-b", "main")
        write(self.repo, f"{DRIVER}/Graphics/src/Recorder.cpp", "recorder 1\n")
        write(self.repo, f"{DRIVER}/Graphics/include/Recorder.hpp", "header 1\n")
        write(self.repo, f"{DRIVER}/Graphics/src/Other.cpp", "other 1\n")
        write(self.repo, f"{DRIVER}/tests/Recorder.cpp", "test 1\n")
        write(self.repo, f"{DRIVER}/specifics/macos/twins/D2-recorder.cmake",
              'list(APPEND agcMetalTwins\n        "Graphics/src/Recorder.cpp=specifics/macos/src/Recorder.mm"\n)\n')
        write(self.repo, f"{DRIVER}/specifics/macos/src/Recorder.mm", "twin\n")
        write(self.repo, f"{DRIVER}/specifics/macos/overlay/prx/libSceAgcDriver/Graphics/include/Recorder.hpp", "twin header\n")
        write(self.repo, f"{DRIVER}/specifics/macos/overlay/prx/libSceAgcDriver/Graphics/include/MetalOnly.hpp", "new\n")
        write(self.repo, "tools/mac/port_ledger.txt", f"# ledger\ntwin-test {DRIVER}/tests/Recorder.cpp {DRIVER}/specifics/macos/tests/RecorderTwin.mm\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        self.base = git(self.repo, "rev-parse", "HEAD")
        for path, text, subject in ((f"{DRIVER}/Graphics/src/Other.cpp", "other 2\n", "unrelated"),
                                    (f"{DRIVER}/Graphics/src/Recorder.cpp", "recorder 2\n", "fix(recorder): first"),
                                    (f"{DRIVER}/Graphics/include/Recorder.hpp", "header 2\n", "fix(recorder): header"),
                                    (f"{DRIVER}/tests/Recorder.cpp", "test 2\n", "test(recorder): more")):
            write(self.repo, path, text)
            git(self.repo, "commit", "-q", "-am", subject)
        self.head = git(self.repo, "rev-parse", "HEAD")

    def tearDown(self):
        self.directory.cleanup()

    def run_tool(self, *arguments):
        return subprocess.run([sys.executable, "-I", str(TOOL), "--repo", str(self.repo), *arguments], capture_output=True, text=True)

    def test_lists_changes_to_twinned_originals_only(self):
        result = self.run_tool("--since", self.base, "--to", "HEAD", "--stat")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3 twinned original(s), 3 changed", result.stdout)
        self.assertIn("fix(recorder): first", result.stdout)
        self.assertIn("fix(recorder): header", result.stdout)
        self.assertIn("test(recorder): more", result.stdout)
        self.assertIn("twin: core/libs/prx/libSceAgcDriver/specifics/macos/src/Recorder.mm", result.stdout)
        self.assertIn("(+1 -1)", result.stdout)
        self.assertNotIn("unrelated", result.stdout)
        self.assertNotIn("MetalOnly", result.stdout)

    def test_check_fails_while_something_is_pending(self):
        self.assertEqual(self.run_tool("--since", self.base, "--to", "HEAD", "--check").returncode, 1)
        self.assertEqual(self.run_tool("--since", self.head, "--to", "HEAD", "--check").returncode, 0)

    def test_mark_records_the_reviewed_commit(self):
        result = self.run_tool("--mark", "HEAD")
        self.assertEqual(result.returncode, 0, result.stderr)
        ledger = (self.repo / "tools/mac/port_ledger.txt").read_text()
        self.assertIn(f"reviewed {self.head}", ledger)
        self.assertIn("twin-test", ledger)
        self.assertEqual(self.run_tool("--mark", "HEAD").returncode, 0)
        self.assertEqual((self.repo / "tools/mac/port_ledger.txt").read_text().count("reviewed "), 1)
        result = self.run_tool("--to", "HEAD", "--check")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("0 changed", result.stdout)

    def test_bad_ledger_line_is_a_usage_error(self):
        write(self.repo, "tools/mac/port_ledger.txt", "nonsense line here\n")
        self.assertEqual(self.run_tool("--since", self.base, "--to", "HEAD").returncode, 2)


if __name__ == "__main__":
    unittest.main()
