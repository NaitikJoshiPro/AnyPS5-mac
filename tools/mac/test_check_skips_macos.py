"""Tests for the macOS mode of tools/check_skips.py: the skip allowlist and its TechnicalDebt.md anchors."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import check_skips  # noqa: E402

TOOL = TOOLS / "check_skips.py"
DEBT = '# Debt\n\n### Metal skips (macOS)\n\n- <a id="metal-shader-clock"></a>Shader clock\n- <a id="metal-geometry-shaders"></a>Geometry\n'


def junit(*cases):
    body = []
    for name, status, message, output in cases:
        inner = f'<skipped message="{message}"/>' if status == "notrun" else (f'<failure message="{message}"/>' if status == "fail" else "")
        body.append(f'<testcase name="{name}" status="{status}">{inner}<system-out>{output}</system-out></testcase>')
    return "<testsuite>" + "".join(body) + "</testsuite>"


class AllowlistTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allow.txt"
            path.write_text(text, encoding="utf-8")
            return check_skips.allowlist(path)

    def test_entries_comments_and_blank_lines(self):
        entries = self.parse("# comment\n\nagc_a | the device lacks geometryShader | #metal-geometry-shaders\n")
        self.assertEqual(list(entries), ["agc_a"])
        self.assertEqual(entries["agc_a"].anchor, "metal-geometry-shaders")
        self.assertEqual(entries["agc_a"].line, 3)
        self.assertTrue(entries["agc_a"].reason.search("skipped, THE DEVICE LACKS GEOMETRYSHADER"))

    def test_wrong_field_count_is_an_error(self):
        for text in ("agc_a | reason\n", "agc_a | a | b | c\n", "agc_a |  | anchor\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_duplicate_test_is_an_error(self):
        with self.assertRaises(ValueError):
            self.parse("agc_a | x | y\nagc_a | z | y\n")

    def test_bad_pattern_is_an_error(self):
        with self.assertRaises(ValueError):
            self.parse("agc_a | (unclosed | y\n")

    def test_anchors_are_ids_and_heading_slugs(self):
        found = check_skips.anchors(DEBT)
        self.assertIn("metal-shader-clock", found)
        self.assertIn("metal-skips-macos", found)


class ReasonTests(unittest.TestCase):
    def test_line_start_wins(self):
        self.assertEqual(check_skips.macos_reason("skipped, SKIP_RETURN_CODE=77\nsetup\nskipped, the device lacks X"), "skipped, the device lacks X")

    def test_prefixed_skip_line_is_found(self):
        output = "skipped, SKIP_RETURN_CODE=77\ntessellation: skipped, the device has no tessellation shaders"
        self.assertEqual(check_skips.macos_reason(output), "skipped, the device has no tessellation shaders")

    def test_silent_skip_falls_back_to_ctest(self):
        self.assertEqual(check_skips.macos_reason("skipped, SKIP_RETURN_CODE=77\n"), "skipped, SKIP_RETURN_CODE=77")


class MacosCheckTests(unittest.TestCase):
    entries = None

    def setUp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allow.txt"
            path.write_text("agc_clock | lacks shaderSubgroupClock | metal-shader-clock\n"
                            "agc_geometry | lacks geometryShader | metal-geometry-shaders\n", encoding="utf-8")
            self.entries = check_skips.allowlist(path)

    def test_allowlisted_skip_passes(self):
        errors, warnings = check_skips.macos_check([("agc_clock", "skipped, the device lacks shaderSubgroupClock")], {"agc_clock": "notrun"}, self.entries, DEBT)
        self.assertEqual((errors, warnings), ([], []))

    def test_unlisted_skip_is_an_error(self):
        errors, _ = check_skips.macos_check([("agc_f64", "skipped, the device has no shaderFloat64")], {"agc_f64": "notrun"}, self.entries, DEBT)
        self.assertEqual(len(errors), 1)
        self.assertIn("agc_f64 skipped and is not in the allowlist", errors[0])

    def test_other_reason_is_an_error(self):
        errors, _ = check_skips.macos_check([("agc_clock", "skipped, no usable Vulkan device")], {"agc_clock": "notrun"}, self.entries, DEBT)
        self.assertEqual(len(errors), 1)
        self.assertIn("another reason", errors[0])

    def test_missing_anchor_is_an_error(self):
        errors, _ = check_skips.macos_check([], {}, self.entries, "# no anchors\n")
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("has no anchor" in error for error in errors))

    def test_entry_whose_test_ran_is_stale(self):
        errors, warnings = check_skips.macos_check([], {"agc_clock": "run", "agc_other": "run"}, self.entries, DEBT)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("stale allowlist entry", warnings[0])

    def test_entry_whose_test_did_not_run_is_not_stale(self):
        self.assertEqual(check_skips.macos_check([], {}, self.entries, DEBT), ([], []))

    def test_not_built_is_an_error_even_when_allowlisted(self):
        errors, _ = check_skips.macos_check([("agc_clock", "skipped, Unable to find executable\nUnable to find executable: /x")], {"agc_clock": "notrun"}, self.entries, DEBT)
        self.assertEqual(len(errors), 1)
        self.assertIn("was not built", errors[0])


class CommandLineTests(unittest.TestCase):
    def run_tool(self, content, allow, debt):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ctest.xml").write_text(content, encoding="utf-8")
            (root / "allow.txt").write_text(allow, encoding="utf-8")
            (root / "debt.md").write_text(debt, encoding="utf-8")
            return subprocess.run([sys.executable, "-I", str(TOOL), str(root / "ctest.xml"), "--job", "macos",
                                   "--allowlist", str(root / "allow.txt"), "--technical-debt", str(root / "debt.md")],
                                  capture_output=True, text=True)

    def test_expected_skips_pass(self):
        content = junit(("agc_clock", "notrun", "SKIP_RETURN_CODE=77", "skipped, the device lacks shaderSubgroupClock"),
                        ("agc_ok", "run", "", "passed"))
        result = self.run_tool(content, "agc_clock | lacks shaderSubgroupClock | metal-shader-clock\n", DEBT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 test(s) skipped, all allowlisted", result.stdout)

    def test_unexpected_skip_fails(self):
        content = junit(("agc_f64", "notrun", "SKIP_RETURN_CODE=77", "skipped, the device has no shaderFloat64"))
        result = self.run_tool(content, "agc_clock | lacks shaderSubgroupClock | metal-shader-clock\n", DEBT)
        self.assertEqual(result.returncode, 1)
        self.assertIn("error: agc_f64 skipped and is not in the allowlist", result.stdout)

    def test_malformed_allowlist_is_a_usage_error(self):
        result = self.run_tool(junit(), "agc_clock | only two fields\n", DEBT)
        self.assertEqual(result.returncode, 2)


class RepositoryAllowlistTests(unittest.TestCase):
    def test_committed_allowlist_parses_and_every_anchor_exists(self):
        entries = check_skips.allowlist(check_skips.MACOS_ALLOWLIST)
        known = check_skips.anchors(check_skips.TECHNICAL_DEBT.read_text(encoding="utf-8"))
        for entry in entries.values():
            with self.subTest(test=entry.test):
                self.assertIn(entry.anchor, known)
                self.assertTrue(entry.anchor.startswith("metal-"), "macOS skips are documented under 'Metal skips (macOS)'")

    def test_committed_allowlist_never_accepts_emulated_features(self):
        for entry in check_skips.allowlist(check_skips.MACOS_ALLOWLIST).values():
            for reason in ("skipped, the device has no shaderFloat64", "skipped, the device has no shaderBufferInt64Atomics",
                           "skipped, the device has no shaderImageInt64Atomics"):
                with self.subTest(test=entry.test, reason=reason):
                    self.assertIsNone(entry.reason.search(reason))


if __name__ == "__main__":
    unittest.main()
