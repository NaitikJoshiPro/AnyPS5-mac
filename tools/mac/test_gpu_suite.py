"""Tests for tools/mac/gpu_groups.py and tools/mac/gpu_suite.py; a fake ctest stands in for the GPU tests."""
from pathlib import Path
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import gpu_groups  # noqa: E402
import gpu_suite  # noqa: E402

FAKE_CTEST = r'''#!/usr/bin/env python3
import json, os, re, sys
arguments = sys.argv[1:]
junit = arguments[arguments.index("--output-junit") + 1]
pattern = re.compile(arguments[arguments.index("-R") + 1])
plan = json.loads(open(os.environ["FAKE_CTEST_PLAN"]).read())
with open(os.environ["FAKE_CTEST_LOG"], "a") as log:
    log.write(" ".join(arguments) + "\n")
cases = []
for name, (status, message, output) in sorted(plan.items()):
    if not pattern.search(name):
        continue
    inner = f'<skipped message="{message}"/>' if status == "notrun" else (f'<failure message="{message}"/>' if status == "fail" else "")
    cases.append(f'<testcase name="{name}" classname="{name}" time="0.5" status="{status}">{inner}<system-out>{output}</system-out></testcase>')
with open(junit, "w") as stream:
    stream.write('<?xml version="1.0" encoding="UTF-8"?><testsuite name="(empty)" tests="%d">%s</testsuite>' % (len(cases), "".join(cases)))
sys.exit(8 if any(plan[name][0] == "fail" for name in plan if pattern.search(name)) else 0)
'''


def write_build(root, tests):
    """A configured-looking build directory: CTestTestfile.cmake with gpu-labelled tests and a build.ninja link line each."""
    build = root / "build"
    (build / "sub").mkdir(parents=True)
    (build / "CTestTestfile.cmake").write_text('subdirs("sub")\n')
    lines, ninja = [], []
    for name, (timeout, source) in tests.items():
        exe = build / "tests" / f"{name}_tests"
        lines.append(f'add_test("{name}" "{exe}")')
        properties = ' LABELS "gpu"' + (f' TIMEOUT "{timeout}"' if timeout else "")
        lines.append(f'set_tests_properties("{name}" PROPERTIES {properties} _BACKTRACE_TRIPLES "x;1;add_test")')
        ninja.append(f"build tests/{name}_tests: CXX_EXECUTABLE_LINKER__{name}_tests_Release core/x/CMakeFiles/{name}_tests.dir/{source}.o | dep")
    (build / "sub" / "CTestTestfile.cmake").write_text("\n".join(lines) + "\n")
    (build / "build.ninja").write_text("\n".join(ninja) + "\n")
    return build


class GroupTests(unittest.TestCase):
    def test_held_tests(self):
        for name in ("agc_driver_lds_atomics64", "agc_driver_flip", "agc_driver_flip_compute", "video_out_flip_lifetime",
                     "video_out_frame_dump_queue", "guest_video_out_output_mode", "pad_sdl", "guest_buffer_reuse"):
            with self.subTest(name=name):
                self.assertTrue(gpu_groups.held(name))
        for name in ("agc_vector_window", "agc_vector_window_stage_inputs", "agc_driver_lds_atomics", "agc_display_buffer_dcc",
                     "linear_display_buffer", "agc_driver_vop1_control", "agc_driver_image_atomics64"):
            with self.subTest(name=name):
                self.assertFalse(gpu_groups.held(name))

    def test_classification_rules(self):
        device = "auto device = OpenVulkanTestDevice();\n"
        cases = {
            ("agc_driver_lds_atomics64", device): "G6",
            ("agc_driver_flip", device): "G5",
            ("pad_sdl", ""): "G5",
            ("agc_metal_msl_corpus", ""): "G7",
            ("agc_pipeline_cache", device): "G7",
            ("agc_driver_pipeline_optimized_readback_d16", device): "G7",
            ("agc_driver_pm4", ""): "G4",
            ("agc_shared_write_partial_collect", ""): "G4",
            ("agc_structurizer", "int main() {}"): "G0",
            ("agc_driver_f64_rounding", device + 'std::puts("skipped, the device has no shaderFloat64");'): "G6",
            ("agc_driver_flat_apertures_3", device + 'std::puts("skipped the 64-bit atomic rows, the device has no shaderBufferInt64Atomics");'): "G1",
            ("agc_driver_global_atomics64", device + 'std::puts("skipped, the device has no shaderBufferInt64Atomics");'): "G6",
            ("agc_driver_store_idle_results", device + "AgcDriver::Submit(&packet, 0);"): "G4",
            ("agc_driver_vertex_fetch", device + "device.Draw(state, draw, shaders);"): "G3",
            ("agc_driver_image_by", device + "TextureDescriptor texture{};"): "G2",
            ("agc_driver_vop1_control", device + "device.Dispatch(result, 1, 1, 1);"): "G1",
        }
        for (name, text), group in cases.items():
            with self.subTest(name=name):
                self.assertEqual(gpu_groups.classify_one(name, "core/libs/prx/libSceAgcDriver", text), group)
        self.assertEqual(gpu_groups.classify_one("guest_buffer_reuse", "core/libs/prx/libSceVideoOut", ""), "G5")
        self.assertEqual(gpu_groups.classify_one("some_cpu_test", "core/libs/prx/libSceVideoOut", ""), "G5")

    def test_executable_looks_through_cmake_env(self):
        self.assertEqual(gpu_groups.executable(["/usr/bin/cmake", "-E", "env", "--unset=X", "A=1", "/b/tests/x_tests", "--flag"]), "/b/tests/x_tests")
        self.assertEqual(gpu_groups.executable(["/b/tests/x_tests", "--flag"]), "/b/tests/x_tests")
        self.assertEqual(gpu_groups.executable([]), "")

    def test_build_directory_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = write_build(root, {"agc_driver_a": (None, "tests/execution/A.cpp"), "agc_b": ("60", "__/__/tests/B.cpp")})
            tests = gpu_groups.registered_tests(build)
            self.assertEqual(sorted(tests), ["agc_b", "agc_driver_a"])
            self.assertEqual(tests["agc_b"]["properties"]["TIMEOUT"], "60")
            self.assertEqual(tests["agc_b"]["properties"]["LABELS"], "gpu")
            self.assertEqual(tests["agc_driver_a"]["directory"], "sub")
            sources = gpu_groups.executable_sources(build)
            self.assertEqual(sources["agc_driver_a_tests"], ["core/x/tests/execution/A.cpp"])
            self.assertEqual(sources["agc_b_tests"], [os.path.normpath("core/x/../../tests/B.cpp")])

    def test_map_round_trip_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.txt"
            gpu_groups.write_map({"b": "G2", "a": "G1", "c": "G1"}, path)
            self.assertEqual(gpu_groups.load_map(path), {"a": "G1", "b": "G2", "c": "G1"})
            path.write_text("a G9\n")
            with self.assertRaises(ValueError):
                gpu_groups.load_map(path)

    def test_committed_map_holds_back_what_gpu_safety_requires(self):
        mapped = gpu_groups.load_map()
        self.assertEqual(mapped.get("agc_driver_lds_atomics64"), "G6")
        for name, group in mapped.items():
            if group in ("G1", "G2", "G3", "G4"):
                self.assertFalse(gpu_groups.held(name), name)


class SuiteHelperTests(unittest.TestCase):
    def test_select(self):
        names = ["a1", "a2", "b1", "flip"]
        groups = {"a1": "G1", "a2": "G1", "b1": "G2", "flip": "G5"}
        self.assertEqual(gpu_suite.select(names, groups, ["G1"]), ["a1", "a2"])
        self.assertEqual(gpu_suite.select(names, groups, ["report"]), ["a1", "a2", "b1"])
        self.assertEqual(gpu_suite.select(names, groups, ["re:^b", "a2"]), ["a2", "b1"])
        self.assertEqual(gpu_suite.select(names, groups, ["all"]), ["a1", "a2", "b1", "flip"])
        with self.assertRaises(ValueError):
            gpu_suite.select(names, groups, ["nonsense"])

    def test_batches_respect_the_worst_case_budget(self):
        names = [f"t{index}" for index in range(10)]
        planned = list(gpu_suite.batches(names, {"t3": "60"}, 30, 120))
        self.assertEqual(planned, [["t0", "t1", "t2"], ["t3", "t4", "t5"], ["t6", "t7", "t8", "t9"]])
        self.assertEqual(list(gpu_suite.batches(["big"], {"big": "600"}, 30, 120)), [["big"]])

    def test_first_error_prefers_emission_and_skips_request_dumps(self):
        output = "setup\nShaderRecompiler::Recompile: MSL emission failed: hash=0x1 stage=4 opcode=LaneId reason=has no MSL emitter\nRecompileRequest:\n" + "A" * 200
        self.assertIn("opcode=LaneId", gpu_suite.first_error(output))
        self.assertEqual(gpu_suite.emission_key(output), "opcode=LaneId")
        self.assertEqual(gpu_suite.emission_key("MSL emission failed: hash=0x1 stage=1 reason=only compute shaders"), "reason=only compute shaders")
        self.assertEqual(gpu_suite.first_error("ok\nvalue mismatch at 3\n"), "value mismatch at 3")
        self.assertEqual(gpu_suite.first_error("a\nb\n"), "b")

    def test_outcome(self):
        self.assertEqual(gpu_suite.outcome("run", "", ""), "passed")
        self.assertEqual(gpu_suite.outcome("fail", "Timeout", ""), "timeout")
        self.assertEqual(gpu_suite.outcome("fail", "Failed", ""), "failed")
        self.assertEqual(gpu_suite.outcome("notrun", "Unable to find executable", ""), "not-built")
        self.assertEqual(gpu_suite.outcome("notrun", "SKIP_RETURN_CODE=77", ""), "skipped")
        self.assertEqual(gpu_suite.outcome("disabled", "Disabled", ""), "skipped")


class SuiteRunTests(unittest.TestCase):
    def run_suite(self, plan, *arguments, tests=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = tests or {name: (None, "tests/execution/X.cpp") for name in plan}
            build = write_build(root, tests)
            bin_directory = root / "bin"
            bin_directory.mkdir()
            ctest = bin_directory / "ctest"
            ctest.write_text(FAKE_CTEST)
            ctest.chmod(ctest.stat().st_mode | stat.S_IEXEC)
            (root / "plan.json").write_text(json.dumps(plan))
            environment = dict(os.environ, PATH=f"{bin_directory}{os.pathsep}{os.environ.get('PATH', '')}",
                               FAKE_CTEST_PLAN=str(root / "plan.json"), FAKE_CTEST_LOG=str(root / "ctest.log"))
            out = root / "out"
            result = subprocess.run([sys.executable, "-I", str(HERE / "gpu_suite.py"), *arguments, "--build", str(build), "--out", str(out)],
                                    capture_output=True, text=True, env=environment)
            files = {name: (out / name).read_text() for name in ("summary.txt", "summary.json", "junit.xml") if (out / name).exists()}
            log = (root / "ctest.log").read_text() if (root / "ctest.log").exists() else ""
            return result, files, log

    def test_counts_failures_skips_and_never_runs_held_tests(self):
        plan = {
            "agc_driver_ok": ["run", "", "passed"],
            "agc_driver_emission": ["fail", "Failed", "MSL emission failed: hash=0x1 stage=4 opcode=FPMul32 reason=has no MSL emitter"],
            "agc_driver_skip": ["notrun", "SKIP_RETURN_CODE=77", "skipped, the device has no shaderFloat64"],
            "agc_driver_lds_atomics64": ["run", "", "must never run"],
            "agc_driver_flip": ["run", "", "must never run"],
        }
        result, files, log = self.run_suite(plan, "re:^agc_driver_", "--batch-seconds", "60")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertNotIn("lds_atomics64", log)
        self.assertNotIn("agc_driver_flip", log)
        self.assertIn("-j1", log)
        self.assertIn("--timeout 30", log)
        summary = json.loads(files["summary.json"])
        outcomes = {entry["test"]: entry["outcome"] for entry in summary["results"]}
        self.assertEqual(outcomes, {"agc_driver_ok": "passed", "agc_driver_emission": "failed", "agc_driver_skip": "skipped"})
        self.assertEqual(sorted(summary["held"]), ["agc_driver_flip", "agc_driver_lds_atomics64"])
        self.assertTrue(any("agc_driver_skip skipped and is not in the allowlist" in error for error in summary["skip_errors"]))
        self.assertIn("opcode=FPMul32", files["summary.txt"])
        self.assertIn("<testcase", files["junit.xml"])
        self.assertEqual(log.count("--output-junit"), 2)

    def test_dry_run_runs_nothing(self):
        result, files, log = self.run_suite({"agc_driver_ok": ["run", "", ""]}, "agc_driver_ok", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 test(s) in 1 batch(es)", result.stdout)
        self.assertEqual(log, "")
        self.assertEqual(files, {})

    def test_a_test_ctest_does_not_report_counts_as_failed(self):
        plan = {"agc_driver_ok": ["run", "", ""]}
        tests = {"agc_driver_ok": (None, "tests/execution/A.cpp"), "agc_driver_missing": (None, "tests/execution/B.cpp")}
        result, files, _ = self.run_suite(plan, "re:^agc_driver_", tests=tests)
        self.assertEqual(result.returncode, 1)
        outcomes = {entry["test"]: entry["outcome"] for entry in json.loads(files["summary.json"])["results"]}
        self.assertEqual(outcomes, {"agc_driver_missing": "failed", "agc_driver_ok": "passed"})

    def test_selecting_only_held_tests_runs_nothing(self):
        result, _, log = self.run_suite({"agc_driver_lds_atomics64": ["run", "", ""]}, "agc_driver_lds_atomics64")
        self.assertEqual(result.returncode, 1)
        self.assertIn("held back", result.stdout)
        self.assertEqual(log, "")


if __name__ == "__main__":
    unittest.main()
