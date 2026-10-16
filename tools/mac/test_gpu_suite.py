"""Tests for tools/mac/gpu_groups.py and tools/mac/gpu_suite.py; a fake ctest stands in for the GPU tests."""
from pathlib import Path
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import gpu_groups  # noqa: E402
import buildlock  # noqa: E402
import gpu_suite  # noqa: E402

# Never touch the real machine-wide GPU lock from tests.
os.environ[buildlock.GPU_LOCK_ENV] = os.path.join(tempfile.mkdtemp(prefix="aps5-gpulock-test-"), "locks", "gpu.lock")

FAKE_CTEST = r'''#!/usr/bin/env python3
import json, os, re, sys
arguments = sys.argv[1:]
junit = arguments[arguments.index("--output-junit") + 1]
pattern = re.compile(arguments[arguments.index("-R") + 1])
plan = json.loads(open(os.environ["FAKE_CTEST_PLAN"]).read())
with open(os.environ["FAKE_CTEST_LOG"], "a") as log:
    log.write(" ".join(arguments) + "\n")
if os.environ.get("FAKE_CTEST_ENV_LOG"):
    with open(os.environ["FAKE_CTEST_ENV_LOG"], "a") as seen:
        seen.write("SDL_VIDEODRIVER=" + os.environ.get("SDL_VIDEODRIVER", "<unset>") + "\n")
cases = []
for name, (status, message, output) in sorted(plan.items()):
    if not pattern.search(name):
        continue
    inner = f'<skipped message="{message}"/>' if status == "notrun" else (f'<failure message="{message}"/>' if status == "fail" else "")
    cases.append(f'<testcase name="{name}" classname="{name}" time="0.5" status="{status}">{inner}<system-out>{output}</system-out></testcase>')
with open(junit, "w") as stream:
    stream.write('<?xml version="1.0" encoding="UTF-8"?><testsuite name="(empty)" tests="%d">%s</testsuite>' % (len(cases), "".join(cases)))
if os.environ.get("FAKE_CTEST_TOUCH"):
    with open(os.environ["FAKE_CTEST_TOUCH"], "a") as touched:
        touched.write("rebuilt\n")
sys.exit(8 if any(plan[name][0] == "fail" for name in plan if pattern.search(name)) else 0)
'''

FAKE_NINJA = r'''#!/usr/bin/env python3
import os, sys
with open(os.environ["FAKE_NINJA_LOG"], "a") as log:
    log.write(" ".join(sys.argv[1:]) + "\n")
# What a real dry run prints with CONFIGURE_DEPENDS globs: the forced glob check and a CMake re-run.
print("ninja explain: build/CMakeFiles/VerifyGlobs.cmake_force is dirty", file=sys.stderr)
print("[0/2] Re-checking globbed directories...")
print("ninja explain: build/CMakeFiles/cmake.verify_globs is dirty", file=sys.stderr)
if os.environ.get("FAKE_NINJA_EXPLAIN"):
    print("ninja explain: " + os.environ["FAKE_NINJA_EXPLAIN"], file=sys.stderr)
print("[1/2] Re-running CMake...")
print(os.environ.get("FAKE_NINJA_OUTPUT", ""))
'''

FAKE_SYSCTL = r'''#!/bin/sh
echo "{ ${FAKE_LOAD:-0.50} 0.40 0.30 }"
'''

FAKE_PS = r'''#!/bin/sh
printf '%s\n' /sbin/launchd /usr/bin/python3 ${FAKE_PROCESSES}
'''

CANARY = "agc_canary_ok"


def source_repository(root):
    """A committed git tree standing in for the source tree that configured the build."""
    source = root / "source"
    (source / "core").mkdir(parents=True)
    (source / "core" / "a.cpp").write_text("int a;\n")
    (source / "docs").mkdir()
    (source / "docs" / "notes.md").write_text("notes\n")
    for command in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(source), *command], check=True, capture_output=True)
    return source


def write_build(root, tests, source=None):
    """A configured-looking build directory: CTestTestfile.cmake with gpu-labelled tests and a build.ninja link line each."""
    build = root / "build"
    (build / "sub").mkdir(parents=True)
    if source is not None:
        (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n")
        prx = build / gpu_suite.PRX
        prx.parent.mkdir(parents=True)
        prx.write_bytes(b"prx")
    (build / "CTestTestfile.cmake").write_text('subdirs("sub")\n')
    lines, ninja = [], []
    for name, (timeout, source) in tests.items():
        exe = build / "tests" / f"{name}_tests"
        lines.append(f'add_test("{name}" "{exe}")')
        properties = ' LABELS "gpu"' + (f' TIMEOUT "{timeout}"' if timeout else "")
        lines.append(f'set_tests_properties("{name}" PROPERTIES {properties} _BACKTRACE_TRIPLES "x;1;add_test")')
        ninja.append(f"build tests/{name}_tests: CXX_EXECUTABLE_LINKER__{name}_tests_Release core/x/CMakeFiles/{name}_tests.dir/{source}.o | dep")
    (build / "sub" / "CTestTestfile.cmake").write_text("\n".join(lines) + "\n")
    if source is not None:
        ninja.append(f"build {gpu_suite.PRX}: CXX_SHARED_LIBRARY_LINKER__libSceAgcDriver_Release dep")
    (build / "build.ninja").write_text("\n".join(ninja) + "\n")
    return build


class GroupTests(unittest.TestCase):
    def test_held_tests(self):
        self.assertFalse(gpu_groups.held("agc_driver_lds_atomics64"))
        self.assertEqual(gpu_groups.HELD_TESTS, ())
        with mock.patch.object(gpu_groups, "HOLD", gpu_groups.hold_pattern(("agc_driver_fake_held",))):
            self.assertTrue(gpu_groups.held("agc_driver_fake_held"))
            self.assertFalse(gpu_groups.held("agc_driver_fake_held_too"))
            self.assertFalse(gpu_groups.held("agc_driver_lds_atomics64"))
        for name in ("agc_driver_flip", "agc_driver_flip_compute", "video_out_flip_lifetime", "video_out_frame_dump_queue",
                     "guest_video_out_output_mode", "pad_sdl", "guest_buffer_reuse"):
            with self.subTest(name=name):
                self.assertFalse(gpu_groups.held(name))
                self.assertTrue(gpu_groups.PRESENTATION.search(name))
                self.assertEqual(gpu_groups.classify_one(name, "x", ""), "G5")
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
                self.assertFalse(bool(gpu_groups.HOLD.search(name)), name)

    def test_quarantine_file(self):
        gpu_groups.load_quarantine()  # the committed list parses (it may be empty)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quarantine.txt"
            path.write_text("# comment\nagc_x | it panicked | anchor-x\n\n")
            self.assertEqual(gpu_groups.load_quarantine(path), {"agc_x": "it panicked (anchor-x)"})
            for bad in ("agc_x | no anchor\n", "agc_x | | anchor\n", "agc x | reason | anchor\n"):
                path.write_text(bad)
                with self.assertRaises(ValueError, msg=bad):
                    gpu_groups.load_quarantine(path)
            with self.assertRaises(OSError):
                gpu_groups.load_quarantine(Path(directory) / "missing.txt")


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


HELD_ENV = {"GPU_GROUPS_EXTRA_HELD": "agc_driver_fake_held"}


class SuiteRunTests(unittest.TestCase):
    def run_suite(self, plan, *arguments, tests=None, canaries=(CANARY,), env=None, prepare=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(plan)
            for canary in canaries:
                plan.setdefault(canary, ["run", "", "canary"])
            tests = dict(tests or {name: (None, "tests/execution/X.cpp") for name in plan})
            for canary in canaries:
                tests.setdefault(canary, (None, "tests/execution/Canary.cpp"))
            source = source_repository(root)
            build = write_build(root, tests, source)
            bin_directory = root / "bin"
            bin_directory.mkdir()
            for name, text in (("ctest", FAKE_CTEST), ("ninja", FAKE_NINJA), ("sysctl", FAKE_SYSCTL), ("ps", FAKE_PS)):
                tool = bin_directory / name
                tool.write_text(text)
                tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "quarantine.txt").write_text("agc_driver_f16_denormal_flush | kernel panic in a test | metal-gpu-quarantine\n")
            (root / "canaries.txt").write_text("# canaries\n" + "".join(f"{name}\n" for name in canaries))
            environment = dict(os.environ, PATH=f"{bin_directory}{os.pathsep}{os.environ.get('PATH', '')}",
                               FAKE_CTEST_PLAN=str(root / "plan.json"), FAKE_CTEST_LOG=str(root / "ctest.log"),
                               FAKE_NINJA_LOG=str(root / "ninja.log"))
            environment.pop("FAKE_PROCESSES", None)
            environment.update({key: value.format(build=build, root=root) for key, value in (env or {}).items()})
            if prepare:
                prepare(root, source, build)
            out = root / "out"
            started = time.monotonic()
            result = subprocess.run([sys.executable, "-I", str(HERE / "gpu_suite.py"), *arguments, "--build", str(build), "--out", str(out),
                                     "--canaries", str(root / "canaries.txt"), "--quarantine", str(root / "quarantine.txt"), "--busy-wait", "0"],
                                    capture_output=True, text=True, env=environment)
            result.seconds = time.monotonic() - started
            files = {name: (out / name).read_text() for name in ("summary.txt", "summary.json", "junit.xml") if (out / name).exists()}
            log = (root / "ctest.log").read_text() if (root / "ctest.log").exists() else ""
            result.ninja = (root / "ninja.log").read_text() if (root / "ninja.log").exists() else ""
            return result, files, log

    def test_counts_failures_skips_and_never_runs_held_tests(self):
        plan = {
            "agc_driver_ok": ["run", "", "passed"],
            "agc_driver_emission": ["fail", "Failed", "MSL emission failed: hash=0x1 stage=4 opcode=FPMul32 reason=has no MSL emitter"],
            "agc_driver_skip": ["notrun", "SKIP_RETURN_CODE=77", "skipped, the device has no shaderFloat64"],
            "agc_driver_fake_held": ["run", "", "must never run"],
            "agc_driver_flip": ["run", "", "presentation runs headless"],
        }
        result, files, log = self.run_suite(plan, "re:^agc_driver_", "--batch-seconds", "60", env=HELD_ENV)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertNotIn("fake_held", log)
        self.assertIn("agc_driver_flip", log)
        self.assertIn("-j1", log)
        self.assertIn("--timeout 30", log)
        summary = json.loads(files["summary.json"])
        outcomes = {entry["test"]: entry["outcome"] for entry in summary["results"]}
        self.assertEqual(outcomes, {"agc_driver_ok": "passed", "agc_driver_emission": "failed", "agc_driver_skip": "skipped", "agc_driver_flip": "passed"})
        self.assertEqual(sorted(summary["held"]), ["agc_driver_fake_held"])
        self.assertTrue(any("agc_driver_skip skipped and is not in the allowlist" in error for error in summary["skip_errors"]))
        self.assertIn("opcode=FPMul32", files["summary.txt"])
        self.assertIn("<testcase", files["junit.xml"])
        self.assertEqual(log.count("--output-junit"), 5)  # the canary, then one ctest process per test
        self.assertTrue(log.splitlines()[0].endswith("^agc_canary_ok$") or "agc_canary_ok" in log.splitlines()[0])

    def test_tests_run_with_sdl_offscreen(self):
        with tempfile.TemporaryDirectory() as scratch:
            seen = Path(scratch) / "env.log"
            result, files, _ = self.run_suite({"agc_driver_flip": ["run", "", ""]}, "agc_driver_flip",
                                              env={"FAKE_CTEST_ENV_LOG": str(seen), "SDL_VIDEODRIVER": "cocoa"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            lines = seen.read_text().splitlines()
            self.assertGreaterEqual(len(lines), 2)  # the canary and the test
            self.assertEqual(set(lines), {"SDL_VIDEODRIVER=offscreen"})
            self.assertEqual(json.loads(files["summary.json"])["meta"]["environment"]["SDL_VIDEODRIVER"], "offscreen")

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

    def test_never_runs_quarantined_tests(self):
        plan = {"agc_driver_ok": ["run", "", ""], "agc_driver_f16_denormal_flush": ["run", "", "must never run"]}
        tests = {"agc_driver_ok": (None, "tests/execution/A.cpp"), "agc_driver_f16_denormal_flush": (None, "tests/execution/F16DenormalFlush.cpp")}
        result, files, log = self.run_suite(plan, "re:^agc_driver_", tests=tests)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("f16_denormal", log)
        self.assertIn("refusing to run quarantined test agc_driver_f16_denormal_flush: kernel panic", result.stdout)
        self.assertIn("quarantined agc_driver_f16_denormal_flush", files["summary.txt"])
        result, _, log = self.run_suite(plan, "agc_driver_f16_denormal_flush", tests=tests)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(log, "")

    def test_a_quarantined_test_reruns_only_alone_under_validation(self):
        plan = {"agc_driver_ok": ["run", "", ""], "agc_driver_f16_denormal_flush": ["run", "", "rerun"]}
        tests = {"agc_driver_ok": (None, "tests/execution/A.cpp"), "agc_driver_f16_denormal_flush": (None, "tests/execution/F16DenormalFlush.cpp")}
        result, _, log = self.run_suite(plan, "agc_driver_f16_denormal_flush", "--rerun-quarantined", tests=tests)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(log, "")
        result, _, log = self.run_suite(plan, "re:^agc_driver_", "--rerun-quarantined", "--validation", tests=tests)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(log, "")
        result, files, log = self.run_suite(plan, "agc_driver_f16_denormal_flush", "--rerun-quarantined", "--validation", tests=tests)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("^agc_driver_f16_denormal_flush$", log)
        self.assertEqual(json.loads(files["summary.json"])["meta"]["quarantine_rerun"], ["agc_driver_f16_denormal_flush"])

    def test_a_held_test_runs_only_alone_under_validation(self):
        plan = {"agc_driver_ok": ["run", "", ""], "agc_driver_fake_held": ["run", "", "held rerun"]}
        result, _, log = self.run_suite(plan, "agc_driver_fake_held", "--rerun-held", "agc_driver_fake_held")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(log, "")
        result, _, log = self.run_suite(plan, "re:^agc_driver_", "--rerun-held", "agc_driver_fake_held", "--validation", env=HELD_ENV)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(log, "")
        result, _, log = self.run_suite(plan, "agc_driver_ok", "--rerun-held", "agc_driver_ok", "--validation", env=HELD_ENV)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(log, "")
        result, files, log = self.run_suite(plan, "agc_driver_fake_held", "--rerun-held", "agc_driver_fake_held", "--validation", env=HELD_ENV)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("^agc_driver_fake_held$", log)
        self.assertNotIn("agc_driver_ok$", log)
        self.assertEqual(json.loads(files["summary.json"])["meta"]["held_rerun"], ["agc_driver_fake_held"])
        result, _, log = self.run_suite(plan, "agc_driver_fake_held", "--validation", env=HELD_ENV)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(log, "")

    def test_selecting_only_held_tests_runs_nothing(self):
        result, _, log = self.run_suite({"agc_driver_fake_held": ["run", "", ""]}, "agc_driver_fake_held", env=HELD_ENV)
        self.assertEqual(result.returncode, 1)
        self.assertIn("held back", result.stdout)
        self.assertEqual(log, "")


class GuardTests(unittest.TestCase):
    """DECISIONS D19: gpu_suite refuses inconsistent binaries, a loaded machine and failing canaries."""
    run_suite = SuiteRunTests.run_suite
    PLAN = {"agc_driver_ok": ["run", "", ""]}

    def assertRefused(self, result, log, text):
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn(text, result.stdout + result.stderr)
        self.assertEqual(log, "", "no test may run after a refusal")

    def test_refuses_uncommitted_changes_under_core(self):
        def edit(root, source, build):
            (source / "core" / "a.cpp").write_text("int b;\n")
        result, files, log = self.run_suite(self.PLAN, "agc_driver_ok", prepare=edit)
        self.assertRefused(result, log, "uncommitted changes")
        self.assertIn("core/a.cpp", json.loads(files["summary.json"])["meta"]["guards"]["dirty"])

    def test_busy_wait_waits_for_a_dirty_tree_to_become_clean(self):
        tree = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tree, ignore_errors=True))
        states = iter([["core/a.cpp"], ["core/a.cpp"], []])
        with mock.patch.object(gpu_suite, "dirty_paths", lambda source: next(states)), mock.patch.object(gpu_suite.time, "sleep"):
            self.assertEqual(gpu_suite.wait_until_clean(tree, 60, poll=0.01), [])
        with mock.patch.object(gpu_suite, "dirty_paths", lambda source: ["core/a.cpp"]):
            self.assertEqual(gpu_suite.wait_until_clean(tree, 0), ["core/a.cpp"])

    def test_refuses_untracked_files_under_specifics(self):
        def add(root, source, build):
            (source / "specifics").mkdir()
            (source / "specifics" / "new.mm").write_text("x\n")
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", prepare=add)
        self.assertRefused(result, log, "specifics/new.mm")

    def test_changes_outside_core_and_specifics_are_allowed(self):
        def edit(root, source, build):
            (source / "docs" / "notes.md").write_text("more\n")
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", prepare=edit)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("agc_driver_ok", log)

    def test_refuses_stale_binaries(self):
        env = {"FAKE_NINJA_OUTPUT": "[1/2] Linking CXX shared library core/libs/libs/unpatched/libSceAgcDriver.prx"}
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", env=env)
        self.assertRefused(result, log, "build step(s) pending")

    def test_a_cmake_rerun_counts_only_with_a_real_reason(self):
        env = {"FAKE_NINJA_EXPLAIN": "output build.ninja older than most recent input /src/CMakeLists.txt (1 vs 2)"}
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", env=env)
        self.assertRefused(result, log, "Re-running CMake")

    def test_stale_check_names_tests_canaries_and_the_prx(self):
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        targets = result.ninja.split()
        for target in ("tests/agc_driver_ok_tests", "tests/agc_canary_ok_tests", gpu_suite.PRX):
            self.assertIn(target, targets)
        self.assertIn("-n", targets)

    def test_refuses_under_load(self):
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", env={"FAKE_LOAD": "7.25"})
        self.assertRefused(result, log, "load average 7.25 > 6")

    def test_refuses_while_a_compiler_runs(self):
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", env={"FAKE_PROCESSES": "/usr/bin/clang++ /opt/x/bin/swift-frontend"})
        self.assertRefused(result, log, "compiler/linker running: clang++ swift-frontend")

    def test_refuses_without_a_canary_list(self):
        def remove(root, source, build):
            (root / "canaries.txt").unlink()
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", prepare=remove)
        self.assertRefused(result, log, "canary list")

    def test_a_failing_canary_stops_the_run(self):
        plan = {"agc_canary_ok": ["fail", "Failed", "error: canary broke"], "agc_driver_ok": ["run", "", ""]}
        result, files, log = self.run_suite(plan, "agc_driver_ok")
        self.assertEqual(result.returncode, 1)
        self.assertIn("canary agc_canary_ok failed", result.stderr)
        self.assertNotIn("agc_driver_ok", log)
        summary = json.loads(files["summary.json"])
        self.assertEqual(summary["meta"]["not_run"], ["agc_driver_ok"])
        self.assertEqual(summary["meta"]["canaries"], {"agc_canary_ok": "failed"})

    def test_canaries_run_first_once_and_tests_are_paused(self):
        plan = {"agc_driver_a": ["run", "", ""], "agc_driver_b": ["run", "", ""]}
        result, files, log = self.run_suite(plan, "re:^agc_", "--validation")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        runs = [line.split("-R ")[1].split()[0] for line in log.splitlines()]
        self.assertEqual(runs, ["^agc_canary_ok$", "^agc_driver_a$", "^agc_driver_b$"])
        self.assertGreaterEqual(result.seconds, 2 * gpu_suite.PAUSE_SECONDS)
        meta = json.loads(files["summary.json"])["meta"]
        self.assertEqual(meta["environment"]["MTL_SHADER_VALIDATION"], "1")
        self.assertEqual(meta["environment"]["MTL_DEBUG_LAYER"], "1")
        self.assertEqual(meta["prx"]["sha256"], hashlib.sha256(b"prx").hexdigest())
        self.assertRegex(meta["git"]["head"], r"^[0-9a-f]{7,}$")
        self.assertIn("sha256 " + hashlib.sha256(b"prx").hexdigest(), files["summary.txt"])

    def test_timeout_is_capped(self):
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", "--timeout", "60")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(log, "")
        tests = {"agc_driver_ok": ("900", "tests/execution/X.cpp")}
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", tests=tests)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--timeout 30 -R ^agc_driver_ok$", log)
        self.assertEqual(gpu_suite.effective_timeout("900", 30), 30)
        self.assertEqual(gpu_suite.effective_timeout("10", 30), 10)
        self.assertEqual(gpu_suite.effective_timeout(None, 20), 20)

    def test_pause_below_two_seconds_is_rejected(self):
        result, _, log = self.run_suite(self.PLAN, "agc_driver_ok", "--pause", "0.5")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(log, "")

    def test_a_binary_rebuilt_during_the_run_fails_it(self):
        result, files, _ = self.run_suite(self.PLAN, "agc_driver_ok", env={"FAKE_CTEST_TOUCH": "{build}/" + gpu_suite.PRX})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("changed while the suite ran", files["summary.txt"])

    def test_the_process_group_is_killed_at_the_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            ctest = root / "bin" / "ctest"
            ctest.write_text("#!/bin/sh\nsleep 60 &\necho $! > \"$FAKE_CHILD\"\nwait\n")
            ctest.chmod(ctest.stat().st_mode | stat.S_IEXEC)
            environment = dict(os.environ, PATH=f"{root / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}", FAKE_CHILD=str(root / "child"))
            started = time.monotonic()
            code, seconds, killed = gpu_suite.run_test(root, "x", 1, root / "x.xml", root / "x.log", environment)
            self.assertTrue(killed)
            self.assertLess(time.monotonic() - started, 20)
            child = int((root / "child").read_text())
            time.sleep(0.2)
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.build = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def write_lease(self, pid, started, max_seconds):
        (self.build / buildlock.LEASE_NAME).write_text(json.dumps(
            {"pid": pid, "started": started, "max_seconds": max_seconds, "run": "r1"}))

    def dead_pid(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()
        return process.pid

    def builder(self, *extra_env):
        env = dict(os.environ)
        return subprocess.Popen([sys.executable, "-I", str(HERE / "buildlock.py"), str(self.build), "--", "true"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    def test_a_live_lease_blocks_a_builder_until_released(self):
        buildlock.take_lease(self.build, 600, run="r1")
        process = self.builder()
        time.sleep(1.0)
        self.assertIsNone(process.poll(), "the builder must wait for the lease")
        buildlock.release_lease(self.build)
        _, error = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0)
        self.assertIn("waiting for gpu_suite run r1", error)
        self.assertIn(f"pid {os.getpid()}", error)

    def test_a_stale_lease_is_ignored_and_removed(self):
        for pid, started, limit in ((self.dead_pid(), time.time(), 600), (os.getpid(), time.time() - 700, 600)):
            self.write_lease(pid, started, limit)
            process = self.builder()
            process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0)
            self.assertFalse((self.build / buildlock.LEASE_NAME).exists())

    def test_an_unreadable_lease_is_ignored(self):
        (self.build / buildlock.LEASE_NAME).write_text("not json")
        self.assertIsNone(buildlock.read_lease(self.build))
        self.assertFalse((self.build / buildlock.LEASE_NAME).exists())

    def test_release_only_removes_an_own_lease(self):
        self.write_lease(self.dead_pid(), time.time(), 600)
        buildlock.release_lease(self.build)
        self.assertTrue((self.build / buildlock.LEASE_NAME).exists())

    def test_the_owner_is_not_blocked_by_its_own_lease(self):
        buildlock.take_lease(self.build, 600)
        self.assertLess(buildlock.wait_for_lease(self.build, owner_pid=os.getpid(), poll=0.01), 1)

    def test_lease_duration_is_the_worst_case_plus_ten_minutes_capped(self):
        registered = {"a": {"properties": {"TIMEOUT": "10"}}, "b": {"properties": {}}}
        self.assertEqual(gpu_suite.lease_seconds(["a", "b"], registered, 30, 2.0), 10 + 2 + 30 + 2 + 600)
        many = {str(i): {"properties": {}} for i in range(500)}
        self.assertEqual(gpu_suite.lease_seconds(list(many), many, 30, 2.0), 5400)

    def test_gpu_suite_releases_the_lease_on_an_exception(self):
        def boom():
            buildlock.take_lease(self.build, 600)
            gpu_suite.held_leases.append(self.build)
            raise RuntimeError("boom")
        original = gpu_suite.run_suite
        gpu_suite.run_suite = boom
        try:
            with self.assertRaises(RuntimeError):
                gpu_suite.main()
        finally:
            gpu_suite.run_suite = original
        self.assertFalse((self.build / buildlock.LEASE_NAME).exists())
        self.assertEqual(gpu_suite.held_leases, [])


HOLDER = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
import buildlock
with buildlock.gpu_lock(label="holder-cmd"):
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
"""


class GpuLockTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "locks", "gpu.lock")
        self.env = dict(os.environ, **{buildlock.GPU_LOCK_ENV: self.path})
        self.env.pop(buildlock.GPU_HELD_ENV, None)
        self.patch = mock.patch.dict(os.environ, {buildlock.GPU_LOCK_ENV: self.path})
        self.patch.start()
        os.environ.pop(buildlock.GPU_HELD_ENV, None)

    def tearDown(self):
        self.patch.stop()
        self.dir.cleanup()

    def start_holder(self, seconds):
        process = subprocess.Popen([sys.executable, "-c", HOLDER, str(HERE), str(seconds)], env=self.env,
                                   stdout=subprocess.PIPE, text=True)
        self.assertEqual(process.stdout.readline().strip(), "held")
        return process

    def test_second_process_waits_and_prints_holder(self):
        import io
        holder = self.start_holder(1.5)
        out = io.StringIO()
        start = time.monotonic()
        with buildlock.gpu_lock(label="second", out=out, poll=0.05):
            waited = time.monotonic() - start
        holder.wait()
        self.assertGreater(waited, 0.5)
        text = out.getvalue()
        self.assertIn("waiting for", text)
        self.assertIn(f"pid {holder.pid}", text)
        self.assertIn("holder-cmd", text)
        self.assertIn("since 20", text)
        self.assertIn("acquired after", text)

    def test_dead_holder_does_not_block(self):
        import io
        holder = self.start_holder(60)
        holder.kill()
        holder.wait()
        self.assertTrue(os.path.exists(self.path + ".info"))  # stale sidecar left behind
        out = io.StringIO()
        start = time.monotonic()
        with buildlock.gpu_lock(label="next", out=out, poll=0.05):
            self.assertEqual(buildlock.read_gpu_holder()["pid"], os.getpid())
        self.assertLess(time.monotonic() - start, 5)
        self.assertEqual(out.getvalue(), "")
        self.assertFalse(os.path.exists(self.path + ".info"))

    def test_nested_in_holder_environment_is_a_no_op(self):
        with buildlock.gpu_lock(label="outer"):
            self.assertEqual(os.environ[buildlock.GPU_HELD_ENV], str(os.getpid()))
            with buildlock.gpu_lock(label="inner"):  # would deadlock if it re-took the flock
                pass
        self.assertNotIn(buildlock.GPU_HELD_ENV, os.environ)

    def test_run_test_takes_the_gpu_lock_per_test(self):
        build = Path(self.dir.name) / "build"
        build.mkdir()
        seen = []
        real = subprocess.Popen

        def fake_popen(command, **kwargs):
            seen.append(buildlock.read_gpu_holder())
            return real([sys.executable, "-c", "pass"], **kwargs)

        with mock.patch.object(gpu_suite.subprocess, "Popen", fake_popen):
            gpu_suite.run_test(build, "some_test", 10, build / "j.xml", build / "t.log", dict(os.environ))
        self.assertEqual(seen[0]["pid"], os.getpid())
        self.assertIn("some_test", seen[0]["command"])
        self.assertIsNone(buildlock.read_gpu_holder())  # released between tests

    def test_buildlock_gpu_option_waits_for_the_gpu_lock(self):
        holder = self.start_holder(1.5)
        build = os.path.join(self.dir.name, "b")
        start = time.monotonic()
        result = subprocess.run([sys.executable, str(HERE / "buildlock.py"), "--gpu", build, "--", "true"], env=self.env,
                                capture_output=True, text=True)
        holder.wait()
        self.assertEqual(result.returncode, 0)
        self.assertGreater(time.monotonic() - start, 0.5)
        self.assertIn(f"pid {holder.pid}", result.stderr)


if __name__ == "__main__":
    unittest.main()
