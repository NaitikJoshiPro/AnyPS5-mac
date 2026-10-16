"""Pure-Python tests for tools/mac/title_run.py (no title, GPU or subprocess)."""
import itertools
import os
import tempfile
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import title_run  # noqa: E402
import buildlock  # noqa: E402


class PadScriptTests(unittest.TestCase):
    def test_intro_and_loop(self):
        intro, loop = title_run.parse_pad_script(
            "# comment\nwait 2\npress cross 300  # start\nloop\npress release 1500 128 0 128 128\nwait 0.5\n")
        self.assertEqual(intro, [("wait", 2.0), ("press", "cross", 300, None)])
        self.assertEqual(loop, [("press", "release", 1500, (128, 0, 128, 128)), ("wait", 0.5)])

    def test_default_hold(self):
        intro, _ = title_run.parse_pad_script("press r2\n")
        self.assertEqual(intro, [("press", "r2", 300, None)])

    def test_errors(self):
        for text in ("press nope\n", "press cross 0\n", "press cross 6000\n", "press r2 100 1 2 3 256\n",
                     "jump\n", "wait -1\n", "loop\nloop\n", "loop\nwait 0\n", "press r2 100 1 2\n"):
            with self.subTest(text=text), self.assertRaises(title_run.PadScriptError):
                title_run.parse_pad_script(text)

    def test_iter_repeats_loop(self):
        steps = list(itertools.islice(title_run.iter_pad_steps([("wait", 1.0)], [("press", "r2", 10, None)]), 4))
        self.assertEqual(steps, [("wait", 1.0)] + [("press", "r2", 10, None)] * 3)

    def test_iter_without_loop_ends(self):
        self.assertEqual(list(title_run.iter_pad_steps([("wait", 1.0)], [])), [("wait", 1.0)])

    def test_pad_line_matches_pad_input(self):
        self.assertEqual(title_run.pad_line("cross", 300, None, 5), "5 16384 300\n")
        self.assertEqual(title_run.pad_line("release", 900, (128, 0, 255, 128), 7), "7 0 900 128 0 255 128\n")

    def test_shipped_scripts_parse(self):
        scripts = sorted((HERE / "pads").glob("*.txt"))
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                intro, loop = title_run.parse_pad_script(script.read_text())
                self.assertTrue(intro or loop)


class ProcessTests(unittest.TestCase):
    PS = """  1     0 /sbin/launchd
  100     1 /usr/bin/python3
  101   100 /usr/bin/perl
  102   101 /Users/x/build-metal/core/fex/aps5-fex
  200     1 /other/aps5-fex
"""

    def test_find_descendant(self):
        table = title_run.parse_ps_table(self.PS)
        self.assertEqual(table[102], (101, "/Users/x/build-metal/core/fex/aps5-fex"))
        self.assertEqual(title_run.find_descendant(table, 100, "aps5-fex"), 102)
        self.assertIsNone(title_run.find_descendant(table, 101 + 1000, "aps5-fex"))

    def test_cputime(self):
        self.assertAlmostEqual(title_run.parse_cputime("1:02.50"), 62.5)
        self.assertAlmostEqual(title_run.parse_cputime("1:00:01.00"), 3601.0)
        self.assertAlmostEqual(title_run.parse_cputime("2-00:00:00.00"), 172800.0)

    def test_sample_and_translation_flag(self):
        rss, cpu, flags = title_run.parse_sample("  524288   3:10.25     4004")
        self.assertEqual((rss, cpu, flags), (524288, 190.25, 0x4004))
        self.assertFalse(title_run.is_translated(flags))
        self.assertTrue(title_run.is_translated(0x24004))
        self.assertIsNone(title_run.parse_sample("garbage"))

    def test_leak_slope(self):
        flat = [(t, 1000) for t in range(0, 600, 30)]
        self.assertEqual(title_run.leak_slope_mib_per_min(flat), 0.0)
        growing = [(t, 1024 * t / 60) for t in range(0, 600, 30)]  # 1 MiB per minute
        self.assertAlmostEqual(title_run.leak_slope_mib_per_min(growing), 1.0)
        self.assertEqual(title_run.leak_slope_mib_per_min([(0, 5)]), 0.0)

    def test_run_verdict(self):
        self.assertEqual(title_run.run_verdict(242), "alive")
        self.assertEqual(title_run.run_verdict(142), "alive")
        self.assertEqual(title_run.run_verdict(0), "exited")
        self.assertEqual(title_run.run_verdict(1), "failed")
        self.assertEqual(title_run.run_verdict(-11), "failed")


class NativeProofTests(unittest.TestCase):
    VMMAP = """Process:         aps5-fex [4242]
Path:            /Users/x/build-metal/core/fex/aps5-fex
Code Type:       ARM64
__TEXT                      100000000-100400000    [ 4096K  4096K     0K     0K] r-x/r-x SM=COW          /Users/x/build-metal/core/fex/aps5-fex
__TEXT                      101000000-101100000    [ 1024K  1024K     0K     0K] r-x/r-x SM=COW          /Users/x/libs/libSceAgcDriver.dylib
__TEXT                      101000000-101100000    [ 1024K  1024K     0K     0K] r-x/r-x SM=COW          /Users/x/libs/libSceAgcDriver.dylib
MALLOC_SMALL                600000000-600800000    [ 8192K   512K   512K     0K] rw-/rwx SM=PRV
"""

    def test_images_and_code_type(self):
        self.assertEqual(title_run.vmmap_images(self.VMMAP),
                         ["/Users/x/build-metal/core/fex/aps5-fex", "/Users/x/libs/libSceAgcDriver.dylib"])
        self.assertEqual(title_run.vmmap_code_type(self.VMMAP), "ARM64")
        self.assertEqual(title_run.vmmap_x86_findings(self.VMMAP), [])

    def test_rosetta_found(self):
        text = self.VMMAP + "__TEXT  7ff000-7ff100 [ 1K ] r-x/r-x SM=COW  /Library/Apple/usr/libexec/oah/libRosettaRuntime\n"
        self.assertEqual(len(title_run.vmmap_x86_findings(text)), 1)
        self.assertEqual(title_run.vmmap_code_type("Code Type: X86-64 (translated)\n"), "X86-64 (translated)")

    def test_baseline_stub_is_not_rosetta(self):
        stub = "__TEXT   2b756a000-2b7571000 [ 28K 12K 0K 0K] r-x/r-x SM=COW          /usr/lib/libRosetta.dylib"
        runtime = "__TEXT   7ff000-7ff100 [ 1K ] r-x/r-x SM=COW  /Library/Apple/usr/libexec/oah/libRosettaRuntime"
        new, baseline = title_run.split_findings([stub, runtime], stub + "\n" + runtime + "\n")
        self.assertEqual((new, baseline), ([runtime], [stub]))
        new, baseline = title_run.split_findings([stub], "")
        self.assertEqual((new, baseline), ([stub], []))

    def test_file_verdict(self):
        self.assertEqual(title_run.file_verdict("Mach-O 64-bit executable arm64"), "arm64")
        self.assertEqual(title_run.file_verdict("Mach-O universal binary with 2 architectures: [x86_64] [arm64]"), "universal")
        self.assertEqual(title_run.file_verdict("Mach-O 64-bit executable x86_64"), "x86")
        self.assertEqual(title_run.file_verdict("ELF 64-bit LSB executable, x86-64"), "other")
        self.assertEqual(title_run.file_verdict("data"), "other")


class EnvAndAudioTests(unittest.TestCase):
    def test_env_headless(self):
        env = title_run.build_env({"A": "1"}, windowed=False, frames=6, every=30, validation=True,
                                  pad_file=Path("/r/pad"), audio_file=Path("/r/a.raw"))
        self.assertEqual(env["SDL_VIDEODRIVER"], "offscreen")
        self.assertEqual((env["APS5_DUMP_FRAMES"], env["APS5_DUMP_FRAMES_EVERY"]), ("6", "30"))
        self.assertEqual((env["MTL_DEBUG_LAYER"], env["MTL_SHADER_VALIDATION"]), ("1", "1"))
        self.assertEqual(env["APS5_PAD_INPUT_FILE"], "/r/pad")
        self.assertEqual((env["SDL_AUDIODRIVER"], env["SDL_DISKAUDIOFILE"]), ("disk", "/r/a.raw"))
        self.assertEqual(env["A"], "1")

    def test_env_windowed_plain(self):
        env = title_run.build_env({"SDL_VIDEODRIVER": "cocoa"}, windowed=True, frames=0, every=1,
                                  validation=False, pad_file=None, audio_file=None)
        self.assertEqual(env["SDL_VIDEODRIVER"], "cocoa")
        for key in ("MTL_DEBUG_LAYER", "APS5_PAD_INPUT_FILE", "SDL_AUDIODRIVER"):
            self.assertNotIn(key, env)

    def test_load(self):
        self.assertTrue(title_run.load_ok(5.9, 6))
        self.assertFalse(title_run.load_ok(6.0, 6))

    def test_pcm_windows(self):
        silence = struct.pack("<4h", 0, 0, 0, 0)
        loud = struct.pack("<4h", 16384, -16384, 16384, -16384)
        rms = title_run.pcm_windows(silence + loud, "s16", 2, 2, 1.0)
        self.assertEqual(len(rms), 2)
        self.assertEqual(rms[0], 0.0)
        self.assertAlmostEqual(rms[1], 0.5)
        f32 = struct.pack("<2f", 0.25, -0.25)
        self.assertAlmostEqual(title_run.pcm_windows(f32, "f32", 2, 1, 1.0)[0], 0.25)


class LibsLinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.run_dir = root / "runs" / "m3" / "T"
        self.run_dir.mkdir(parents=True)
        self.builds = {}
        for name in ("build-a", "build-b"):
            (root / name / "core/libs/libs").mkdir(parents=True)
            self.builds[name] = root / name
        self.out = open(os.devnull, "w")

    def tearDown(self):
        self.out.close()
        self.tmp.cleanup()

    def target(self):
        return Path(os.readlink(self.run_dir / "libs"))

    def test_matching_link_is_kept(self):
        (self.run_dir / "libs").symlink_to(self.builds["build-a"] / "core/libs/libs")
        self.assertIsNone(title_run.ensure_libs_link(self.run_dir, self.builds["build-a"], out=self.out))
        self.assertEqual(self.target(), self.builds["build-a"] / "core/libs/libs")

    def test_stale_link_is_repointed_at_the_run_build(self):
        (self.run_dir / "libs").symlink_to(self.builds["build-a"] / "core/libs/libs")
        previous = title_run.ensure_libs_link(self.run_dir, self.builds["build-b"], out=self.out)
        self.assertEqual(previous, self.builds["build-a"] / "core/libs/libs")
        self.assertEqual(self.target(), self.builds["build-b"].resolve() / "core/libs/libs")

    def test_dangling_or_missing_link_points_at_the_run_build(self):
        (self.run_dir / "libs").symlink_to(Path(self.tmp.name) / "gone/core/libs/libs")
        title_run.ensure_libs_link(self.run_dir, self.builds["build-b"], out=self.out)
        self.assertEqual(self.target(), self.builds["build-b"].resolve() / "core/libs/libs")
        (self.run_dir / "libs").unlink()
        title_run.ensure_libs_link(self.run_dir, self.builds["build-a"], out=self.out)
        self.assertEqual(self.target(), self.builds["build-a"].resolve() / "core/libs/libs")

    def test_refuses_a_build_without_libraries_or_a_real_directory(self):
        with self.assertRaises(title_run.LibsLinkError):
            title_run.ensure_libs_link(self.run_dir, Path(self.tmp.name) / "build-none", out=self.out)
        (self.run_dir / "libs").mkdir()
        with self.assertRaises(title_run.LibsLinkError):
            title_run.ensure_libs_link(self.run_dir, self.builds["build-a"], out=self.out)

    def test_run_refuses_before_starting_the_runner(self):
        (self.run_dir / "app.elf").write_bytes(b"")
        (self.run_dir / "libs").mkdir()
        args = mock.Mock(id="T", build=str(self.builds["build-a"]), log_dir=str(Path(self.tmp.name) / "logs"))
        with mock.patch.object(title_run, "work_dir", return_value=Path(self.tmp.name)), \
                mock.patch.object(title_run.subprocess, "Popen") as popen, mock.patch("sys.stderr", self.out):
            self.assertEqual(title_run.run_locked(args), 2)
            popen.assert_not_called()


class GpuLockTests(unittest.TestCase):
    def test_cmd_run_holds_the_gpu_lock_for_the_whole_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "locks", "gpu.lock")
            seen = []
            args = mock.Mock(id="T1", tag="x", build=tmp)
            with mock.patch.dict(os.environ, {buildlock.GPU_LOCK_ENV: path}), \
                    mock.patch.object(title_run, "run_locked", lambda a: seen.append(buildlock.read_gpu_holder(path)) or 7):
                os.environ.pop(buildlock.GPU_HELD_ENV, None)
                self.assertEqual(title_run.cmd_run(args), 7)
                self.assertIsNone(buildlock.read_gpu_holder(path))
            self.assertEqual(seen[0]["pid"], os.getpid())
            self.assertIn("title_run T1", seen[0]["command"])

    def test_lock_order_has_no_deadlock_between_a_suite_and_a_title_run(self):
        """A suite-like process holds the lease and takes the GPU lock per test; a title-run-like one waits for the
        lease, takes the GPU lock and runs buildlock.py as a child. Before the fix each waited on the other forever."""
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        suite = (
            "import sys, time\nsys.path.insert(0, %r)\nimport buildlock\nbuild = sys.argv[1]\n"
            "buildlock.take_lease(build, 60, run='t')\ntry:\n print('leased', flush=True)\n"
            " time.sleep(1.0)\n for i in range(3):\n  with buildlock.gpu_lock(label='suite test', poll=0.05):\n   time.sleep(0.3)\n"
            "finally:\n buildlock.release_lease(build)\n" % here)
        title = (
            "import sys, subprocess\nsys.path.insert(0, %r)\nimport buildlock\nbuild = sys.argv[1]\n"
            "buildlock.wait_for_lease(build, poll=0.05)\nwith buildlock.gpu_lock(label='title run', poll=0.05):\n"
            " import time; time.sleep(0.5)\n"
            " sys.exit(subprocess.call([sys.executable, %r, build, '--', 'true']))\n" % (here, os.path.join(here, "buildlock.py")))
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, **{buildlock.GPU_LOCK_ENV: os.path.join(tmp, "locks", "gpu.lock")})
            env.pop(buildlock.GPU_HELD_ENV, None)
            build = os.path.join(tmp, "build")
            suite_proc = subprocess.Popen([sys.executable, "-c", suite, build], env=env, stdout=subprocess.PIPE, text=True)
            self.assertEqual(suite_proc.stdout.readline().strip(), "leased")
            # Title run starts while the lease exists; it waits for it, then races the suite for the GPU lock. Force the
            # bad interleaving too: the title run holds the GPU lock while the suite (lease holder) is queued.
            title_proc = subprocess.Popen([sys.executable, "-c", title, build], env=env)
            try:
                self.assertEqual(title_proc.wait(timeout=30), 0)
                self.assertEqual(suite_proc.wait(timeout=30), 0)
            finally:
                for proc in (suite_proc, title_proc):
                    if proc.poll() is None:
                        proc.kill()

    def test_buildlock_under_the_gpu_lock_does_not_wait_for_the_lease(self):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as tmp:
            build = os.path.join(tmp, "build")
            os.makedirs(build)
            with open(os.path.join(build, buildlock.LEASE_NAME), "w") as stream:
                import json, time
                json.dump({"pid": os.getppid(), "started": time.time(), "max_seconds": 600, "run": "x"}, stream)
            env = dict(os.environ, **{buildlock.GPU_HELD_ENV: "1"})
            code = subprocess.run([sys.executable, os.path.join(here, "buildlock.py"), build, "--", "true"],
                                  env=env, timeout=20).returncode
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
