"""Pure-Python tests for tools/mac/profile_title.py and profile_parse.py (no title, GPU or subprocess)."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import profile_parse as pp  # noqa: E402
import profile_title  # noqa: E402
import trace_digest  # noqa: E402

SAMPLE = """Analysis of sampling aps5-fex (pid 1) every 1 millisecond
Process:         aps5-fex [1]

Call graph:
    100 Thread_11   DispatchQueue_1: com.apple.main-thread  (serial)
    + 100 start  (in dyld) + 6688  [0x18ad07e80]
    +   100 main  (in aps5-fex) + 2168  [0x102d358a8]  Main.cpp:93
    +     100 CFRunLoopRun  (in CoreFoundation) + 64  [0x18b2154c4]
    +       100 mach_msg2_trap  (in libsystem_kernel.dylib) + 8  [0x18b098c20]
    100 Thread_12
    + 90 ???  (in <unknown binary>)  [0xfc44004494]
    + ! 60 Aps5BridgeThunkCall  (in aps5-fex) + 88  [0x102d3af40]  Bridge.cpp:977
    + ! : 60 Aps5NativeCall  (in aps5-fex) + 112  [0x102d4dae0]  Trampolines.S:42
    + ! :   50 sceKernelUsleep_nid_postfix  (in libkernel.prx) + 580  [0x107fa3f94]  Time.cpp:258
    + ! :   | 50 TimedWait::SleepNanos(unsigned long long)  (in libkernel.prx) + 84  [0x107fa8fbc]  TimedWait.cpp:344
    + ! :   |   50 nanosleep  (in libsystem_c.dylib) + 220  [0x18af73054]
    + ! :   |     50 __semwait_signal  (in libsystem_kernel.dylib) + 8,16  [0x18b09c308,0x18b09c310]
    + ! :   10 GuestWriteWatchProtection_nid_postfix  (in libc.prx) + 100  [0x10812f9c0]  GuestWriteWatch.cpp:726
    + ! :     10 cthread_yield  (in libsystem_pthread.dylib) + 36  [0x18b0dad08]
    + ! :       10 swtch_pri  (in libsystem_kernel.dylib) + 8  [0x18b098c74]
    + ! 30 ???  (in <unknown binary>)  [0xfc44004498]
    + 10 FEXCore::Context::ContextImpl::CompileCode(FEXCore::Core::InternalThreadState*, unsigned long long)  (in aps5-fex) + 4  [0x1]
    +   7 malloc  (in libsystem_malloc.dylib) + 4  [0x2]
    100 Thread_13
    + 100 thread_start  (in libsystem_pthread.dylib) + 8  [0x18b0d8cec]
    +   100 AgcDriver::DriverDetail::Driver::run(unsigned int)  (in libSceAgcDriver.prx) + 2628  [0x10c7fafe8]  QueueWorker.cpp:210
    +     40 AgcDriver::Graphics::(anonymous namespace)::WaitEvent(id<MTLSharedEvent>, unsigned long long, char const*)  (in libSceAgcDriver.prx) + 144  [0x10c8c7d34]  Recorder.mm:289
    +     ! 40 -[IOSurfaceSharedEvent waitUntilSignaledValue:timeoutMS:]  (in IOSurface) + 72  [0x197d79508]
    +     !   40 iokit_user_client_trap  (in IOKit) + 8  [0x18f5d8cc0]
    +     30 AgcDriver::GuestMemory::GpuMutexType::lock()  (in libSceAgcDriver.prx) + 8  [0x1]
    +     ! 30 __psynch_mutexwait  (in libsystem_kernel.dylib) + 8  [0x18b09b9dc]
    +     20 AgcDriver::DecodeDisplayBuffer(AgcDriver::DisplayBuffer const&)  (in libSceAgcDriver.prx) + 8  [0x1]
    +     10 AgcDriver::Metal::Residency::Commit()  (in libSceAgcDriver.prx) + 8  [0x1]
    +       10 -[IOGPUMetalCommandQueue commit]  (in IOGPU) + 8  [0x1]

Total number in stack (recursive counted multiple, when >=5):
"""

POWERMETRICS = """*** Sampled system activity (Thu Oct 15 13:48:01 2026 +0530) (505.16ms elapsed) ***
**** Processor usage ****
E-Cluster HW active frequency: 1227 MHz
E-Cluster HW active residency:  86.47% (744 MHz:   0% 1044 MHz:  85% 2748 MHz: 2.9%)
CPU 0 active residency:  50.00% (744 MHz: 0% 1044 MHz: 50%)
CPU 1 active residency:  10.00% (744 MHz: 0% 1044 MHz: 10%)
CPU 2 active residency:   0.00% (744 MHz: 0% 1044 MHz: 0%)
CPU 3 active residency:   0.00% (744 MHz: 0% 1044 MHz: 0%)
P-Cluster HW active frequency: 0 MHz
P-Cluster HW active residency:   0.00% (816 MHz:   0% 4056 MHz:   0%)
CPU 4 active residency:  40.00% (816 MHz: 0% 3000 MHz: 20% 4056 MHz: 20%)
CPU 5 active residency:   0.00% (816 MHz: 0% 4056 MHz: 0%)
CPU 6 active residency:   0.00% (816 MHz: 0% 4056 MHz: 0%)
CPU 7 active residency:   0.00% (816 MHz: 0% 4056 MHz: 0%)
CPU Power: 2042 mW
GPU Power: 809 mW
**** Thermal pressure ****
Current pressure level: Nominal
**** GPU usage ****
GPU HW active frequency: 378 MHz
GPU HW active residency:  82.70% (338 MHz:  80% 618 MHz: 2.7% 1338 MHz:   0%)
GPU idle residency:  17.30%
*** Sampled system activity (Thu Oct 15 13:48:02 2026 +0530) (500.00ms elapsed) ***
GPU HW active frequency: 400 MHz
GPU HW active residency:  90.00% (338 MHz:  90% 1056 MHz: 0%)
Current pressure level: Moderate
"""

GPUTIME_OLD = ("[gputime] 0 ms of GPU time in 604 batches over 10 s (batch 5369 ms first to last command; classes 0 ms, "
               "all ranges 0 ms, untimed 5369 ms outside every range; 0 presents; 0 ranges dropped at the 512 cap); "
               "by program:; by class: label-run x303 0.2ms 0.0MiB (per present x0.0 0.00ms 0.00MiB)")
GPUTIME_NEW = ("[gputime] 12 ms of GPU time in 640 batches over 10 s (batch 6119 ms first to last command; classes 178 ms, "
               "all ranges 178 ms, untimed 5941 ms outside every range; 3 presents; 1 ranges dropped at the 512 cap, "
               "463 with an unwritten or reversed stamp); by program: 0x1 x2 12ms; by class: storage-upload x150 87.5ms "
               "1646.9MiB (per present x0.0 0.00ms 0.00MiB) storage-writeback x389 90.5ms 1614.9MiB (per present x0.0 "
               "0.00ms 0.00MiB)")

FEX_SAMPLE = """[fex-sample] sampling guest threads every 2 ms, report every 15 s
[fex-sample] window 0: 15.0 s, 43 guest threads registered
[fex-sample] thread 0 "": 1000 samples, 170 distinct RIPs
[fex-sample]    50.0% rip 0x118d204cc libc.prx.guest.prx+0x104cc host libsystem_kernel.dylib`__read_nocancel+0x8
[fex-sample]          rax=4150029f rcx=500145300
[fex-sample]    30.0% rip 0x1000 app.elf+0x397ee jit
[fex-sample]    10.0% rip 0x104e7ec70 ?+0x104e7ec70 host libSceAgcDriver.prx`Submit+0x8
[fex-sample] thread 3 "worker": 200 samples, 4 distinct RIPs
[fex-sample]   100.0% rip 0x2000 app.elf+0x10 jit
"""


class SampleTests(unittest.TestCase):
    def setUp(self):
        self.threads = pp.parse_sample(SAMPLE)

    def test_threads_and_self_counts(self):
        self.assertEqual(sorted(self.threads), ["Thread_11", "Thread_12", "Thread_13"])
        self.assertEqual(self.threads["Thread_11"]["queue"], "com.apple.main-thread")
        for thread in self.threads.values():
            self.assertEqual(sum(thread["stacks"].values()), thread["total"], thread["tid"])

    def test_categories(self):
        summary = pp.summarize_sample(self.threads)
        totals = summary["totals"]
        self.assertEqual(totals["wait: guest sleep"], 50)
        self.assertEqual(totals["spin/yield"], 10)
        self.assertEqual(totals["guest JIT code"], 30)  # the outer JIT frame's 90 are its children's
        self.assertEqual(totals["FEX compile"], 10)  # the malloc under CompileCode is translation work
        self.assertEqual(totals["wait: GPU"], 40)
        self.assertEqual(totals["wait: lock"], 30)
        self.assertEqual(totals["AGC driver"], 20)
        self.assertEqual(totals["Metal API"], 10)
        self.assertEqual(summary["details"]["spin/yield"]["GuestWriteWatchProtection_nid_postfix"], 10)
        roles = {thread["key"]: thread["role"] for thread in summary["threads"]}
        self.assertEqual(roles["Thread_13"], "queue worker")

    def test_file_io(self):
        path = (("ResolvePath_nid_no_patch", "libc.prx"), ("std::__fs::filesystem::__status(path const&)", "libc++.1.dylib"),
                ("stat", "libsystem_kernel.dylib"))
        self.assertEqual(pp.classify_stack(path), ("kernel: file I/O", "ResolvePath_nid_no_patch"))

    def test_short_symbol(self):
        self.assertEqual(pp.short_symbol("A::(anonymous namespace)::Wait(id<X>, unsigned long) const"), "A::{anon}::Wait")
        self.assertEqual(pp.short_symbol("std::f<std::g<int>>(int)"), "std::f")


class CountsTests(unittest.TestCase):
    def raw(self, levels):
        data = struct.pack("<HHI", len(levels), 0, 0)
        for level in levels:
            data += struct.pack("<5Q", *level)
        return data

    def test_decode(self):
        levels = pp.decode_thread_counts(self.raw([(10, 20, 30, 40, 50), (1, 2, 3, 4, 5)]))
        self.assertEqual(levels[0], {"instructions": 10, "cycles": 20, "time": 70, "energy_nj": 50})
        self.assertEqual(levels[1]["time"], 7)
        self.assertEqual(pp.decode_thread_counts(b"\0" * 4), [])

    def test_diff(self):
        def entry(p_time, e_time, p_cycles=0, p_instructions=0):
            return {"name": "t", "levels": [
                {"instructions": p_instructions, "cycles": p_cycles, "time": p_time, "energy_nj": 0},
                {"instructions": 0, "cycles": 0, "time": e_time, "energy_nj": 0}]}
        before = {"1": entry(0, 0), "2": entry(100, 100)}
        after = {"1": entry(750_000, 250_000, 3_000_000, 6_000_000), "2": entry(100, 100), "3": entry(0, 1000)}
        rows = pp.diff_thread_counts(before, after, ns_per_tick=1.0, seconds=1.0)
        self.assertEqual([row["tid"] for row in rows], ["1", "3"])  # thread 2 did nothing
        self.assertAlmostEqual(rows[0]["cpu_ms"], 1.0)
        self.assertAlmostEqual(rows[0]["p_share"], 0.75)
        self.assertAlmostEqual(rows[0]["p_ipc"], 2.0)
        self.assertAlmostEqual(rows[0]["p_ghz"], 4.0)  # 3e6 cycles in 0.75 ms
        self.assertAlmostEqual(rows[1]["p_share"], 0.0)  # a thread started inside the window counts from zero


class PowermetricsTests(unittest.TestCase):
    def test_parse_and_summary(self):
        samples = pp.parse_powermetrics(POWERMETRICS)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["gpu_mhz"], 378)
        self.assertEqual(samples[0]["pressure"], "Nominal")
        self.assertEqual(samples[0]["cpu_active_pct"][4], 40.0)
        summary = pp.summarize_powermetrics(samples)
        self.assertEqual(summary["pressure"], {"Nominal": 1, "Moderate": 1})
        self.assertAlmostEqual(summary["gpu_active_pct"], (82.7 + 90.0) / 2)
        self.assertEqual(summary["gpu_peak_state_mhz"], 618)
        self.assertAlmostEqual(summary["p_active_mhz"], 3528.0)  # 20% at 3000 + 20% at 4056
        self.assertAlmostEqual(summary["e_active_mhz"], 1044.0)


class LogLineTests(unittest.TestCase):
    def test_gputime_both_forms(self):
        old, new = pp.parse_gputime(GPUTIME_OLD + "\nnoise\n" + GPUTIME_NEW)
        self.assertEqual(old["batches"], 604)
        self.assertEqual(old["invalid"], 0)
        self.assertEqual(old["classes"]["label-run"]["count"], 303)
        self.assertEqual(new["program_ms"], 12)
        self.assertEqual(new["invalid"], 463)
        self.assertEqual(new["presents"], 3)
        self.assertAlmostEqual(new["classes"]["storage-writeback"]["mib"], 1614.9)

    def test_fex_sample(self):
        windows = pp.parse_fex_sample(FEX_SAMPLE)
        self.assertEqual(len(windows), 1)
        self.assertEqual([thread["samples"] for thread in windows[0]["threads"]], [1000, 200])
        first = windows[0]["threads"][0]["rips"]
        self.assertEqual([rip["kind"] for rip in first], ["host", "jit", "host"])
        summary = pp.summarize_fex_sample(windows)
        self.assertEqual(summary["samples"], 1200)
        self.assertAlmostEqual(summary["kinds"]["jit"], 300 + 200)
        self.assertAlmostEqual(summary["kinds"]["native wait/syscall/signal"], 500)
        self.assertAlmostEqual(summary["kinds"]["native work"], 100)


class TraceTests(unittest.TestCase):
    def test_critical_path(self):
        events = [
            {"name": "thread_name", "ph": "M", "tid": 1, "args": {"name": "queue worker 0x0"}},
            {"name": "present", "ph": "i", "tid": 2, "ts": 0},
            {"name": "present", "ph": "i", "tid": 2, "ts": 10000},
            {"name": "present", "ph": "i", "tid": 2, "ts": 20000},
            {"name": "Driver.Packet", "ph": "X", "tid": 1, "ts": 0, "dur": 8000},
            {"name": "Driver.Packet.memory_wait", "ph": "X", "tid": 1, "ts": 100, "dur": 6000},
            {"name": "Driver.Packet", "ph": "X", "tid": 1, "ts": 10000, "dur": 8000},
            {"name": "Driver.Packet.memory_wait", "ph": "X", "tid": 1, "ts": 10100, "dur": 2000},
        ]
        path = pp.frame_critical_path(trace_digest.Trace({"traceEvents": events}))
        self.assertEqual(path["frames"], 2)
        self.assertAlmostEqual(path["frame_ms"], 10.0)
        worker = path["threads"][0]
        self.assertEqual(worker["thread"], "queue worker 0x0")
        self.assertAlmostEqual(worker["busy_ms"], 8.0)
        self.assertAlmostEqual(worker["wait_ms"], 4.0)
        self.assertEqual(worker["waits"][0][0], "Driver.Packet.memory_wait")


class ReportTests(unittest.TestCase):
    def test_report_from_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "T-x"
            Path(f"{base}.log").write_text(GPUTIME_NEW + "\n" + FEX_SAMPLE)
            Path(f"{base}-sample.txt").write_text(SAMPLE)
            Path(f"{base}-pm.txt").write_text(POWERMETRICS)
            level = {"instructions": 0, "cycles": 0, "time": 0, "energy_nj": 0}
            Path(f"{base}-threadcounts.json").write_text(json.dumps({
                "ns_per_tick": 1.0, "t0": 0.0, "t1": 1.0, "before": {"5": {"name": "", "levels": [level, level]}},
                "after": {"5": {"name": "w", "levels": [dict(level, time=500_000), level]}}}))
            result = profile_title.build_report(Path(directory), "T", "x")
            self.assertIn("sample", result)
            self.assertEqual(result["threads_cpu"]["rows"][0]["name"], "w")
            text = Path(f"{base}-profile.md").read_text()
            for heading in ("## Clocks", "## CPU per thread", "## Where the CPU goes", "## GPU time", "## Guest time"):
                self.assertIn(heading, text)

    def test_arguments(self):
        self.assertEqual(profile_title.parse_window("60:20"), (60.0, 20.0))
        with self.assertRaises(Exception):
            profile_title.parse_window("5:0")
        self.assertEqual(profile_title.parse_env(["A=1", "B=x=y"]), {"A": "1", "B": "x=y"})
        with self.assertRaises(SystemExit):
            profile_title.parse_env(["nope"])


if __name__ == "__main__":
    unittest.main()
