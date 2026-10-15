#!/usr/bin/env python3
"""Whole-system profile of one headless title run: where the CPU and GPU time goes.

    profile_title.py run <ID> --tag T [--seconds 120] [--window 60:20] [--sample-seconds 10]
                     [--fex-sample 2] [--env K=V ...] [--build DIR] [--log-dir DIR]
    profile_title.py report <log-dir> <ID> <TAG>      re-parse the files of an earlier run

One `title_run.py run` (headless, GPU lock, build lock, load gate) with the driver trace on over the window
(APS5_TRACE_SECONDS) and `[gputime]` on (APS5_PROFILE_GPU=1). During the window, from outside the process:
- `sample <pid>`: per-thread call graphs, classified into guest JIT code, FEX (compile, signals, thunks,
  dispatcher), HLE libraries by module, the AGC driver, Metal, waits (GPU, lock, guest sleep) and spins;
- per-thread CPU time on P and E cores with cycles and instructions (libproc PROC_PIDTHREADCOUNTS), read at
  the window's start and end;
- `sudo -n powermetrics` (cpu_power, gpu_power, thermal; 500 ms): cluster and GPU frequency and residency,
  power, thermal pressure.
After the run: the trace's per-thread phases per frame and what each frame waits on, the `[gputime]` lines
and, with --fex-sample, the guest sampler's split of guest time between translated and native code.

Writes <log-dir>/<ID>-<TAG>-profile.md and -profile.json beside title_run's own files. No per-title code.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import profile_parse as pp  # noqa: E402
import title_run  # noqa: E402

DEFAULT_ENV = {"APS5_PROFILE_GPU": "1", "APS5_PRESENT_SIZE": "2560x1440"}


def parse_window(text):
    start, _, length = text.partition(":")
    start, length = float(start), float(length)
    if start < 0 or length <= 0:
        raise argparse.ArgumentTypeError(f"--window wants START:LENGTH seconds, got {text!r}")
    return start, length


def parse_env(pairs):
    env = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--env wants KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def stem(log_dir, title, tag):
    return Path(log_dir) / f"{title}-{tag}"


def find_title_pid(root_pid, stop, runner=None):
    while not stop.is_set() and (runner is None or runner.poll() is None):
        table = title_run.parse_ps_table(subprocess.run(["ps", "-A", "-o", "pid=,ppid=,comm="],
                                                        capture_output=True, text=True).stdout)
        pid = title_run.find_descendant(table, root_pid, "aps5-fex")
        if pid is not None:
            return pid
        stop.wait(0.25)
    return None


def collect(pid, base, window, sample_seconds, use_powermetrics, started):
    """At the window's start: counters, `sample`, powermetrics; counters again at its end."""
    start, length = window
    delay = started + start - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    counts = {"ns_per_tick": pp.mach_timebase_ns()}
    try:
        counts["before"] = {str(k): v for k, v in pp.read_thread_counts(pid).items()}
    except OSError as error:
        counts["error"] = str(error)
    counts["t0"] = time.monotonic()
    processes = []
    if use_powermetrics and shutil.which("powermetrics"):
        handle = open(f"{base}-pm.txt", "w")
        processes.append(subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,thermal", "-i", "500",
             "-n", str(max(int(length * 2), 1))], stdout=handle, stderr=subprocess.STDOUT))
    sample_at = start + max((length - sample_seconds) / 2.0, 0.0)
    time.sleep(max(started + sample_at - time.monotonic(), 0.0))
    processes.append(subprocess.Popen(["sample", str(pid), str(int(sample_seconds)), "1", "-mayDie",
                                       "-file", f"{base}-sample.txt"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(max(started + start + length - time.monotonic(), 0.0))
    try:
        counts["after"] = {str(k): v for k, v in pp.read_thread_counts(pid).items()}
    except OSError as error:
        counts["error"] = str(error)
    counts["t1"] = time.monotonic()
    Path(f"{base}-threadcounts.json").write_text(json.dumps(counts))
    for process in processes:
        try:
            process.wait(timeout=length + sample_seconds + 60)
        except subprocess.TimeoutExpired:
            process.kill()


def cmd_run(args):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = stem(log_dir, args.title, args.tag)
    window = args.window
    if args.seconds < window[0] + window[1] + 5:
        raise SystemExit(f"--seconds {args.seconds} ends before the window {window[0]:g}:{window[1]:g} closes")
    env = dict(os.environ)
    env.update(DEFAULT_ENV)
    trace_dir = Path(f"{base}-trace")
    trace_dir.mkdir(exist_ok=True)
    env.update({"APS5_TRACE_SECONDS": f"{int(window[0])}:{int(window[1])}", "APS5_TRACE_DIR": str(trace_dir)})
    if args.fex_sample:
        env.update({"APS5_FEX_SAMPLE": str(args.fex_sample), "APS5_FEX_SAMPLE_REPORT": str(int(window[1]))})
    env.update(parse_env(args.env))
    command = [sys.executable, str(HERE / "title_run.py"), "run", args.title, "--tag", args.tag,
               "--seconds", str(args.seconds), "--log-dir", str(log_dir), "--build", args.build,
               "--frames", str(args.frames), "--max-load", str(args.max_load)]
    if args.validation:
        command.append("--validation")
    print("profile:", " ".join(command), flush=True)
    runner = subprocess.Popen(command, env=env)
    stop = threading.Event()
    pid = find_title_pid(runner.pid, stop, runner)
    started = time.monotonic()
    print(f"profile: title pid {pid}", flush=True)
    if pid is None:
        rc = runner.wait()
        print(f"profile: the title never started (runner rc {rc})")
        return 1
    collector = threading.Thread(target=collect, args=(pid, base, window, args.sample_seconds,
                                                       not args.no_powermetrics, started))
    collector.start()
    rc = runner.wait()
    stop.set()
    collector.join()
    report = build_report(log_dir, args.title, args.tag)
    print(f"profile: wrote {base}-profile.md (runner rc {rc})")
    return 0 if report else 1


# ---------------------------------------------------------------------------------------------------------

def build_report(log_dir, title, tag):
    base = stem(log_dir, title, tag)
    result = {"title": title, "tag": tag}
    summary_path = Path(f"{base}-summary.json")
    if summary_path.exists():
        result["run"] = json.loads(summary_path.read_text())
    log_text = Path(f"{base}.log").read_text(errors="replace") if Path(f"{base}.log").exists() else ""

    trace_dir = Path(f"{base}-trace")
    traces = sorted(trace_dir.glob("aps5-trace-*.json.gz")) if trace_dir.exists() else []
    if traces:
        import trace_digest
        trace = trace_digest.load_trace(traces[-1])
        result["trace_summary"] = trace_digest.load_summary(None, traces[-1])
        result["critical_path"] = pp.frame_critical_path(trace)

    counts_path = Path(f"{base}-threadcounts.json")
    if counts_path.exists():
        counts = json.loads(counts_path.read_text())
        if "before" in counts and "after" in counts:
            seconds = counts["t1"] - counts["t0"]
            rows = pp.diff_thread_counts(counts["before"], counts["after"], counts["ns_per_tick"], seconds)
            result["threads_cpu"] = {"seconds": seconds, "rows": rows}

    sample_path = Path(f"{base}-sample.txt")
    if sample_path.exists():
        sampled = pp.summarize_sample(pp.parse_sample(sample_path.read_text(errors="replace")))
        result["sample"] = {
            "samples_per_thread": sampled["samples_per_thread"],
            "totals": dict(sampled["totals"].most_common()),
            "details": {category: details.most_common(8) for category, details in sampled["details"].items()},
            "threads": sampled["threads"][:24],
        }

    pm_path = Path(f"{base}-pm.txt")
    if pm_path.exists():
        result["power"] = pp.summarize_powermetrics(pp.parse_powermetrics(pm_path.read_text(errors="replace")))

    gputime = pp.parse_gputime(log_text)
    if gputime:
        result["gputime"] = gputime
    windows = pp.parse_fex_sample(log_text)
    if windows:
        result["fex_sample"] = pp.summarize_fex_sample(windows)

    Path(f"{base}-profile.json").write_text(json.dumps(result, indent=1, default=str))
    Path(f"{base}-profile.md").write_text(markdown(result))
    return result


def _fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(result):
    out = [f"# Profile {result['title']} {result['tag']}", ""]
    run = result.get("run", {})
    if run:
        out.append(f"Run: verdict {run.get('verdict')}, {run.get('wall_s')} s, validation {run.get('validation')}.")
    summary = result.get("trace_summary") or {}
    if summary:
        frame = summary.get("frame_ms", {})
        out += ["", "## Frames", "",
                f"{summary.get('presents')} presents over {summary.get('window', {}).get('spec')}: "
                f"p50 {_fmt(frame.get('p50'))} ms, p99 {_fmt(frame.get('p99'))} ms, max {_fmt(frame.get('max'))} ms "
                f"at {summary.get('resolution')}. Worker busy %: {summary.get('worker_busy_pct')}."]
    power = result.get("power") or {}
    if power:
        out += ["", "## Clocks, residency, power, thermals (powermetrics medians)", "",
                f"- P cores {_fmt(power.get('p_active_mhz'), 0)} MHz when active, cluster {_fmt(power.get('p_active_pct'))}% "
                f"(per-core mean {_fmt(power.get('p_core_active_pct'))}%); E cores {_fmt(power.get('e_active_mhz'), 0)} MHz, "
                f"cluster {_fmt(power.get('e_active_pct'))}% (per-core {_fmt(power.get('e_core_active_pct'))}%).",
                f"- GPU {_fmt(power.get('gpu_mhz'), 0)} MHz, active {_fmt(power.get('gpu_active_pct'))}%, highest "
                f"state used {power.get('gpu_peak_state_mhz')} MHz (top state 1338 MHz on the M3).",
                f"- Power: CPU {_fmt(power.get('cpu_mw'), 0)} mW, GPU {_fmt(power.get('gpu_mw'), 0)} mW. "
                f"Thermal pressure: {power.get('pressure')} over {power.get('samples')} samples."]
    threads = result.get("threads_cpu")
    if threads:
        rows = threads["rows"]
        total = sum(row["cpu_ms"] for row in rows)
        p_total = sum(row["p_ms"] for row in rows)
        out += ["", f"## CPU per thread over {threads['seconds']:.1f} s (P/E time from the kernel's per-thread counters)", "",
                f"Process: {total / 1000.0 / threads['seconds']:.2f} cores, {100.0 * p_total / total if total else 0:.0f}% "
                "of it on P cores.", "",
                "| thread | name | CPU % of a core | P share | P GHz | P IPC | E GHz | E IPC |", "|---|---|---|---|---|---|---|---|"]
        for row in rows[:16]:
            out.append(f"| {row['tid']} | {row['name'] or '-'} | {row['cpu_pct']:.1f} | {100 * row['p_share']:.0f}% | "
                       f"{row['p_ghz']:.2f} | {row['p_ipc']:.2f} | {row['e_ghz']:.2f} | {row['e_ipc']:.2f} |")
    sample = result.get("sample")
    if sample:
        total = sum(sample["totals"].values())
        busy = sum(count for category, count in sample["totals"].items() if not category.startswith(("wait", "idle")))
        out += ["", "## Where the CPU goes (`sample`, all threads)", "",
                f"{sample['samples_per_thread']} samples per thread; {busy} busy samples of {total}.", "",
                "Cores is samples / samples per thread (CPU cores kept busy, or threads parked, on average); ms per "
                "frame is that times the mean frame time.", "",
                "| category | samples | cores | ms/frame | % of busy | top items |", "|---|---|---|---|---|---|"]
        frame_ms = (result.get("critical_path") or {}).get("frame_ms")
        per_thread = sample["samples_per_thread"] or 1
        for category, count in sample["totals"].items():
            share = f"{100.0 * count / busy:.1f}" if busy and not category.startswith(("wait", "idle")) else "-"
            cores = count / per_thread
            per_frame = f"{cores * frame_ms:.1f}" if frame_ms else "-"
            top = "; ".join(f"{detail} {n}" for detail, n in sample["details"].get(category, [])[:4])
            out.append(f"| {category} | {count} | {cores:.2f} | {per_frame} | {share} | {top} |")
        out += ["", "### Busiest threads", ""]
        for thread in sample["threads"][:10]:
            if thread["busy"] == 0:
                continue
            top = "; ".join(f"{category}: {detail} {n}" for category, detail, n in thread["top"][:5])
            out.append(f"- {thread['key']} ({thread['role']}): busy {thread['busy']}/{thread['total']}. {top}")
    path = result.get("critical_path")
    if path and path.get("frames"):
        out += ["", f"## What each frame waits on (trace, {path['frames']} frames, {path['frame_ms']:.1f} ms each)", "",
                "| trace thread | in phases ms/frame | of which waits | work | largest waits |", "|---|---|---|---|---|"]
        for thread in path["threads"]:
            waits = "; ".join(f"{name} {ms:.1f}" for name, ms in thread["waits"][:4])
            out.append(f"| {thread['thread']} | {thread['busy_ms']:.1f} | {thread['wait_ms']:.1f} | "
                       f"{thread['work_ms']:.1f} | {waits} |")
    gputime = result.get("gputime")
    if gputime:
        out += ["", "## GPU time (`[gputime]`, per 10 s)", ""]
        for entry in gputime[-3:]:
            classes = ", ".join(f"{name} {value['ms']:.0f} ms/{value['mib']:.0f} MiB"
                                for name, value in sorted(entry["classes"].items(), key=lambda item: -item[1]["ms"])[:6])
            out.append(f"- {entry['batches']} batches, batch span {entry['batch_ms']} ms, timed ranges "
                       f"{entry['ranges_ms']} ms, GPU program time {entry['program_ms']} ms (untimed {entry['untimed_ms']} ms; {entry['invalid']} invalid pairs); {classes}")
    fex = result.get("fex_sample")
    if fex:
        kinds = fex["kinds"]
        total = sum(kinds.values()) or 1.0
        out += ["", "## Guest time (guest sampler, top RIPs)", "",
                f"{fex['samples']} samples over {fex['threads']} guest threads; of the top RIPs: " +
                ", ".join(f"{kind} {100.0 * value / total:.0f}%" for kind, value in sorted(kinds.items(), key=lambda kv: -kv[1])), ""]
        for module, offset, kind, host, samples in fex["hot"][:8]:
            out.append(f"- {module}+0x{offset:x} {kind} {host}: {samples:.0f}")
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("title")
    run.add_argument("--tag", required=True)
    run.add_argument("--seconds", type=int, default=120, help="title_run's hard cap")
    run.add_argument("--window", type=parse_window, default=(60.0, 20.0), help="START:LENGTH seconds")
    run.add_argument("--sample-seconds", type=float, default=10.0)
    run.add_argument("--fex-sample", type=int, default=0, help="APS5_FEX_SAMPLE period in ms (0: off)")
    run.add_argument("--env", action="append", help="KEY=VALUE for the title (repeatable)")
    run.add_argument("--build", default=str(REPO / "build-metal"))
    run.add_argument("--log-dir", default=str(title_run.work_dir() / "logs" / "profile"))
    run.add_argument("--frames", type=int, default=0, help="APS5_DUMP_FRAMES (0: no dumps)")
    run.add_argument("--max-load", type=float, default=8.0)
    run.add_argument("--validation", action="store_true")
    run.add_argument("--no-powermetrics", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("log_dir")
    report.add_argument("title")
    report.add_argument("tag")
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    result = build_report(Path(args.log_dir), args.title, args.tag)
    print(Path(f"{stem(args.log_dir, args.title, args.tag)}-profile.md").read_text())
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
