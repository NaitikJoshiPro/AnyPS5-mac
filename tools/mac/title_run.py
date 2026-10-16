#!/usr/bin/env python3
"""Relink, boot and soak a homebrew title on the native arm64 + Metal build, with scripted pad input.

    title_run.py prepare <ID>                 relink .mac-work/titles/<ID>/eboot.bin into .mac-work/runs/m3/<ID>/
    title_run.py run <ID> --tag T [options]   one run under buildlock (holds the GPU mutex), logged

A run writes everything to <log-dir>/<ID>-<TAG>*:
    <ID>-<TAG>.log             host stdout/stderr
    <ID>-<TAG>-download0/      copy of the guest's <run>/download0/ (titles freopen their stdout/stderr there)
    <ID>-<TAG>-frames/         frame_*.bmp dumps (APS5_DUMP_FRAMES / APS5_DUMP_FRAMES_EVERY)
    <ID>-<TAG>-samples.csv     RSS and CPU of the title process every --sample-every seconds
    <ID>-<TAG>-native.txt      native proof (--native-proof-at): `file` on every loaded image, vmmap x86/Rosetta
                               scan, lsof, the P_TRANSLATED process flag
    <ID>-<TAG>-pads.log        pad commands sent
    <ID>-<TAG>-summary.json    exit status, wall time, counts

Pad scripts (--pads FILE) are plain text, one command per line, `#` comments:
    wait <seconds>
    press <button> [hold_ms] [lx ly rx ry]   button as tools/pad_input.py; `release` sends sticks only.
                                             The runner waits hold_ms after each press.
    loop                                     the commands after it repeat until the run ends
Pad input reaches the title through APS5_PAD_INPUT_FILE (libScePad PadState.cpp).

Headless (default): SDL_VIDEODRIVER=offscreen. --windowed opens the Cocoa window. --audio-file captures the
title's audio with SDL's disk driver instead of playing it; `title_run.py audio <file>` then reports the
per-window signal level. Only one title run at a time (GPU safety, CLAUDE.md): the run waits for the 1-minute
load average to drop below --max-load and for the build-metal lock.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))
import buildlock  # noqa: E402  (machine-wide GPU lock)
import pad_input  # noqa: E402  (BUTTONS table)

P_TRANSLATED = 0x00020000  # sys/proc.h: process runs under Rosetta translation


# ---- pad scripts ---------------------------------------------------------------------------------------------

class PadScriptError(ValueError):
    pass


def parse_pad_script(text):
    """(intro, loop): lists of ("wait", seconds) / ("press", button, hold_ms, sticks or None)."""
    intro, loop, seen_loop = [], [], False
    target = intro
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        words = line.split()
        try:
            if words[0] == "loop" and len(words) == 1:
                if seen_loop:
                    raise PadScriptError("second loop")
                seen_loop, target = True, loop
            elif words[0] == "wait" and len(words) == 2:
                seconds = float(words[1])
                if not 0 <= seconds <= 3600:
                    raise PadScriptError("wait out of range")
                target.append(("wait", seconds))
            elif words[0] == "press" and len(words) in (2, 3, 7):
                button = words[1]
                if button not in pad_input.BUTTONS:
                    raise PadScriptError(f"unknown button {button}")
                hold = int(words[2]) if len(words) >= 3 else 300
                if not 1 <= hold <= 5000:
                    raise PadScriptError("hold_ms must be 1..5000")
                sticks = None
                if len(words) == 7:
                    sticks = tuple(int(w) for w in words[3:])
                    if not all(0 <= axis <= 255 for axis in sticks):
                        raise PadScriptError("stick axes must be 0..255")
                target.append(("press", button, hold, sticks))
            else:
                raise PadScriptError(f"cannot parse {line!r}")
        except ValueError as error:
            raise PadScriptError(f"pad script line {number}: {error}") from None
    if seen_loop and not any(step[0] == "press" or step[1] > 0 for step in loop):
        raise PadScriptError("loop has no time-consuming command")
    return intro, loop


def iter_pad_steps(intro, loop):
    yield from intro
    while loop:
        yield from loop


def pad_line(button, hold_ms, sticks, now_ns):
    """The APS5_PAD_INPUT_FILE line pad_input.py writes."""
    tail = "".join(f" {axis}" for axis in sticks) if sticks else ""
    return f"{now_ns} {pad_input.BUTTONS[button]} {hold_ms}{tail}\n"


def write_pad(path, button, hold_ms, sticks):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(pad_line(button, hold_ms, sticks, time.monotonic_ns()))
    os.replace(temporary, path)


# ---- process discovery and sampling -------------------------------------------------------------------------

def parse_ps_table(text):
    """`ps -A -o pid=,ppid=,comm=` → {pid: (ppid, comm)}."""
    table = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2].strip())
    return table


def find_descendant(table, root, name):
    """The first descendant of root whose command basename is name, breadth first, or None."""
    frontier = [root]
    while frontier:
        nxt = []
        for parent in frontier:
            for pid in sorted(p for p, (pp, _) in table.items() if pp == parent):
                if os.path.basename(table[pid][1]) == name:
                    return pid
                nxt.append(pid)
        frontier = nxt
    return None


def parse_cputime(text):
    """ps `time` ([dd-]hh:mm:ss.cc or mm:ss.cc) → seconds."""
    text = text.strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [float(p) for p in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return days * 86400 + seconds


def parse_sample(line):
    """`ps -o rss=,time=,flags= -p PID` → (rss_kib, cpu_seconds, flags) or None."""
    parts = line.split()
    if len(parts) != 3:
        return None
    return int(parts[0]), parse_cputime(parts[1]), int(parts[2], 16)


def is_translated(flags):
    return bool(flags & P_TRANSLATED)


def leak_slope_mib_per_min(samples):
    """Least-squares RSS slope over (seconds, rss_kib) samples, in MiB per minute (0 with < 2 samples)."""
    if len(samples) < 2:
        return 0.0
    n = len(samples)
    mx = sum(t for t, _ in samples) / n
    my = sum(r for _, r in samples) / n
    den = sum((t - mx) ** 2 for t, _ in samples)
    if den == 0:
        return 0.0
    slope = sum((t - mx) * (r - my) for t, r in samples) / den
    return slope * 60 / 1024


# ---- native proof --------------------------------------------------------------------------------------------

X86_VMMAP = re.compile(r"rosetta|oahd|libRosetta|x86_64|x86-64|\bi386\b", re.IGNORECASE)
# The translator itself (/Library/Apple/usr/libexec/oah/libRosettaRuntime) and translated code types. Every native
# arm64 process on macOS 27 also maps the 28 KiB shared-cache stub /usr/lib/libRosetta.dylib (Finder, Dock and
# python3 do too, measured 2026-10-13), so a match is only evidence when a native reference process lacks it.
ROSETTA_RUNTIME = re.compile(r"/oah/|libRosettaRuntime|RosettaLinux|\(translated\)", re.IGNORECASE)


def vmmap_path(line):
    match = re.search(r"SM=\S+\s+(/.*\S)\s*$", line)
    return match.group(1) if match else None


def split_findings(findings, baseline_text):
    """(new, baseline): findings whose mapped path a native reference process also maps go to baseline."""
    base_paths = {vmmap_path(line) for line in baseline_text.splitlines()} - {None}
    new, baseline = [], []
    for line in findings:
        path = vmmap_path(line)
        (baseline if path is not None and path in base_paths and not ROSETTA_RUNTIME.search(line) else new).append(line)
    return new, baseline


def vmmap_images(text):
    """Absolute paths of the Mach-O images (__TEXT regions) in vmmap output."""
    images = []
    for line in text.splitlines():
        if line.lstrip().startswith("__TEXT"):
            path = vmmap_path(line)
            if path:
                if path not in images:
                    images.append(path)
    return images


def vmmap_x86_findings(text):
    return [line.strip() for line in text.splitlines() if X86_VMMAP.search(line)]


def file_verdict(text):
    """`file` output for a Mach-O → "arm64" (arm64 only), "universal" (arm64 and x86 slices: dyld maps the arm64
    one in an arm64 process), "x86" (no arm64 slice) or "other"."""
    x86, arm = bool(re.search(r"x86_64|i386", text)), bool(re.search(r"arm64", text))
    if x86 and arm:
        return "universal"
    if x86:
        return "x86"
    return "arm64" if arm else "other"


def vmmap_code_type(text):
    match = re.search(r"^Code Type:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def native_proof(pid, out_path):
    lines = [f"# native proof for pid {pid} at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"]
    ok = True
    flags = subprocess.run(["ps", "-o", "flags=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    translated = is_translated(int(flags, 16)) if flags else None
    lines.append(f"ps flags=0x{flags} P_TRANSLATED(0x20000)={'set' if translated else 'clear'}")
    ok &= translated is False
    vm = subprocess.run(["vmmap", str(pid)], capture_output=True, text=True).stdout
    code_type = vmmap_code_type(vm)
    lines.append(f"vmmap Code Type: {code_type}")
    ok &= code_type is not None and "ARM64" in code_type.upper()
    findings = vmmap_x86_findings(vm)
    reference = subprocess.run(["vmmap", str(os.getpid())], capture_output=True, text=True).stdout
    new, baseline = split_findings(findings, reference)
    lines.append(f"vmmap lines matching {X86_VMMAP.pattern}: {len(findings)}; also mapped by the native reference "
                 f"process (this python3, Code Type {vmmap_code_type(reference)}): {len(baseline)}; title-only: {len(new)}")
    lines += ["  [baseline] " + f for f in baseline[:50]]
    lines += ["  [TITLE-ONLY] " + f for f in new[:50]]
    runtime = [line for line in vm.splitlines() if ROSETTA_RUNTIME.search(line)]
    lines.append(f"Rosetta runtime (oah/libRosettaRuntime) mappings: {len(runtime)}")
    ok &= not new and not runtime
    images = vmmap_images(vm)
    lines.append(f"loaded Mach-O images (__TEXT): {len(images)}")
    verdicts = {}
    for image in images:
        if os.path.exists(image):
            text = subprocess.run(["file", "-b", image], capture_output=True, text=True).stdout.strip()
        else:
            text = "(not on disk: an image of the arm64e dyld shared cache this ARM64 process maps)"
        verdict = file_verdict(text)
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        ok &= verdict not in ("x86", "other") or not os.path.exists(image)
        lines.append(f"  [{verdict}] {image}: {text}")
    lines.append(f"file verdicts: {verdicts}")
    lsof = subprocess.run(["lsof", "-p", str(pid)], capture_output=True, text=True).stdout
    hits = [line for line in lsof.splitlines() if X86_VMMAP.search(line)]
    lines.append(f"lsof: {len(lsof.splitlines())} lines, {len(hits)} matching rosetta/x86")
    lines += ["  " + h for h in hits[:50]]
    ok &= not hits
    lines.append("--- lsof txt (mapped files) ---")
    lines += [line for line in lsof.splitlines() if " txt " in line]
    lines.append(f"VERDICT: {'NATIVE' if ok else 'NOT PROVEN NATIVE'}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    return ok


# ---- audio ---------------------------------------------------------------------------------------------------

def pcm_windows(data, sample_format, channels, rate, window_seconds):
    """RMS (0..1) per window of interleaved PCM; sample_format "s16" or "f32"."""
    width = 2 if sample_format == "s16" else 4
    code = "<h" if sample_format == "s16" else "<f"
    scale = 32768.0 if sample_format == "s16" else 1.0
    frame = width * channels
    per_window = max(1, int(rate * window_seconds)) * frame
    out = []
    for start in range(0, len(data) - frame + 1, per_window):
        chunk = data[start:start + per_window]
        chunk = chunk[:len(chunk) - len(chunk) % width]
        count = len(chunk) // width
        if count == 0:
            break
        total = sum(v * v for (v,) in struct.iter_unpack(code, chunk))
        out.append(math.sqrt(total / count) / scale)
    return out


# ---- run -----------------------------------------------------------------------------------------------------

def work_dir():
    return Path(os.environ.get("APS5_MAC_WORK", REPO.parent.parent if REPO.parent.name == "wt" else REPO / ".mac-work"))


def build_env(base, *, windowed, frames, every, validation, pad_file, audio_file):
    env = dict(base)
    if not windowed:
        env["SDL_VIDEODRIVER"] = "offscreen"
    env["APS5_DUMP_FRAMES"] = str(frames)
    env["APS5_DUMP_FRAMES_EVERY"] = str(every)
    if validation:
        env["MTL_DEBUG_LAYER"] = "1"
        env["MTL_SHADER_VALIDATION"] = "1"
    if pad_file:
        env["APS5_PAD_INPUT_FILE"] = str(pad_file)
    if audio_file:
        env["SDL_AUDIODRIVER"] = "disk"
        env["SDL_DISKAUDIOFILE"] = str(audio_file)
    return env


# The cap's SIGALRM kills aps5-fex; buildlock.py exits with sys.exit(-14) → 242 (142 if run directly).
ALARM_CODES = (142, 242, -14)


def run_verdict(rc):
    """"alive" (killed by the cap), "exited" (clean exit) or "failed"."""
    if rc in ALARM_CODES:
        return "alive"
    return "exited" if rc == 0 else "failed"


def load_ok(load1, limit):
    return load1 < limit


class LibsLinkError(RuntimeError):
    pass


def ensure_libs_link(run_dir, build, out=sys.stdout):
    """Makes run_dir/libs point at build's libraries (the runner loads the PS5 libraries from there).

    `prepare` sets the link; a later `run --build X` must not silently use another build's libraries, so the link is
    re-pointed at X's when it points elsewhere (or dangles). Refuses (LibsLinkError) when X has no libraries or the
    run dir holds a real `libs` directory instead of a link. Returns the previous target, or None when unchanged."""
    run_dir, build = Path(run_dir), Path(build).resolve()
    wanted = build / "core/libs/libs"
    if not wanted.is_dir():
        raise LibsLinkError(f"{wanted} does not exist: build the libraries of {build} first")
    link = run_dir / "libs"
    if link.is_symlink():
        current = Path(os.readlink(link))
        if not current.is_absolute():
            current = run_dir / current
        if current.exists() and current.resolve() == wanted.resolve():
            return None
        link.unlink()
        link.symlink_to(wanted)
        print(f"libs: {link} pointed at {current}; re-pointed at {wanted}", file=out, flush=True)
        return current
    if link.exists():
        raise LibsLinkError(f"{link} is not a symlink: cannot tell which build's libraries it holds; "
                            f"run `title_run.py prepare {run_dir.name} --build {build}`")
    link.symlink_to(wanted)
    print(f"libs: {link} was missing; pointed at {wanted}", file=out, flush=True)
    return None


def cmd_prepare(args):
    work = work_dir()
    build = Path(args.build).resolve()
    src = work / "titles" / args.id
    out = work / "runs" / "m3" / args.id
    logs = Path(args.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with open(logs / f"{args.id}-relink.log", "w") as log:
        rc = subprocess.call([str(build / "core/relinker/relinker"), "--registry", str(src / "eboot.bin"),
                              str(out / "app.elf")], stdout=log, stderr=subprocess.STDOUT)
    print(f"relink rc={rc}")
    if rc != 0:
        return rc
    (out / "libs").symlink_to(build / "core/libs/libs")
    app0 = out / "app0"
    app0.mkdir(exist_ok=True)
    for entry in sorted(src.iterdir()):
        if entry.name in ("sce_module",):
            continue
        link = app0 / entry.name
        if not link.exists():
            link.symlink_to(entry)
    print(f"run dir {out}")
    return 0


def cmd_run(args):
    """One title run; the whole run holds the machine-wide GPU lock.

    Lock order (see buildlock.py): wait for the build's gpu_suite lease first, WITHOUT the GPU lock, then take the
    GPU lock; the buildlock child does not wait for the lease again (the suite may be queued on the GPU lock)."""
    buildlock.wait_for_lease(Path(args.build).resolve(), out=sys.stdout)
    with buildlock.gpu_lock(label=f"title_run {args.id} --tag {args.tag}", out=sys.stdout):
        return run_locked(args)


def run_locked(args):
    work = work_dir()
    build = Path(args.build).resolve()
    run_dir = work / "runs" / "m3" / args.id
    if not (run_dir / "app.elf").exists():
        print(f"{run_dir}/app.elf missing: run `title_run.py prepare {args.id}` first", file=sys.stderr)
        return 2
    try:
        ensure_libs_link(run_dir, build)
    except LibsLinkError as error:
        print(f"refusing to run: {error}", file=sys.stderr)
        return 2
    logs = Path(args.log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    stem = logs / f"{args.id}-{args.tag}"
    intro, loop = parse_pad_script(Path(args.pads).read_text()) if args.pads else ([], [])

    while not load_ok(os.getloadavg()[0], args.max_load):
        print(f"waiting: load {os.getloadavg()[0]:.1f} >= {args.max_load}", flush=True)
        time.sleep(30)

    for old in run_dir.glob("frame_*.bmp"):
        old.unlink()
    pad_file = run_dir / "pad-input" if args.pads else None
    if pad_file and pad_file.exists():
        pad_file.unlink()
    audio_file = Path(args.audio_file).resolve() if args.audio_file else None
    env = build_env(os.environ, windowed=args.windowed, frames=args.frames, every=args.every,
                    validation=args.validation, pad_file=pad_file, audio_file=audio_file)
    fex = build / "core/fex/aps5-fex"
    command = [sys.executable, str(HERE / "buildlock.py"), str(build), "--",
               "perl", "-e", "alarm shift; exec @ARGV", str(args.seconds), str(fex), "./app.elf"]
    start = time.monotonic()
    log = open(f"{stem}.log", "w")
    proc = subprocess.Popen(command, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    stop = threading.Event()
    state = {"pid": None, "translated": None, "native": None, "pads": 0}

    def find_pid():
        while not stop.is_set() and state["pid"] is None:
            table = parse_ps_table(subprocess.run(["ps", "-A", "-o", "pid=,ppid=,comm="],
                                                  capture_output=True, text=True).stdout)
            state["pid"] = find_descendant(table, proc.pid, "aps5-fex")
            if state["pid"] is None:
                stop.wait(0.5)

    def sampler():
        find_pid()
        if state["pid"] is None:
            return
        with open(f"{stem}-samples.csv", "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["t_s", "rss_kib", "cpu_s", "cpu_pct_since_last", "translated"])
            last = None
            while not stop.is_set():
                result = subprocess.run(["ps", "-o", "rss=,time=,flags=", "-p", str(state["pid"])],
                                        capture_output=True, text=True).stdout.strip()
                sample = parse_sample(result) if result else None
                if sample is None:
                    break
                now = time.monotonic() - start
                rss, cpu, flags = sample
                pct = "" if last is None else f"{100 * (cpu - last[1]) / max(now - last[0], 1e-6):.0f}"
                state["translated"] = is_translated(flags)
                writer.writerow([f"{now:.0f}", rss, f"{cpu:.1f}", pct, int(state["translated"])])
                out.flush()
                last = (now, cpu)
                stop.wait(args.sample_every)

    def pads():
        find_pid()
        if state["pid"] is None:
            return
        with open(f"{stem}-pads.log", "w") as out:
            for step in iter_pad_steps(intro, loop):
                if stop.is_set():
                    return
                if step[0] == "wait":
                    stop.wait(step[1])
                    continue
                _, button, hold, sticks = step
                write_pad(pad_file, button, hold, sticks)
                state["pads"] += 1
                out.write(f"{time.monotonic() - start:7.1f} {button} {hold} {sticks or ''}\n")
                out.flush()
                stop.wait(hold / 1000)

    def prover():
        find_pid()
        if state["pid"] is not None and not stop.wait(args.native_proof_at) and proc.poll() is None:
            state["native"] = native_proof(state["pid"], f"{stem}-native.txt")

    threads = [threading.Thread(target=sampler, daemon=True)]
    if intro or loop:
        threads.append(threading.Thread(target=pads, daemon=True))
    if args.native_proof_at is not None:
        threads.append(threading.Thread(target=prover, daemon=True))
    for thread in threads:
        thread.start()
    try:
        rc = proc.wait(timeout=args.seconds + args.lock_wait + 30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        rc = proc.wait()
    stop.set()
    for thread in threads:
        thread.join(timeout=60)
    log.close()
    wall = time.monotonic() - start

    frames_dir = Path(f"{stem}-frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames = sorted(run_dir.glob("frame_*.bmp"))
    if frames:
        frames_dir.mkdir()
        for frame in frames:
            shutil.move(str(frame), frames_dir / frame.name)
    download = run_dir / "download0"
    mirror = Path(f"{stem}-download0")
    if download.is_dir():
        if mirror.exists():
            shutil.rmtree(mirror)
        shutil.copytree(download, mirror, ignore=shutil.ignore_patterns("*.raw"))
    summary = {"id": args.id, "tag": args.tag, "rc": rc, "verdict": run_verdict(rc), "wall_s": round(wall, 1),
               "seconds_cap": args.seconds, "pid": state["pid"], "translated": state["translated"],
               "native_proof": state["native"], "pad_commands": state["pads"], "frames": len(frames),
               "windowed": args.windowed, "validation": args.validation}
    Path(f"{stem}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0 if run_verdict(rc) != "failed" else 1


def cmd_audio(args):
    data = Path(args.file).read_bytes()
    windows = pcm_windows(data, args.format, args.channels, args.rate, args.window)
    silent = sum(1 for v in windows if v < args.silence)
    seconds = len(data) / ((2 if args.format == "s16" else 4) * args.channels * args.rate)
    print(f"{seconds:.1f} s of audio, {len(windows)} windows of {args.window} s, {silent} below {args.silence} RMS")
    for index, value in enumerate(windows):
        print(f"{index * args.window:7.0f} s  rms {value:.4f}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    default_logs = str(work_dir() / "logs" / "m3-qa")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("id")
    prepare.add_argument("--build", default=str(REPO / "build-metal"))
    prepare.add_argument("--log-dir", default=default_logs)
    run = sub.add_parser("run")
    run.add_argument("id")
    run.add_argument("--tag", required=True)
    run.add_argument("--build", default=str(REPO / "build-metal"))
    run.add_argument("--log-dir", default=default_logs)
    run.add_argument("--seconds", type=int, default=60, help="hard cap (SIGALRM)")
    run.add_argument("--lock-wait", type=int, default=600, help="extra seconds allowed for the build lock")
    run.add_argument("--pads", help="pad script")
    run.add_argument("--frames", type=int, default=6, help="APS5_DUMP_FRAMES")
    run.add_argument("--every", type=int, default=30, help="APS5_DUMP_FRAMES_EVERY (flips)")
    run.add_argument("--validation", action="store_true", help="Metal API + shader validation")
    run.add_argument("--windowed", action="store_true", help="open the Cocoa window (not offscreen)")
    run.add_argument("--audio-file", help="capture audio with SDL's disk driver into this file")
    run.add_argument("--sample-every", type=float, default=30.0)
    run.add_argument("--native-proof-at", type=float, help="seconds after start to record the native proof")
    run.add_argument("--max-load", type=float, default=6.0)
    audio = sub.add_parser("audio")
    audio.add_argument("file")
    audio.add_argument("--format", choices=("s16", "f32"), default="s16")
    audio.add_argument("--channels", type=int, default=2)
    audio.add_argument("--rate", type=int, default=48000)
    audio.add_argument("--window", type=float, default=10.0)
    audio.add_argument("--silence", type=float, default=1e-4)
    args = parser.parse_args(argv)
    return {"prepare": cmd_prepare, "run": cmd_run, "audio": cmd_audio}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
