#!/usr/bin/env python3
"""Runs GPU test groups one test at a time under the build-directory lock and reports pass counts per group.

Usage (tools/mac/gpu_suite.sh is the entry point):
    tools/mac/gpu_suite.sh [options] SELECTOR...

SELECTOR: a group G0..G7 (tools/mac/gpu_groups.txt), `report` (G1..G4), `all` (G0..G7), `re:<regex>` (test names
matched with re.search) or an exact test name. Selections are unions.

GPU safety (CLAUDE.md, MISSION §5): every ctest invocation runs -j1 with a per-test timeout inside
tools/mac/buildlock.py <build>, which is also the machine-wide GPU mutex; the tests gpu_groups.HOLD matches
(none now: agc_driver_lds_atomics64 was released on 2026-10-13 after supervised passes) and the quarantined ones (tools/mac/
gpu_quarantine.txt: tests that panicked or hung the machine; without a readable quarantine file nothing runs) are
never run, whatever is selected; a batch
holds the lock for at most --batch-seconds of worst-case test time (sum of the tests' timeouts), so other agents can
build between batches.

Safety guards (DECISIONS D19, after the 2026-10-12 kernel panic); gpu_suite refuses to run (exit 3) when:
    - the source tree that configured the build has uncommitted or untracked changes under core/ or specifics/
      (after waiting up to --busy-wait seconds for it to become clean);
    - `ninja -n` reports work for a selected test executable, a canary or libSceAgcDriver.prx (stale binaries);
    - the 1-minute load average is above 6 or a compiler/linker (clang, ld, swift-frontend, metal, ...) runs,
      after waiting up to --busy-wait seconds (checked again before every test);
    - the canaries (tools/mac/gpu_canaries.txt: known-good tests, run first in every invocation) fail, or the
      canary list is missing or empty.
    Every test runs in its own ctest process, holding the build lock, with a timeout of at most 30 s (a TIMEOUT
    property above that is capped; the process group is killed at the cap), followed by a 2 s pause. A test
    executable or the prx that changes during the run makes the run fail. summary.json records HEAD, the prx's
    sha256 and the guard results.

Output directory (--out, default .mac-work/logs/gpu-suite/<time>-<selectors> of the main checkout):
    junit.xml     every testcase of every batch in one ctest-style JUnit file (tools/check_skips.py reads it)
    summary.txt   per-group counts, failures with their first error line, skips with their reason and allowlist
                  verdict, the MSL emission histogram (opcode or reason) and the most common first errors
    summary.json  the same per test, for scripts
    batches/      the JUnit file and console log of each ctest invocation
Exit status: 0 when every selected test passed or skipped with an allowlisted reason, 1 otherwise, 2 on usage errors.
"""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))
import buildlock  # noqa: E402
import gpu_groups  # noqa: E402
import check_skips  # noqa: E402

LOCK = HERE / "buildlock.py"
CANARIES = HERE / "gpu_canaries.txt"
MAX_TIMEOUT = 30
LEASE_SLACK = 600.0
LEASE_CAP = 5400.0
PAUSE_SECONDS = 2.0
LOAD_LIMIT = 6.0
# The supervised rerun of a held test (--rerun-held) waits for a quieter machine.
HELD_LOAD_LIMIT = 3.0
load_limit = LOAD_LIMIT
GUARDED_PATHS = ("core", "specifics")
PRX = "core/libs/libs/unpatched/libSceAgcDriver.prx"
# Compilers and linkers whose work loads the machine (and, for metal, the GPU compiler service).
BUSY_PROCESSES = frozenset({"clang", "clang++", "c++", "cc", "ld", "ld64", "ld-classic", "swift-frontend", "swiftc",
                            "metal", "metallib", "air-lld"})
SELECTOR_GROUPS = {"report": ("G1", "G2", "G3", "G4"), "all": gpu_groups.GROUPS}
EMISSION = re.compile(r"MSL emission failed: (.*)")
OPCODE = re.compile(r"\bopcode=(\S+)")
EMISSION_REASON = re.compile(r"\breason=(.*)")
ERROR_LINE = re.compile(r"error|fail|exception|what\(\)|terminate|abort|assert|mismatch|expected|unsupported|not implemented|"
                        r"cannot|could not|invalid|timeout|timed out|signal|segmentation|bus error|nan", re.IGNORECASE)
NOISE = re.compile(r"^\s*$|^\[gpu\] skip reports|^RecompileRequest:|^[A-Za-z0-9+/=]{120,}$")
VOLATILE = re.compile(r"0x[0-9a-fA-F]+|\b\d{3,}\b|hash=\S+")


def default_logs():
    """.mac-work/logs/gpu-suite of the main checkout (a worktree shares it through the common git directory)."""
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        root = Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        root = REPO
    return root / ".mac-work" / "logs" / "gpu-suite"


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_state(source=REPO):
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    head = run("rev-parse", "--short", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    dirty = len([line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()])
    return {"head": head, "branch": branch, "modified_files": dirty}


class Refused(Exception):
    """A D19 guard refused the run."""


def load_canaries(path=CANARIES):
    """The canary test names (one per line, # comments); a missing or empty list refuses the run (fail closed)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise Refused(f"the canary list {path} cannot be read ({error})") from error
    names = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    names = [name for name in names if name]
    if not names:
        raise Refused(f"the canary list {path} is empty")
    return names


def source_tree(build):
    """CMAKE_HOME_DIRECTORY of the build (the tree whose sources produced its binaries)."""
    try:
        for line in (Path(build) / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:"):
                return Path(line.split("=", 1)[1].strip())
    except OSError:
        pass
    raise Refused(f"cannot tell which source tree configured {build} (no CMAKE_HOME_DIRECTORY in CMakeCache.txt)")


def dirty_paths(source):
    """Uncommitted or untracked (not ignored) paths under core/ and specifics/ of the source tree."""
    try:
        result = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all", "--", *GUARDED_PATHS],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise Refused(f"cannot read the git status of {source} ({error})") from error
    return [line[3:] for line in result.stdout.splitlines() if line.strip()]


GLOB_CHECK = re.compile(r"VerifyGlobs\.cmake_force|cmake\.verify_globs")


def stale_targets(build, targets):
    """Edges `ninja -n` would run for the targets (empty when everything is up to date).

    With CONFIGURE_DEPENDS globs a dry run always lists "Re-checking globbed directories" and "Re-running CMake"
    (the glob check is forced and cannot run in a dry run); the CMake step counts as stale only when `-d explain`
    gives a reason other than that forced check (an edited CMakeLists.txt, a changed cache). New or removed source
    files are caught by the clean-tree check."""
    if not targets:
        return []
    try:
        result = subprocess.run(["ninja", "-C", str(build), "-n", "-d", "explain", *targets], capture_output=True, text=True)
    except OSError as error:
        raise Refused(f"cannot run ninja to check for stale binaries ({error})") from error
    if result.returncode != 0:
        raise Refused("ninja -n failed: " + ((result.stderr.strip() or result.stdout.strip()).splitlines() or ["(no output)"])[-1][:300])
    lines = (result.stdout + "\n" + result.stderr).splitlines()
    reasons = [line for line in lines if line.startswith("ninja explain:") and not GLOB_CHECK.search(line)]
    steps = [line for line in result.stdout.splitlines() if re.match(r"^\[\d+/\d+\]", line)]
    stale = []
    for step in steps:
        if "Re-checking globbed directories" in step:
            continue
        if "Re-running CMake" in step and not any("build.ninja" in reason for reason in reasons):
            continue
        stale.append(step)
    return stale


def load_average():
    """The 1-minute load average (sysctl vm.loadavg)."""
    try:
        text = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True, check=True).stdout
        return float(text.replace("{", " ").split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return os.getloadavg()[0]


def busy_processes():
    """Running compilers and linkers (by executable name)."""
    try:
        text = subprocess.run(["ps", "-A", "-o", "comm="], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise Refused(f"cannot list processes ({error})") from error
    return sorted({os.path.basename(line.strip()) for line in text.splitlines()} & BUSY_PROCESSES)


def machine_busy():
    """Why the machine is too busy for a GPU test, or None."""
    load = load_average()
    if load > load_limit:
        return f"1-minute load average {load:.2f} > {load_limit:g}"
    running = busy_processes()
    if running:
        return "compiler/linker running: " + " ".join(running)
    return None


def wait_until_idle(max_wait, poll=10.0):
    """Waits (bounded) until machine_busy() is None; raises Refused with the last reason otherwise."""
    deadline = time.monotonic() + max_wait
    while True:
        reason = machine_busy()
        if reason is None:
            return
        if time.monotonic() >= deadline:
            raise Refused(f"machine busy after waiting {max_wait:g} s: {reason}")
        print(f"waiting: {reason}", flush=True)
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())) or poll)


def wait_until_clean(source, max_wait, poll=10.0):
    """dirty_paths(source), re-checked for up to max_wait seconds while it is non-empty (someone may be committing).
    Waiting is safe: nothing is run yet, and the result is the last check, so a still-dirty tree is refused by the caller."""
    deadline = time.monotonic() + max_wait
    while True:
        dirty = dirty_paths(source)
        if not dirty or time.monotonic() >= deadline:
            return dirty
        print(f"waiting for a clean tree ({len(dirty)} uncommitted: {' '.join(dirty[:5])})", flush=True)
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())) or poll)


def sha256(path):
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def effective_timeout(property_value, default_timeout):
    """The per-test timeout gpu_suite enforces: the TIMEOUT property or the default, capped at MAX_TIMEOUT."""
    try:
        value = float(property_value) if property_value else float(default_timeout)
    except ValueError:
        value = float(default_timeout)
    return max(1, min(int(value), MAX_TIMEOUT))


def select(names, groups, selectors):
    """The tests the selectors name, in group then name order; unknown selectors raise ValueError."""
    chosen = set()
    for selector in selectors:
        if selector in SELECTOR_GROUPS or selector in gpu_groups.GROUPS:
            wanted = SELECTOR_GROUPS.get(selector, (selector,))
            chosen.update(name for name in names if groups.get(name) in wanted)
        elif selector.startswith("re:"):
            pattern = re.compile(selector[3:])
            chosen.update(name for name in names if pattern.search(name))
        elif selector in names:
            chosen.add(selector)
        else:
            raise ValueError(f"unknown selector {selector!r}: not a group (G0..G7, report, all), re:<regex> or a registered test")
    order = {group: index for index, group in enumerate(gpu_groups.GROUPS)}
    return sorted(chosen, key=lambda name: (order.get(groups.get(name), len(order)), name))


def batches(names, timeouts, default_timeout, budget):
    """Consecutive batches whose summed worst-case test time stays within the budget (at least one test each)."""
    current, spent = [], 0.0
    for name in names:
        cost = float(timeouts.get(name) or default_timeout)
        if current and spent + cost > budget:
            yield current
            current, spent = [], 0.0
        current.append(name)
        spent += cost
    if current:
        yield current


def first_error(output):
    """The line that best explains a failure: an MSL emission failure, else the first error-like line, else the last line."""
    lines = [line.strip() for line in output.splitlines() if not NOISE.search(line.strip())]
    for line in lines:
        if EMISSION.search(line):
            return line[:400]
    for line in lines:
        if ERROR_LINE.search(line):
            return line[:400]
    return lines[-1][:400] if lines else ""


def emission_key(output):
    found = EMISSION.search(output)
    if not found:
        return None
    detail = found.group(1)
    opcode = OPCODE.search(detail)
    if opcode:
        return "opcode=" + opcode.group(1)
    why = EMISSION_REASON.search(detail)
    return "reason=" + (why.group(1) if why else detail).strip()[:120]


def normalized(line):
    return VOLATILE.sub("#", line)[:160]


def read_cases(path):
    """[(name, status, time, message, output, element)] from one ctest JUnit file."""
    cases = []
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("name")
        if not name:
            continue
        status = case.get("status", "")
        detail = case.find("failure")
        if detail is None:
            detail = case.find("skipped")
        message = detail.get("message", "") if detail is not None else ""
        output = "\n".join(case.findtext(channel) or "" for channel in ("system-out", "system-err"))
        cases.append((name, status, float(case.get("time") or 0.0), message, output, case))
    return cases


def outcome(status, message, output):
    if status == "run":
        return "passed"
    if status == "fail":
        return "timeout" if "timeout" in message.lower() else "failed"
    if check_skips.NOT_BUILT.search(message) or check_skips.NOT_BUILT.search(output):
        return "not-built"
    if status in ("notrun", "disabled"):
        return "skipped"
    return "failed"


def merge_junit(cases, path, started):
    suite = ET.Element("testsuite", {
        "name": "gpu_suite",
        "tests": str(len(cases)),
        "failures": str(sum(1 for case in cases if case[1] == "fail")),
        "disabled": str(sum(1 for case in cases if case[1] == "disabled")),
        "skipped": str(sum(1 for case in cases if case[1] == "notrun")),
        "hostname": "",
        "time": f"{sum(case[2] for case in cases):.3f}",
        "timestamp": started,
    })
    for case in cases:
        suite.append(case[5])
    ET.ElementTree(suite).write(path, encoding="UTF-8", xml_declaration=True)


def executable_mtimes(build, names, registered):
    stamps = {}
    for name in names:
        exe = gpu_groups.executable(registered.get(name, {}).get("command", []))
        try:
            stamps[exe] = os.stat(exe).st_mtime
        except OSError:
            stamps[exe] = None
    return stamps


def run_test(build, name, timeout, junit, log, environment):
    """One test in its own ctest process under the build lock; the process group is killed at the timeout cap."""
    command = ["ctest", "--test-dir", str(build), "-j1", "--timeout", str(timeout), "-R", "^" + re.escape(name) + "$",
               "--output-junit", str(junit), "--no-tests=error"]
    lock_path = Path(build) / ".aps5-build.lock"
    # GPU lock first (machine-wide, per test), then the per-build lock: the same order everywhere.
    with buildlock.gpu_lock(label=f"gpu_suite test {name} ({build})", out=sys.stdout), \
            open(lock_path, "a+") as lock, open(log, "w", encoding="utf-8") as stream:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"waiting for the build lock {lock_path}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        stream.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        stream.flush()
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(REPO), env=environment,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout + 5)
            killed = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            killed = True
            stream.write(f"\ngpu_suite: killed the test's process group at the {timeout} s cap\n")
    return code, time.monotonic() - started, killed


def build_executables(build, names, registered, log, environment):
    targets = sorted({os.path.basename(gpu_groups.executable(registered[name]["command"])) for name in names if name in registered})
    targets = [target for target in targets if target and target not in ("cmake", "cmake.exe")]
    command = [sys.executable, "-I", str(LOCK), str(build), "--", "ninja", "-C", str(build), "-k", "0", *targets]
    with open(log, "w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        stream.flush()
        return subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(REPO), env=environment)


def summarize(results, groups, held, selectors, meta, allow_errors, allow_warnings):
    lines = []
    lines.append(f"gpu_suite {' '.join(selectors)}  {meta['started']} -> {meta['finished']}  ({meta['seconds']:.0f} s, {meta['batches']} tests run, one ctest process each)")
    lines.append(f"git {meta['git']['branch']} {meta['git']['head']} ({meta['git']['modified_files']} modified tracked files), build {meta['build']}")
    if meta["environment"]:
        lines.append("environment: " + " ".join(f"{key}={value}" for key, value in sorted(meta["environment"].items())))
    lines.append(f"prx {meta['prx']['path']} sha256 {meta['prx']['sha256']}")
    if meta.get("canaries"):
        lines.append("canaries: " + " ".join(f"{name}={status}" for name, status in meta["canaries"].items()))
    if meta.get("stopped"):
        lines.append(f"STOPPED: {meta['stopped']}; {len(meta.get('not_run', []))} selected test(s) not run")
    if meta["rebuilt"]:
        lines.append(f"FAILED: {len(meta['rebuilt'])} binary(ies) changed while the suite ran (another agent built): "
                     + " ".join(sorted(os.path.basename(path) for path in meta["rebuilt"])[:10]))
    lines.append("")
    kinds = ("passed", "failed", "timeout", "skipped", "not-built")
    lines.append(f"{'group':6} {'tests':>5} " + " ".join(f"{kind:>9}" for kind in kinds) + f" {'held':>5}")
    totals = collections.Counter()
    for group in gpu_groups.GROUPS + ("G?",):
        members = [result for result in results if (groups.get(result["test"]) or "G?") == group]
        held_here = [name for name in held if (groups.get(name) or "G?") == group]
        if not members and not held_here:
            continue
        counts = collections.Counter(result["outcome"] for result in members)
        totals.update(counts)
        totals["tests"] += len(members)
        totals["held"] += len(held_here)
        lines.append(f"{group:6} {len(members):5d} " + " ".join(f"{counts[kind]:9d}" for kind in kinds) + f" {len(held_here):5d}")
    lines.append(f"{'total':6} {totals['tests']:5d} " + " ".join(f"{totals[kind]:9d}" for kind in kinds) + f" {totals['held']:5d}")
    head = len(lines)
    lines.append("")
    if held:
        lines.append("held back (never run by gpu_suite): " + " ".join(sorted(held)))
        quarantined = meta.get("quarantine", {})
        lines += [f"  quarantined {name}: {quarantined[name]}" for name in sorted(held) if name in quarantined]
        lines.append("")
    broken = [result for result in results if result["outcome"] in ("failed", "timeout", "not-built")]
    if broken:
        lines.append(f"failures ({len(broken)}):")
        for result in broken:
            lines.append(f"  {result['test']} [{result['group']}] {result['outcome']}: {result['first_error']}")
        lines.append("")
    skipped = [result for result in results if result["outcome"] == "skipped"]
    if skipped:
        lines.append(f"skips ({len(skipped)}):")
        for result in skipped:
            lines.append(f"  {result['test']} [{result['group']}]: {result['reason']}")
        lines.append("")
    if allow_errors or allow_warnings:
        lines.append(f"skip check (tools/check_skips.py --job macos): {len(allow_errors)} error(s), {len(allow_warnings)} warning(s)")
        lines += [f"  error: {error}" for error in allow_errors]
        lines += [f"  warning: {warning}" for warning in allow_warnings]
        lines.append("")
    emission = collections.Counter(result["emission"] for result in results if result["emission"])
    if emission:
        lines.append("MSL emission failures by opcode or reason (tests whose output has one):")
        lines += [f"  {count:4d}  {key}" for key, count in emission.most_common()]
        lines.append("")
    first = collections.Counter(normalized(result["first_error"]) for result in broken if result["first_error"] and not result["emission"])
    if first:
        lines.append("other first errors (numbers and addresses folded to #):")
        lines += [f"  {count:4d}  {line}" for line, count in first.most_common(25)]
        lines.append("")
    return "\n".join(lines[:head]), "\n".join(lines)


def lease_seconds(order, registered, default_timeout, pause):
    """Worst-case duration of a run (every test at its timeout, plus the pauses) plus 10 minutes, capped at 90 minutes."""
    total = sum(effective_timeout(registered[name]["properties"].get("TIMEOUT"), default_timeout) + pause for name in order)
    return min(total + LEASE_SLACK, LEASE_CAP)


held_leases = []


def release_leases():
    while held_leases:
        buildlock.release_lease(held_leases.pop())


def main():
    """Runs the suite; the build lease it takes is released on every exit path, including exceptions and SIGTERM."""
    previous = signal.signal(signal.SIGTERM, lambda number, frame: sys.exit(128 + number))
    try:
        return run_suite()
    finally:
        release_leases()
        signal.signal(signal.SIGTERM, previous)


def run_suite():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 2)[2])
    parser.add_argument("selectors", nargs="+", metavar="SELECTOR")
    parser.add_argument("--build", default="build-metal", help="build directory, relative to the repository root (default: build-metal)")
    parser.add_argument("--out", type=Path, help="output directory (default: .mac-work/logs/gpu-suite/<time>-<selectors> of the main checkout)")
    parser.add_argument("--timeout", type=int, default=MAX_TIMEOUT, help=f"per-test timeout in seconds for tests without a TIMEOUT property (default and maximum: {MAX_TIMEOUT})")
    parser.add_argument("--batch-seconds", type=float, default=240.0, help="worst-case test seconds per batch in --dry-run listings (default: 240); every test runs in its own ctest process")
    parser.add_argument("--canaries", type=Path, default=CANARIES, help="canary list run first (default: tools/mac/gpu_canaries.txt)")
    parser.add_argument("--busy-wait", type=float, default=300.0, help="seconds to wait for load/compilers to go away and for the source tree to become clean before refusing (default: 300)")
    parser.add_argument("--quarantine", type=Path, default=gpu_groups.QUARANTINE, help="quarantine list (default: tools/mac/gpu_quarantine.txt); missing or malformed: nothing runs")
    parser.add_argument("--rerun-held", metavar="NAME", help="the supervised run of ONE test gpu_groups.HOLD holds back (HOLD is empty now; agc_driver_lds_atomics64 was released on 2026-10-13; use with the coordinator's agreement): NAME must be the only selected test, needs --validation, runs alone after the canaries with a 1-minute load limit of 3; the test stays held for normal runs")
    parser.add_argument("--rerun-quarantined", action="store_true", help="the supervised rerun of ONE quarantined test (D19.4/6): needs exactly one selected test, --validation, and runs it alone after the canaries; the test stays quarantined until the list is edited with this run as evidence")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS, help=f"seconds between tests (default and minimum: {PAUSE_SECONDS:g})")
    parser.add_argument("--validation", action="store_true", help="run with MTL_SHADER_VALIDATION=1 (reported to stderr) and MTL_DEBUG_LAYER=1")
    parser.add_argument("--dump-requests", type=Path, metavar="DIR", help="export APS5_DUMP_REQUESTS=DIR (relative to the repository root): every recompile request is written there, the corpus of agc_metal_msl_corpus (build-metal/msl-corpus)")
    parser.add_argument("--build-first", action="store_true", help="build the selected tests' executables (ninja -k 0, under the lock) before running")
    parser.add_argument("--no-lease", action="store_true", help="do not freeze the build directory for the run (short single-test runs); other agents' builds may then void it")
    parser.add_argument("--allowlist", type=Path, default=check_skips.MACOS_ALLOWLIST, help="skip allowlist (default: tools/mac/m2_skip_allowlist.txt)")
    parser.add_argument("--dry-run", action="store_true", help="print the selection and the batches and run nothing")
    args = parser.parse_args()
    if not 1 <= args.timeout <= MAX_TIMEOUT:
        parser.error(f"--timeout must be between 1 and {MAX_TIMEOUT} s (DECISIONS D19)")
    if args.pause < PAUSE_SECONDS:
        parser.error(f"--pause must be at least {PAUSE_SECONDS:g} s (DECISIONS D19)")
    if args.busy_wait < 0:
        parser.error("--busy-wait must not be negative")

    build = Path(args.build)
    if not build.is_absolute():
        build = REPO / build
    if not (build / "CTestTestfile.cmake").exists():
        parser.error(f"{build} is not a configured build directory")
    registered = gpu_groups.registered_tests(build)
    gpu_tests = {name for name, test in registered.items() if "gpu" in test["properties"].get("LABELS", "").split(";")}
    mapped = gpu_groups.load_map()
    groups = {name: mapped[name] for name in gpu_tests if name in mapped}
    unmapped = sorted(gpu_tests - set(groups))
    if unmapped:
        groups.update({name: group for name, group in gpu_groups.classify(build).items() if name in unmapped})
    try:
        chosen = select(sorted(registered), groups, args.selectors)
    except (ValueError, re.error) as error:
        parser.error(str(error))
    try:
        quarantine = gpu_groups.load_quarantine(args.quarantine)
    except (OSError, ValueError) as error:
        print(f"refusing to run GPU tests: the quarantine list cannot be read ({error})", file=sys.stderr)
        return 2
    rerun = []
    if args.rerun_quarantined:
        if len(chosen) != 1 or chosen[0] not in quarantine:
            parser.error("--rerun-quarantined needs exactly one selected test, and it must be quarantined")
        if not args.validation:
            parser.error("--rerun-quarantined needs --validation (DECISIONS D19.4)")
        rerun = list(chosen)
        print(f"supervised rerun of quarantined test {chosen[0]} ({quarantine[chosen[0]]})")
    rerun_held = []
    if args.rerun_held:
        if chosen != [args.rerun_held] or not gpu_groups.HOLD.search(args.rerun_held):
            parser.error("--rerun-held NAME needs NAME to be the only selected test, and gpu_groups.HOLD must hold it back")
        if args.rerun_quarantined or args.rerun_held in quarantine:
            parser.error("--rerun-held cannot run a quarantined test (use --rerun-quarantined)")
        if not args.validation:
            parser.error("--rerun-held needs --validation (DECISIONS D19.4)")
        global load_limit
        load_limit = HELD_LOAD_LIMIT
        rerun_held = list(chosen)
        print(f"supervised run of held test {args.rerun_held} (load limit {HELD_LOAD_LIMIT:g})")
    for name in chosen:
        if name in quarantine and name not in rerun:
            print(f"refusing to run quarantined test {name}: {quarantine[name]}")
    held = [name for name in chosen if (gpu_groups.HOLD.search(name) and name not in rerun_held) or (name in quarantine and name not in rerun)]
    runnable = [name for name in chosen if name not in held]
    timeouts = {name: registered[name]["properties"].get("TIMEOUT") for name in runnable}
    planned = list(batches(runnable, timeouts, args.timeout, args.batch_seconds))
    if args.dry_run:
        print(f"{len(runnable)} test(s) in {len(planned)} batch(es), {len(held)} held back: {' '.join(held)}")
        for index, batch in enumerate(planned, 1):
            print(f"batch {index}: {' '.join(batch)}")
        if unmapped:
            print(f"not in {gpu_groups.MAP.name} (classified now): {' '.join(unmapped)}")
        return 0
    if not runnable:
        print("nothing to run" + (f": {len(held)} selected test(s) are held back" if held else ""))
        return 1

    try:
        canaries = [name for name in load_canaries(args.canaries)]
    except Refused as error:
        print(f"refusing to run GPU tests: {error}", file=sys.stderr)
        return 3
    missing_canaries = [name for name in canaries if name not in registered or name in quarantine or gpu_groups.HOLD.search(name)]
    if missing_canaries:
        print(f"refusing to run GPU tests: canaries not registered or held back: {' '.join(missing_canaries)}", file=sys.stderr)
        return 3

    stamp = utc_stamp()
    out = args.out or default_logs() / f"{stamp}-{'_'.join(re.sub(r'[^A-Za-z0-9_.-]', '', selector) for selector in args.selectors)[:60]}"
    out = out if out.is_absolute() else Path.cwd() / out
    (out / "batches").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    # Presentation (G5) tests run headless: windowed runs happen only outside gpu_suite, with coordinator approval.
    extra = {"SDL_VIDEODRIVER": "offscreen"}
    if args.validation:
        extra.update({"MTL_SHADER_VALIDATION": "1", "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1", "MTL_DEBUG_LAYER": "1"})
    if args.dump_requests:
        dump = args.dump_requests if args.dump_requests.is_absolute() else REPO / args.dump_requests
        dump.mkdir(parents=True, exist_ok=True)
        extra["APS5_DUMP_REQUESTS"] = str(dump)
    environment.update(extra)
    environment["SDL_VIDEODRIVER"] = "offscreen"

    started_wall = datetime.datetime.now().isoformat(timespec="seconds")
    started = time.monotonic()
    if args.build_first:
        code = build_executables(build, sorted(set(runnable) | set(canaries)), registered, out / "build.log", environment)
        print(f"build: exit {code} (log {out / 'build.log'})")

    # D19 guards: consistent committed binaries, an idle machine, then the canaries.
    order = [name for name in canaries if name not in runnable] + [name for name in canaries if name in runnable] \
        + [name for name in runnable if name not in canaries]
    if not args.no_lease:
        # Builders (tools/mac/buildlock.py) wait while this lease exists; the GPU mutex is still taken per test.
        lease = buildlock.take_lease(build, lease_seconds(order, registered, args.timeout, args.pause), run=out.name)
        held_leases.append(build)
        print(f"build lease {lease} held until the run ends", flush=True)
    guards = {}
    refusal = None
    try:
        source = source_tree(build)
        guards["source"] = str(source)
        dirty = wait_until_clean(source, args.busy_wait)
        guards["dirty"] = dirty
        if dirty:
            raise Refused(f"{source} has uncommitted changes under {'/ '.join(GUARDED_PATHS)}/: " + " ".join(dirty[:10])
                          + (" ..." if len(dirty) > 10 else "") + " (commit them, or run from a clean worktree)")
        targets = sorted({os.path.relpath(gpu_groups.executable(registered[name]["command"]), build) for name in order
                          if gpu_groups.executable(registered[name]["command"])})
        if (build / "build.ninja").exists() and ("build " + PRX + ":") in (build / "build.ninja").read_text(encoding="utf-8", errors="replace"):
            targets.append(PRX)
        stale = stale_targets(build, targets)
        guards["stale"] = stale
        if stale:
            raise Refused(f"{len(stale)} build step(s) pending for the GPU test binaries or the prx (build first): " + stale[0][:200])
        wait_until_idle(args.busy_wait)
    except Refused as error:
        refusal = str(error)
    git_start = git_state(guards.get("source", REPO))
    prx_path = build / PRX
    prx_before = sha256(prx_path)
    before = executable_mtimes(build, order, registered)

    cases = []
    canary_results = {}
    stopped = refusal
    for index, name in enumerate(order, 1):
        if stopped:
            break
        if index > 1:
            time.sleep(args.pause)
            try:
                wait_until_idle(args.busy_wait)
            except Refused as error:
                stopped = str(error)
                break
        timeout = effective_timeout(registered[name]["properties"].get("TIMEOUT"), args.timeout)
        junit = out / "batches" / f"test-{index:03d}.xml"
        code, seconds, killed = run_test(build, name, timeout, junit, out / "batches" / f"test-{index:03d}.log", environment)
        found = [case for case in (read_cases(junit) if junit.exists() else []) if case[0] == name]
        if not found:
            message = f"killed at the {timeout} s cap (timeout)" if killed else f"ctest did not report the test (exit {code})"
            element = ET.Element("testcase", {"name": name, "classname": name, "time": f"{seconds:.3f}", "status": "fail"})
            ET.SubElement(element, "failure", {"message": message})
            found = [(name, "fail", seconds, message, "", element)]
        case = found[0]
        status = outcome(case[1], case[3], case[4])
        print(f"[{index}/{len(order)}] {name}: {status} ({seconds:.1f} s){' [canary]' if name in canaries else ''}", flush=True)
        if name in canaries:
            canary_results[name] = status
            if status != "passed":
                stopped = f"canary {name} {status}: {first_error(case[4]) or case[3]}"
        if name in runnable:
            cases.append(case)
    after = executable_mtimes(build, order, registered)
    prx_after = sha256(prx_path)
    finished_wall = datetime.datetime.now().isoformat(timespec="seconds")
    if stopped:
        print(f"STOPPED: {stopped}", file=sys.stderr)

    merge_junit(cases, out / "junit.xml", started_wall)
    results = []
    for name, status, seconds, message, output, _ in cases:
        result = outcome(status, message, output)
        results.append({
            "test": name, "group": groups.get(name) or "G?", "outcome": result, "seconds": round(seconds, 3),
            "first_error": first_error(output) if result in ("failed", "timeout", "not-built") else "",
            "emission": emission_key(output) if result in ("failed", "timeout") else None,
            "reason": check_skips.macos_reason("skipped, " + message + "\n" + output) if result == "skipped" else "",
        })
    skipped_found = [(name, "\n".join(["skipped, " + message, output])) for name, status, _, message, output, _ in cases
                     if status in ("notrun", "disabled") and not check_skips.NOT_BUILT.search(message)]
    try:
        allow_errors, allow_warnings = check_skips.macos_check(
            skipped_found, {case[0]: case[1] for case in cases}, check_skips.allowlist(args.allowlist),
            check_skips.TECHNICAL_DEBT.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        allow_errors, allow_warnings = [f"cannot check skips: {error}"], []
    rebuilt = [path for path, stamp_before in before.items() if after.get(path) != stamp_before and stamp_before is not None]
    if prx_before != prx_after:
        rebuilt.append(str(prx_path))
    not_run = [name for name in runnable if name not in {case[0] for case in cases}]
    meta = {
        "started": started_wall, "finished": finished_wall, "seconds": time.monotonic() - started, "batches": len(cases),
        "git": git_start, "build": os.path.relpath(build, REPO), "environment": extra,
        "rebuilt": rebuilt, "prx": {"path": PRX, "sha256": prx_before}, "guards": guards, "canaries": canary_results,
        "stopped": stopped, "not_run": not_run, "timeout_cap": MAX_TIMEOUT, "pause": args.pause, "quarantine_rerun": rerun, "held_rerun": rerun_held, "quarantine": {name: quarantine[name] for name in held if name in quarantine},
    }
    table, summary = summarize(results, groups, held, args.selectors, meta, allow_errors, allow_warnings)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"meta": meta, "selectors": args.selectors, "held": held, "results": results,
                                                  "skip_errors": allow_errors, "skip_warnings": allow_warnings}, indent=1) + "\n", encoding="utf-8")
    print(table)
    if allow_errors:
        print(f"skip check: {len(allow_errors)} error(s), see the summary")
    print(f"\nfull summary: {out / 'summary.txt'}\njunit: {out / 'junit.xml'}")
    if rebuilt:
        print(f"FAILED: {len(rebuilt)} binary(ies) changed while the suite ran; the results are not trustworthy", file=sys.stderr)
    if refusal:
        return 3
    bad = any(result["outcome"] in ("failed", "timeout", "not-built") for result in results) or allow_errors or stopped or rebuilt
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
