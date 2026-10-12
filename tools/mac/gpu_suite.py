#!/usr/bin/env python3
"""Runs GPU test groups one test at a time under the build-directory lock and reports pass counts per group.

Usage (tools/mac/gpu_suite.sh is the entry point):
    tools/mac/gpu_suite.sh [options] SELECTOR...

SELECTOR: a group G0..G7 (tools/mac/gpu_groups.txt), `report` (G1..G4), `all` (G0..G7), `re:<regex>` (test names
matched with re.search) or an exact test name. Selections are unions.

GPU safety (CLAUDE.md, MISSION §5): every ctest invocation runs -j1 with a per-test timeout inside
tools/mac/buildlock.py <build>, which is also the machine-wide GPU mutex; the tests gpu_groups.HOLD matches
(agc_driver_lds_atomics64, flip, present, window and video-out tests) are never run, whatever is selected; a batch
holds the lock for at most --batch-seconds of worst-case test time (sum of the tests' timeouts), so other agents can
build between batches.

Output directory (--out, default .mac-work/logs/gpu-suite/<time>-<selectors>):
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
import json
import os
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "tools"))
import gpu_groups  # noqa: E402
import check_skips  # noqa: E402

LOCK = HERE / "buildlock.py"
SELECTOR_GROUPS = {"report": ("G1", "G2", "G3", "G4"), "all": gpu_groups.GROUPS}
EMISSION = re.compile(r"MSL emission failed: (.*)")
OPCODE = re.compile(r"\bopcode=(\S+)")
EMISSION_REASON = re.compile(r"\breason=(.*)")
ERROR_LINE = re.compile(r"error|fail|exception|what\(\)|terminate|abort|assert|mismatch|expected|unsupported|not implemented|"
                        r"cannot|could not|invalid|timeout|timed out|signal|segmentation|bus error|nan", re.IGNORECASE)
NOISE = re.compile(r"^\s*$|^\[gpu\] skip reports|^RecompileRequest:|^[A-Za-z0-9+/=]{120,}$")
VOLATILE = re.compile(r"0x[0-9a-fA-F]+|\b\d{3,}\b|hash=\S+")


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_state():
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    head = run("rev-parse", "--short", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    dirty = len([line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()])
    return {"head": head, "branch": branch, "modified_files": dirty}


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


def run_batch(build, names, default_timeout, junit, log, environment):
    pattern = "^(" + "|".join(re.escape(name) for name in names) + ")$"
    command = [sys.executable, "-I", str(LOCK), str(build), "--", "ctest", "--test-dir", str(build), "-j1",
               "--timeout", str(default_timeout), "-R", pattern, "--output-junit", str(junit), "--no-tests=error"]
    started = time.monotonic()
    with open(log, "w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        stream.flush()
        code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(REPO), env=environment)
    return code, time.monotonic() - started


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
    lines.append(f"gpu_suite {' '.join(selectors)}  {meta['started']} -> {meta['finished']}  ({meta['seconds']:.0f} s, {meta['batches']} ctest invocations)")
    lines.append(f"git {meta['git']['branch']} {meta['git']['head']} ({meta['git']['modified_files']} modified tracked files), build {meta['build']}")
    if meta["environment"]:
        lines.append("environment: " + " ".join(f"{key}={value}" for key, value in sorted(meta["environment"].items())))
    if meta["rebuilt"]:
        lines.append(f"warning: {len(meta['rebuilt'])} test executable(s) changed while the suite ran (another agent built): "
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 2)[2])
    parser.add_argument("selectors", nargs="+", metavar="SELECTOR")
    parser.add_argument("--build", default="build-metal", help="build directory, relative to the repository root (default: build-metal)")
    parser.add_argument("--out", type=Path, help="output directory (default: .mac-work/logs/gpu-suite/<time>-<selectors>)")
    parser.add_argument("--timeout", type=int, default=30, help="per-test timeout in seconds for tests without a TIMEOUT property (default: 30)")
    parser.add_argument("--batch-seconds", type=float, default=240.0, help="worst-case test seconds per lock hold (default: 240)")
    parser.add_argument("--validation", action="store_true", help="run with MTL_SHADER_VALIDATION=1 (reported to stderr) and MTL_DEBUG_LAYER=1")
    parser.add_argument("--dump-requests", type=Path, metavar="DIR", help="export APS5_DUMP_REQUESTS=DIR: every recompile request is written there (MSL corpus)")
    parser.add_argument("--build-first", action="store_true", help="build the selected tests' executables (ninja -k 0, under the lock) before running")
    parser.add_argument("--allowlist", type=Path, default=check_skips.MACOS_ALLOWLIST, help="skip allowlist (default: tools/mac/m2_skip_allowlist.txt)")
    parser.add_argument("--dry-run", action="store_true", help="print the selection and the batches and run nothing")
    args = parser.parse_args()

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
    held = [name for name in chosen if gpu_groups.held(name)]
    runnable = [name for name in chosen if not gpu_groups.held(name)]
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

    stamp = utc_stamp()
    out = args.out or REPO / ".mac-work" / "logs" / "gpu-suite" / f"{stamp}-{'_'.join(re.sub(r'[^A-Za-z0-9_.-]', '', selector) for selector in args.selectors)[:60]}"
    out = out if out.is_absolute() else Path.cwd() / out
    (out / "batches").mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    extra = {}
    if args.validation:
        extra.update({"MTL_SHADER_VALIDATION": "1", "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1", "MTL_DEBUG_LAYER": "1"})
    if args.dump_requests:
        dump = args.dump_requests if args.dump_requests.is_absolute() else Path.cwd() / args.dump_requests
        dump.mkdir(parents=True, exist_ok=True)
        extra["APS5_DUMP_REQUESTS"] = str(dump)
    environment.update(extra)

    started_wall = datetime.datetime.now().isoformat(timespec="seconds")
    started = time.monotonic()
    if args.build_first:
        code = build_executables(build, runnable, registered, out / "build.log", environment)
        print(f"build: exit {code} (log {out / 'build.log'})")
    before = executable_mtimes(build, runnable, registered)
    cases = []
    for index, batch in enumerate(planned, 1):
        junit = out / "batches" / f"batch-{index:03d}.xml"
        code, seconds = run_batch(build, batch, args.timeout, junit, out / "batches" / f"batch-{index:03d}.log", environment)
        found = read_cases(junit) if junit.exists() else []
        seen = {case[0] for case in found}
        for missing in (name for name in batch if name not in seen):
            element = ET.Element("testcase", {"name": missing, "classname": missing, "time": "0", "status": "fail"})
            ET.SubElement(element, "failure", {"message": f"ctest did not report the test (exit {code})"})
            found.append((missing, "fail", 0.0, f"ctest did not report the test (exit {code})", "", element))
        cases += found
        passed = sum(1 for case in found if case[1] == "run")
        print(f"batch {index}/{len(planned)}: {len(batch)} test(s), {passed} passed, {seconds:.1f} s", flush=True)
    after = executable_mtimes(build, runnable, registered)
    finished_wall = datetime.datetime.now().isoformat(timespec="seconds")

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
    meta = {
        "started": started_wall, "finished": finished_wall, "seconds": time.monotonic() - started, "batches": len(planned),
        "git": git_state(), "build": os.path.relpath(build, REPO), "environment": extra,
        "rebuilt": [path for path, stamp_before in before.items() if after.get(path) != stamp_before and stamp_before is not None],
    }
    table, summary = summarize(results, groups, held, args.selectors, meta, allow_errors, allow_warnings)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"meta": meta, "selectors": args.selectors, "held": held, "results": results,
                                                  "skip_errors": allow_errors, "skip_warnings": allow_warnings}, indent=1) + "\n", encoding="utf-8")
    print(table)
    if allow_errors:
        print(f"skip check: {len(allow_errors)} error(s), see the summary")
    print(f"\nfull summary: {out / 'summary.txt'}\njunit: {out / 'junit.xml'}")
    bad = any(result["outcome"] in ("failed", "timeout", "not-built") for result in results) or allow_errors
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
