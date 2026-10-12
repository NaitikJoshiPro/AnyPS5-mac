"""Microbenchmarks of aps5-fex against native code. The kernels in bench/ are built from one source as
an arm64 program and as x86-64 guest programs, the way run_fex_fixtures.py builds its fixtures, for
three guest targets: sse2 (-march=x86-64), sse4 (-march=x86-64-v2) and avx2 (-march=znver2, the PS5).

    python3 run_fex_bench.py <relinker> <aps5-fex> <patched prx directory> [--smoke] [options]
    python3 run_fex_bench.py --compare <result.json>...

--smoke runs every kernel at a tiny size natively and as the avx2 guest and checks each result
against the native one (the ctest fex_bench_smoke). Otherwise each kernel runs in its own process:
one cold repetition, which for a guest includes translating its code, then warm ones. The table of
median times and of speed relative to native (native time / guest time, by the clock and, for one
thread, by the CPU time of that thread, which a loaded machine disturbs less) goes to stdout, the
samples, machine and load averages to a JSON file in .mac-work/bench/."""
import argparse
import concurrent.futures
import datetime
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fex_toolchain

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE / "bench"
REPOSITORY = HERE.parents[2]
RESULTS = REPOSITORY / ".mac-work" / "bench"

VARIANTS = {"sse2": "x86-64", "sse4": "x86-64-v2", "avx2": "znver2"}
SMOKE_VARIANTS = ["avx2"]
SMOKE_PERMILLE = 10
KERNEL_SOURCES = ["kernels.cpp", "bench_main.cpp"]
IMPORTS = {
    "libc.prx": ["memcpy", "memset", "qsort", "puts", "exit"],
    "libkernel.prx": ["sceKernelGetProcessTimeCounter", "sceKernelGetProcessTimeCounterFrequency", "clock_gettime",
                      "scePthreadCreate", "scePthreadJoin"],
}
FLAGS = ["-std=c++20", "-O2"]
FREESTANDING = ["-ffreestanding", "-fno-builtin", "-fno-stack-protector"]
LINE = re.compile(r"^BENCH (\S+) threads=(\d+) work=(\d+) sum=([0-9a-f]+) extra=(\d+) stable=([01]) ns=([\d,]+) cpu=([\d,]+)$")
GROUPS = ["int", "mem", "simd", "atomic", "bridge"]


class BenchError(Exception):
    pass


def check(command, **options):
    done = subprocess.run(command, capture_output=True, text=True, **options)
    if done.returncode != 0:
        raise BenchError(f"{' '.join(map(str, command))} failed: {(done.stderr or done.stdout).strip()[-800:]}")
    return done


def build_native(directory):
    """bench-native: the kernels and harness as freestanding code, as the guest gets them, around macOS's libc."""
    clang = ["xcrun", "clang++", "-arch", "arm64", *FLAGS]
    jobs = [(BENCH / source, FREESTANDING) for source in KERNEL_SOURCES] + [(BENCH / "bench_native.cpp", [])]
    objects = [directory / f"native-{source.stem}.o" for source, _ in jobs]

    def compile_one(index):
        source, flags = jobs[index]
        check([*clang, *flags, "-c", str(source), "-o", str(objects[index])])

    with concurrent.futures.ThreadPoolExecutor() as pool:
        list(pool.map(compile_one, range(len(jobs))))
    executable = directory / "bench-native"
    check([*clang, *map(str, objects), "-o", str(executable)])
    return executable


def build_guest(directory, march, relinker, libraries):
    """<directory>/eboot.elf: the kernels, harness and bench_guest.cpp relinked for aps5-fex."""
    directory.mkdir(parents=True)
    sources = [BENCH / source for source in KERNEL_SOURCES] + [BENCH / "bench_guest.cpp"]
    objects = [directory / f"{source.stem}.o" for source in sources]
    extra = [*FLAGS, f"-march={march}"]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        list(pool.map(lambda pair: fex_toolchain.compile(pair[0], pair[1], extra=extra), zip(sources, objects)))
    defined = set()
    for obj in objects:
        defined.update(fex_toolchain.symbols(obj, "-g", "--defined-only"))
    for obj in objects[:len(KERNEL_SOURCES)]:
        library_calls = [name for name in fex_toolchain.symbols(obj, "-u") if name not in defined]
        if library_calls:
            raise BenchError(f"{obj.name} ({march}) calls {', '.join(library_calls)}: the compiler turned kernel code into library calls")
    allowed = {name for names in IMPORTS.values() for name in names}
    imported = {name for obj in objects for name in fex_toolchain.symbols(obj, "-u")} - defined
    if imported - allowed:
        raise BenchError(f"the {march} guest imports {', '.join(sorted(imported - allowed))}, which IMPORTS does not list")
    renamed = []
    for obj in objects:
        fex_toolchain.nidify(obj, obj.with_suffix(".nid.o"))
        renamed.append(obj.with_suffix(".nid.o"))
    stubs = []
    for name, names in IMPORTS.items():
        fex_toolchain.stub_library(names, directory / name, name)
        stubs.append(directory / name)
    fex_toolchain.link_executable(renamed, stubs, directory / "input.elf")
    fex_toolchain.make_header_room(directory / "input.elf")
    check([str(relinker), "--skip-sce-module", "input.elf", "eboot.elf"], cwd=directory, timeout=60)
    (directory / "libs").symlink_to(libraries, target_is_directory=True)
    return directory / "eboot.elf"


def run(command, cwd, environment, timeout):
    """The BENCH lines of one process by kernel name, its exit status and the tail of its error output."""
    try:
        done = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, None, f"timed out after {timeout} s"
    results = {}
    for line in done.stdout.splitlines():
        if found := LINE.match(line.strip()):
            name, threads, work, digest, extra, stable, times, cpu = found.groups()
            results[name] = {"threads": int(threads), "work": int(work), "sum": digest, "extra": int(extra),
                             "stable": stable == "1", "ns": [int(value) for value in times.split(",")],
                             "cpu": [int(value) for value in cpu.split(",")]}
    return results, done.returncode, done.stderr.strip()[-600:]


def kernel_names(native, environment):
    done = check([str(native), "list"], env=environment, timeout=30)
    return [line.split()[1] for line in done.stdout.splitlines() if line.startswith("KERNEL ")]


def environments(arguments):
    """The native program's environment and a guest environment for each --config."""
    native = {**os.environ, "APS5_FEX_QOS": arguments.qos}
    native.pop("APS5_FEX_CONFIG", None)
    return native, {config: {**native, **({"APS5_FEX_CONFIG": config} if config else {})} for config in arguments.config or [""]}


def smoke(arguments, native, guests):
    native_environment, guest_environments = environments(arguments)
    guest_environment = next(iter(guest_environments.values()))
    names = kernel_names(native, native_environment)
    expected, status, errors = run([str(native), "all", str(SMOKE_PERMILLE), "2"], None, native_environment, 20)
    failures = [] if status == 0 else [f"native: exit {status}: {errors}"]
    failures += [f"native: {name} gave no result" for name in names if name not in expected]
    for variant, eboot in guests.items():
        got, status, errors = run([str(arguments.runner), "eboot.elf", "all", str(SMOKE_PERMILLE), "2"], eboot.parent,
                                  guest_environment, 25)
        if status != 0:
            failures.append(f"{variant}: exit {status}: {errors}")
        for name in names:
            if name not in got:
                failures.append(f"{variant}: {name} gave no result")
            elif not got[name]["stable"]:
                failures.append(f"{variant}: {name} gave different results in its two repetitions")
            elif name in expected and got[name]["sum"] != expected[name]["sum"]:
                failures.append(f"{variant}: {name} gave {got[name]['sum']}, the native program {expected[name]['sum']}")
    for failure in failures:
        print(failure)
    if failures:
        return 1
    print(f"{len(names)} kernels gave the native results as {', '.join(guests)} guest programs under aps5-fex")
    return 0


def summary(samples, cpu):
    """Cold time and the median of the warm repetitions, in nanoseconds, by the clock and as the CPU time
    of the thread that timed them."""
    warm = [value for process in samples for value in process[1:]]
    return {"cold": samples[0][0], "median": statistics.median(warm), "min": min(warm), "max": max(warm), "count": len(warm),
            "cpu_median": statistics.median(value for process in cpu for value in process[1:]), "samples": samples, "cpu": cpu}


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else None


def machine():
    def sysctl(name):
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()

    def git(*command):
        return subprocess.run(["git", "-C", str(REPOSITORY), *command], capture_output=True, text=True).stdout.strip()

    return {"cpu": sysctl("machdep.cpu.brand_string"), "performance_cores": sysctl("hw.perflevel0.logicalcpu"),
            "efficiency_cores": sysctl("hw.perflevel1.logicalcpu"), "memory": sysctl("hw.memsize"),
            "macos": subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip(),
            "commit": git("rev-parse", "HEAD"), "fex": git("rev-parse", "HEAD:3rdparty/FEX")}


def benchmark(arguments, native, guests):
    native_environment, guest_environments = environments(arguments)
    names = [name for name in kernel_names(native, native_environment)
             if not arguments.kernels or any(re.fullmatch(pattern, name) for pattern in arguments.kernels.split(","))]
    binaries = {"native": ([str(native)], None, native_environment)}
    columns = {}
    for variant, eboot in guests.items():
        for config, environment in guest_environments.items():
            columns[f"{variant}:{config}" if config else variant] = {"march": VARIANTS[variant], "config": config}
            binaries[f"{variant}:{config}" if config else variant] = ([str(arguments.runner), "eboot.elf"], eboot.parent, environment)
    guests = list(columns)
    report = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "machine": machine(), "load_start": os.getloadavg(),
              "permille": arguments.permille, "repetitions": arguments.reps, "processes": arguments.processes,
              "qos": arguments.qos, "label": arguments.label, "variants": columns,
              "kernels": {}}
    print(f"load average {' '.join(f'{value:.2f}' for value in report['load_start'])}; {len(names)} kernels, "
          f"{arguments.processes} process(es) of {arguments.reps} repetitions each", flush=True)
    print("\n".join(header(guests)), flush=True)
    for name in names:
        entry = {"load": os.getloadavg()[0]}
        samples = {binary: [] for binary in binaries}
        cpu = {binary: [] for binary in binaries}
        for _ in range(arguments.processes):
            for binary, (command, cwd, environment) in binaries.items():
                results, status, errors = run([*command, name, str(arguments.permille), str(arguments.reps)], cwd, environment,
                                              arguments.timeout)
                if status != 0 or name not in results:
                    entry.setdefault("errors", {})[binary] = f"exit {status}: {errors}"
                    continue
                result = results[name]
                entry.setdefault("sum", {})[binary] = result["sum"]
                entry.setdefault("extra", {})[binary] = result["extra"]
                if not result["stable"]:
                    entry.setdefault("errors", {})[binary] = "the repetitions gave different results"
                entry["threads"] = result["threads"]
                samples[binary].append(result["ns"])
                cpu[binary].append(result["cpu"])
        for binary, values in samples.items():
            if values:
                entry[binary] = summary(values, cpu[binary])
        digests = set(entry.get("sum", {}).values())
        entry["matches"] = len(digests) == 1 and "errors" not in entry
        report["kernels"][name] = entry
        print(row(name, entry, guests), flush=True)
    report["load_end"] = os.getloadavg()
    for key, clock in (("speed", "median"), ("cpu_speed", "cpu_median")):
        report[key] = {variant: {group: geometric_mean([speed(entry, variant, clock) for name, entry in report["kernels"].items()
                                                        if name.startswith(group + "_") and speed(entry, variant, clock)])
                                 for group in GROUPS} for variant in guests}
    report["bridge"] = {variant: bridge_costs(report["kernels"], variant) for variant in guests}
    print(table(report, guests))
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{arguments.label}.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"load average at the end {' '.join(f'{value:.2f}' for value in report['load_end'])}; results in {path}")
    return 0 if all(entry["matches"] for entry in report["kernels"].values()) else 1


def speed(entry, variant, clock="median"):
    """Native time over guest time; by CPU time only for kernels that run on one thread."""
    if "native" not in entry or variant not in entry or (clock == "cpu_median" and entry.get("threads", 1) != 1):
        return None
    return entry["native"][clock] / entry[variant][clock] if entry[variant][clock] else None


def row(name, entry, guests):
    cells = [name, f"{entry['native']['median'] / 1e6:.2f}" if "native" in entry else "-"]
    for variant in guests:
        if variant in entry:
            cells += [f"{entry[variant]['median'] / 1e6:.2f}"]
            cells += [f"{value:.2f}" if (value := speed(entry, variant, clock)) else "-" for clock in ("median", "cpu_median")]
            cells.append(f"{entry[variant]['cold'] / 1e6:.2f}")
        else:
            cells += ["-", "-", "-", "-"]
    cells.append(f"{entry['load']:.1f}")
    cells.append("ok" if entry["matches"] else "MISMATCH " + "; ".join(f"{k}: {v}" for k, v in entry.get("errors", {}).items()))
    return "| " + " | ".join(cells) + " |"


def bridge_costs(kernels, variant):
    """Nanoseconds per guest->native call and per native->guest callback over native code doing the same,
    and per thread created and joined."""
    def per_operation(name, binary):
        entry = kernels.get(name, {})
        operations = entry.get("extra", {}).get(binary)
        return entry[binary]["median"] / operations if binary in entry and operations else None

    costs = {}
    for name, key in (("bridge_call", "call_ns"), ("bridge_qsort", "callback_ns"), ("bridge_thread", "thread_ns")):
        guest, native = per_operation(name, variant), per_operation(name, "native")
        if guest is not None and native is not None:
            costs[key] = guest - native if key != "thread_ns" else guest
            costs["native_" + key] = native
    return costs


def header(guests):
    cells = ["kernel", "native ms"]
    for variant in guests:
        cells += [f"{variant} ms", f"{variant} speed", f"{variant} cpu speed", f"{variant} cold ms"]
    cells += ["load 1m", "result"]
    return ["| " + " | ".join(cells) + " |", "|" + "---|" * len(cells)]


def table(report, guests):
    lines = ["", *header(guests)]
    lines += [row(name, entry, guests) for name, entry in report["kernels"].items()]
    lines.append("")
    for group in GROUPS:
        values = [f"{variant} {report['speed'][variant][group]:.2f}"
                  + (f" (cpu {report['cpu_speed'][variant][group]:.2f})" if report["cpu_speed"][variant][group] else "")
                  for variant in guests if report["speed"][variant][group]]
        if values:
            lines.append(f"geometric mean speed, {group}: {', '.join(values)}")
    for variant in guests:
        costs = report["bridge"][variant]
        parts = [f"{label} {costs[key]:.0f} ns" for key, label in (("call_ns", "guest->native call"),
                 ("callback_ns", "native->guest callback")) if key in costs]
        if "thread_ns" in costs:
            parts.append(f"thread create+join {costs['thread_ns'] / 1000:.0f} us (native {costs['native_thread_ns'] / 1000:.0f} us)")
        if parts:
            lines.append(f"bridge, {variant}: {', '.join(parts)}")
    return "\n".join(lines)


def compare(paths):
    """The speed of every kernel in several result files, side by side."""
    reports = [(report.get("label", path.stem), report) for path in paths for report in [json.loads(path.read_text())]]
    columns = [(label, report, variant) for label, report in reports for variant in report["variants"]]
    names = list(dict.fromkeys(name for _, report in reports for name in report["kernels"]))
    lines = ["| kernel | " + " | ".join(f"{label} {variant}" for label, _, variant in columns) + " |", "|---|" + "---|" * len(columns)]
    for name in names:
        cells = []
        for _, report, variant in columns:
            value = speed(report["kernels"][name], variant) if name in report["kernels"] else None
            cells.append(f"{value:.2f}" if value else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for group in GROUPS:
        values = [report["speed"][variant].get(group) for _, report, variant in columns]
        lines.append(f"| geomean {group} | " + " | ".join(f"{value:.2f}" if value else "-" for value in values) + " |")
    print("\n".join(lines))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("relinker", type=pathlib.Path, nargs="?")
    parser.add_argument("runner", type=pathlib.Path, nargs="?")
    parser.add_argument("libraries", type=pathlib.Path, nargs="?")
    parser.add_argument("--compare", type=pathlib.Path, nargs="+", metavar="JSON", help="print the speeds of earlier runs side by side")
    parser.add_argument("--smoke", action="store_true", help="check every kernel at a tiny size against the native results")
    parser.add_argument("--variants", help="guest targets, from " + ", ".join(VARIANTS) + " (default: all; avx2 with --smoke)")
    parser.add_argument("--kernels", default="", help="comma-separated regular expressions of kernel names")
    parser.add_argument("--permille", type=int, default=1000, help="work per repetition, in thousandths of the full work")
    parser.add_argument("--reps", type=int, default=8, help="repetitions per process, the first one cold")
    parser.add_argument("--processes", type=int, default=1, help="processes per kernel and program")
    parser.add_argument("--config", action="append", help="APS5_FEX_CONFIG for the guests, e.g. TSOEnabled=0; repeat it to "
                        "run each guest under several (an empty value is FEX's defaults)")
    parser.add_argument("--qos", default="user-interactive", help="APS5_FEX_QOS for every thread of both programs")
    parser.add_argument("--label", default="default", help="suffix of the JSON file")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per process")
    parser.add_argument("--keep", type=pathlib.Path, help="build in this new directory and keep it")
    arguments = parser.parse_args()
    if arguments.compare:
        return compare(arguments.compare)
    if arguments.libraries is None:
        parser.error("the relinker, aps5-fex and the patched prx directory are required")
    arguments.runner = arguments.runner.resolve()
    variants = [v for v in arguments.variants.split(",") if v] if arguments.variants else SMOKE_VARIANTS if arguments.smoke else list(VARIANTS)
    if unknown := [v for v in variants if v not in VARIANTS]:
        parser.error(f"unknown variant {', '.join(unknown)}")
    if not arguments.smoke and (arguments.reps < 6 or arguments.processes < 1):
        parser.error("--reps must leave at least five warm repetitions")
    if not fex_toolchain.available():
        print("skipped, clang, llvm-nm or an LLVM ELF linker is missing")
        return 77
    with tempfile.TemporaryDirectory(prefix="aps5-fex-bench-") as scratch:
        directory = arguments.keep or pathlib.Path(scratch)
        directory.mkdir(parents=True, exist_ok=arguments.keep is None)
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                native = pool.submit(build_native, directory)
                guests = {v: pool.submit(build_guest, directory / v, VARIANTS[v], arguments.relinker.resolve(),
                                         arguments.libraries.resolve()) for v in variants}
                native = native.result()
                guests = {v: future.result() for v, future in guests.items()}
            return smoke(arguments, native, guests) if arguments.smoke else benchmark(arguments, native, guests)
        except (BenchError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            print(f"error: {error}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
