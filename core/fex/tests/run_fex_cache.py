"""FEXCore's disk cache under aps5-fex, on fixtures from run_fex_fixtures.py.

    python3 run_fex_cache.py <relinker> <aps5-fex> <patched prx directory> [--report <directory>] [<fixture>...]

For each fixture (default: callbacks, heap, crypto):
  1. run it twice on a fresh cache directory: same exit status and output both times, and the second run compiles
     0 blocks at run time (every block comes from the cache);
  2. precompile it (aps5-fex --precompile) into another fresh cache and run it on that cache, and print the coverage
     summary of tools/mac/fex_coverage.py: how much of the executed code was found ahead of time.
--report keeps each fixture's coverage files and summary in <directory>/<fixture>/."""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "tools" / "mac"))
import fex_coverage
import fex_toolchain
import run_fex_fixtures

DEFAULT = ["callbacks", "heap", "crypto"]


def run(runner, directory, cache, coverage, *arguments):
    environment = {**os.environ, "APS5_FEX_CACHE_DIR": str(cache), "APS5_FEX_COVERAGE": str(coverage)}
    environment.pop("APS5_FEX_CLEAR_CODE", None)
    return subprocess.run([str(runner), *arguments], cwd=directory, env=environment, capture_output=True, text=True, timeout=120)


def check(name, relinker, runner, libraries, report):
    build, status = run_fex_fixtures.COMPILED[name]
    failures = []
    with tempfile.TemporaryDirectory(prefix=f"aps5-fex-cache-{name}-") as directory:
        directory = pathlib.Path(directory)
        build(directory)
        relinked = subprocess.run([str(relinker), "--skip-sce-module", "input.elf", "eboot.elf"], cwd=directory, capture_output=True,
                                  text=True, timeout=60)
        if relinked.returncode != 0:
            return [f"{name}: relink failed: {relinked.stderr.strip()}"]
        (directory / "libs").symlink_to(libraries, target_is_directory=True)

        first = run(runner, directory, directory / "cache", directory / "run1.json", "eboot.elf")
        second = run(runner, directory, directory / "cache", directory / "run2.json", "eboot.elf")
        if first.returncode != status:
            failures.append(f"{name}: first run exit {first.returncode}, expected {status}: {first.stderr.strip()[-300:]}")
        if (second.returncode, second.stdout) != (first.returncode, first.stdout):
            failures.append(f"{name}: second run differs: exit {second.returncode} vs {first.returncode}, output {second.stdout!r} vs {first.stdout!r}")
        if not failures:
            again = json.loads((directory / "run2.json").read_text())
            compiled = sum(module["jit_blocks"] for module in again["modules"]) + again["blocks_outside_modules"]["jit"]
            hits = sum(module["cache_hit_blocks"] for module in again["modules"])
            if compiled != 0 or hits == 0:
                failures.append(f"{name}: second run compiled {compiled} blocks at run time ({hits} from the cache), expected 0")

        precompiled = run(runner, directory, directory / "aot", directory / "unused.json", "--precompile", "eboot.elf", "--coverage",
                          str(directory / "precompile.json"))
        if precompiled.returncode != 0:
            failures.append(f"{name}: --precompile exit {precompiled.returncode}: {precompiled.stderr.strip()[-300:]}")
            return failures
        after = run(runner, directory, directory / "aot", directory / "run-aot.json", "eboot.elf")
        if (after.returncode, after.stdout) != (first.returncode, first.stdout):
            failures.append(f"{name}: run on the precompiled cache differs: exit {after.returncode}, output {after.stdout!r}")
            return failures
        summary = fex_coverage.summarise(fex_coverage.load(directory / "precompile.json", "precompile"),
                                         fex_coverage.load(directory / "run-aot.json", "run"))
        print(f"{name}: {fex_coverage.text_table(summary)}")
        if report is not None:
            target = report / name
            target.mkdir(parents=True, exist_ok=True)
            for file in ("precompile.json", "run-aot.json", "run1.json", "run2.json"):
                shutil.copy(directory / file, target / file)
            (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return failures


def main(arguments):
    relinker, runner, libraries = (pathlib.Path(argument).resolve() for argument in arguments[:3])
    rest = arguments[3:]
    report = None
    if "--report" in rest:
        index = rest.index("--report")
        report = pathlib.Path(rest[index + 1]).resolve()
        del rest[index:index + 2]
    if not fex_toolchain.available():
        print("skipped: clang, llvm-nm or an LLVM ELF linker is missing")
        return 77
    failures = []
    for name in rest or DEFAULT:
        failures += check(name, relinker, runner, libraries, report)
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
