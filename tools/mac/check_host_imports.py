#!/usr/bin/env python3
"""Host system call shim inventory check (D35).

Every libSystem symbol a built prx library imports must be classified in core/libs/hostsys/host_imports.tsv, so a new
library cannot bring back EFAULT on displaced guest buffers by calling a kernel function nobody looked at. Rows whose
handling is `shim` must be defined by the shim archive; with --shim-on no prx may still import them from libSystem
(the shim was not linked in).
"""
import argparse
import subprocess
import sys
from pathlib import Path

CLASSES = {"buffer", "path", "stdio", "address", "retaining", "scalar", "userspace", "data"}
HANDLING = {"shim", "hand", "none"}
KERNEL_CLASSES = {"buffer", "path", "stdio", "retaining"}


def load(path):
    rows = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"{path}:{number}: expected 4 tab-separated fields, got {len(fields)}")
        symbol, kind, handling, _ = fields
        if kind not in CLASSES:
            raise ValueError(f"{path}:{number}: {symbol}: unknown class {kind!r}")
        if handling not in HANDLING:
            raise ValueError(f"{path}:{number}: {symbol}: unknown handling {handling!r}")
        if symbol in rows:
            raise ValueError(f"{path}:{number}: {symbol} listed twice")
        rows[symbol] = (kind, handling)
    return rows


def symbols(command):
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()


def imports(library):
    found = set()
    for line in symbols(["nm", "-m", "-u", str(library)]):
        if not line.endswith("(from libSystem)"):
            continue
        name = line.split()[-3]
        found.add(name[1:] if name.startswith("_") else name)
    return found


def defined(archive):
    found = set()
    for line in symbols(["nm", "-U", "-g", str(archive)]):
        parts = line.split()
        if len(parts) == 3 and parts[1] == "T":
            found.add(parts[2][1:])
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--libraries", required=True, help="directory of the built (patched) prx libraries")
    parser.add_argument("--archive", help="the shim's static library (checks that every `shim` row is defined)")
    parser.add_argument("--shim-on", action="store_true", help="APS5_HOST_SYSCALL_SHIM is ON: no prx imports a shimmed symbol")
    parser.add_argument("--drop", action="append", default=[], help="ignore this row (the negative test: an unclassified import must fail)")
    parser.add_argument("--minimum", type=int, default=100, help="fewest prx libraries expected (guards an empty directory)")
    arguments = parser.parse_args()

    rows = load(arguments.inventory)
    for name in arguments.drop:
        rows.pop(name, None)
    libraries = sorted(Path(arguments.libraries).glob("*.prx"))
    if len(libraries) < arguments.minimum:
        print(f"error: {len(libraries)} prx libraries in {arguments.libraries}, expected at least {arguments.minimum} (build the libs target)")
        return 1
    importers = {}
    for library in libraries:
        for name in imports(library):
            importers.setdefault(name, []).append(library.stem)

    errors = []
    for name in sorted(set(importers) - set(rows)):
        errors.append(f"{name} (imported by {', '.join(importers[name])}) is not classified in {arguments.inventory}: decide whether "
                      "it hands a guest pointer to the kernel, shim it if so, and add a row")
    shimmed = {name for name, (_, handling) in rows.items() if handling == "shim"}
    for name, (kind, handling) in sorted(rows.items()):
        if handling == "shim" and kind not in KERNEL_CLASSES:
            errors.append(f"{name}: handling shim on class {kind}")
    if arguments.archive:
        missing = sorted(shimmed - defined(arguments.archive))
        if missing:
            errors.append(f"shim rows the archive does not define: {', '.join(missing)}")
    if arguments.shim_on:
        for name in sorted(shimmed & set(importers)):
            errors.append(f"{name} is still imported from libSystem by {', '.join(importers[name])} with the shim on")

    stale = sorted(set(rows) - set(importers) - (shimmed if arguments.shim_on else set()))
    counts = {}
    for name in importers:
        kind, handling = rows.get(name, ("unclassified", "-"))
        counts[(kind, handling)] = counts.get((kind, handling), 0) + 1
    summary = ", ".join(f"{kind}/{handling} {count}" for (kind, handling), count in sorted(counts.items()))
    print(f"{len(libraries)} prx libraries import {len(importers)} libSystem symbols: {summary}")
    if arguments.shim_on:
        print(f"shim on: the {len(shimmed)} shimmed symbols are defined inside the libraries, none imported from libSystem")
    if stale:
        print(f"note: {len(stale)} classified symbols are no longer imported: {', '.join(stale)}")
    for error in errors:
        print(f"error: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
