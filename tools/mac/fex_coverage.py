#!/usr/bin/env python3
"""How much of the guest code a title executes was found ahead of time, per binary, from aps5-fex's coverage files.

    python3 fex_coverage.py <precompile.json> [<run.json>] [--text]

<precompile.json> comes from `aps5-fex --precompile <elf> --coverage <file>`; <run.json> from a run with
APS5_FEX_COVERAGE=<file> on the same cache. Modules are matched by file ID (a hash of the file's bytes). Prints
JSON (or a table with --text):

    {"binaries": [{"path", "file_id", "text_bytes",
                   "fde_bytes", "fde_pct",            # .eh_frame FDE ranges / executable bytes
                   "seeds",                           # seed counts by kind
                   "aot_bytes", "aot_pct",            # bytes the precompile compiled / executable bytes
                   "executed_bytes",                  # run: bytes of all blocks used (JIT or cache)
                   "executed_aot_bytes", "executed_aot_pct",   # of those, inside precompiled blocks
                   "jit_only_bytes", "jit_only_pct",           # the rest: only the JIT found them
                   "run_jit_blocks", "run_cache_hit_blocks"}]}

Without a run file only the static fields are given. Percentages are 0-100 with one decimal."""
import json
import sys

VERSION = 1


def merge(ranges):
    """[[begin, size], ...] -> sorted disjoint [(begin, end), ...]."""
    spans = sorted((begin, begin + size) for begin, size in ranges if size > 0)
    merged = []
    for begin, end in spans:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return merged


def size(spans):
    return sum(end - begin for begin, end in spans)


def intersect(a, b):
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        begin, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if begin < end:
            result.append((begin, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def percent(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def load(path, mode):
    with open(path) as file:
        report = json.load(file)
    if report.get("version") != VERSION:
        raise ValueError(f"{path}: coverage version {report.get('version')}, expected {VERSION}")
    if report.get("mode") != mode:
        raise ValueError(f"{path}: a {report.get('mode')} report, expected {mode}")
    for module in report["modules"]:
        for key in ("path", "file_id", "text", "jit", "cache_hit"):
            if key not in module:
                raise ValueError(f"{path}: a module has no {key}")
    return report


def summarise(precompile, run=None):
    runs = {module["file_id"]: module for module in run["modules"]} if run else {}
    binaries = []
    for module in precompile["modules"]:
        text = merge(module["text"])
        aot = intersect(merge(module["jit"] + module["cache_hit"]), text)
        fde = intersect(merge(module.get("fde", [])), text)
        entry = {
            "path": module["path"],
            "file_id": module["file_id"],
            "text_bytes": size(text),
            "fde_bytes": size(fde),
            "fde_pct": percent(size(fde), size(text)),
            "seeds": module.get("seeds", {}),
            "aot_bytes": size(aot),
            "aot_pct": percent(size(aot), size(text)),
        }
        ran = runs.get(module["file_id"])
        if ran is not None:
            executed = intersect(merge(ran["jit"] + ran["cache_hit"]), text)
            found = intersect(executed, aot)
            entry.update({
                "executed_bytes": size(executed),
                "executed_aot_bytes": size(found),
                "executed_aot_pct": percent(size(found), size(executed)),
                "jit_only_bytes": size(executed) - size(found),
                "jit_only_pct": percent(size(executed) - size(found), size(executed)),
                "run_jit_blocks": ran.get("jit_blocks", 0),
                "run_cache_hit_blocks": ran.get("cache_hit_blocks", 0),
            })
        binaries.append(entry)
    return {"binaries": binaries}


def text_table(summary):
    lines = []
    for binary in summary["binaries"]:
        line = (f"{binary['path'].rsplit('/', 1)[-1]}: text {binary['text_bytes']} B, FDEs {binary['fde_pct']}%, "
                f"precompiled {binary['aot_pct']}% ({binary['seeds'].get('total', 0)} seeds)")
        if "executed_bytes" in binary:
            line += (f"; executed {binary['executed_bytes']} B: {binary['executed_aot_pct']}% found ahead of time, "
                     f"{binary['jit_only_pct']}% JIT only ({binary['run_jit_blocks']} blocks compiled at run time)")
        lines.append(line)
    return "\n".join(lines)


def main(arguments):
    text = "--text" in arguments
    paths = [argument for argument in arguments if argument != "--text"]
    if not 1 <= len(paths) <= 2:
        sys.stderr.write(__doc__)
        return 2
    summary = summarise(load(paths[0], "precompile"), load(paths[1], "run") if len(paths) == 2 else None)
    print(text_table(summary) if text else json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
