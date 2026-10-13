#!/usr/bin/env python3
"""Lists the upstream commits that changed a file the Metal port replaces with a twin, so the twin can be updated.

Usage:
    python3 -I tools/mac/port_ledger.py [--to REF] [--since REF] [--stat] [--check]
    python3 -I tools/mac/port_ledger.py --mark REF

The macOS build replaces upstream Vulkan sources with Objective-C++ twins (docs/mac/M2_DESIGN.md D-M2-3): the
pairs come from core/libs/prx/libSceAgcDriver/specifics/macos/twins/*.cmake ("original=twin"), the twin headers
from specifics/macos/overlay (each shadows the header at the same path under core/libs; overlay headers with no
such original are Metal-only and not listed), and the twinned tests
from the "twin-test" lines of tools/mac/port_ledger.txt. An upstream change to an original is not compiled on
macOS until someone ports it to the twin, so after every `git fetch upstream` (and before merging it) run this
and port what it lists; then record the reviewed upstream commit with --mark.

  --to REF     the upstream state to compare with (default: upstream/main)
  --since REF  the last reviewed upstream commit (default: the "reviewed" line of tools/mac/port_ledger.txt, else
               the merge base of HEAD and --to)
  --stat       show lines added/removed per original (may fetch blobs in a blobless clone)
  --check      exit 1 when there is something to port
  --mark REF   write REF (resolved to a commit) as the reviewed upstream commit into tools/mac/port_ledger.txt
"""
import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = Path(__file__).resolve().parent / "port_ledger.txt"
DRIVER = "core/libs/prx/libSceAgcDriver"
TWINS = f"{DRIVER}/specifics/macos/twins"
OVERLAY = f"{DRIVER}/specifics/macos/overlay"
PAIR = re.compile(r'"([^"=\s]+)=([^"=\s]+)"')


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def twin_pairs(repo):
    """[(original, twin)] repository-relative: twin sources, overlay headers and twinned tests."""
    pairs = []
    for listing in sorted((Path(repo) / TWINS).glob("*.cmake")):
        for original, twin in PAIR.findall(listing.read_text(encoding="utf-8")):
            pairs.append((f"{DRIVER}/{original}", f"{DRIVER}/{twin}"))
    overlay = Path(repo) / OVERLAY
    if overlay.is_dir():
        for header in sorted(path for path in overlay.rglob("*") if path.is_file()):
            relative = header.relative_to(overlay).as_posix()
            if (Path(repo) / "core/libs" / relative).exists():
                pairs.append((f"core/libs/{relative}", f"{OVERLAY}/{relative}"))
    reviewed, tests = read_ledger(Path(repo) / LEDGER.relative_to(REPO))
    pairs += tests
    return pairs


def read_ledger(path):
    """(reviewed upstream commit or None, [(original test, twin test)]) from tools/mac/port_ledger.txt."""
    reviewed, tests = None, []
    if not path.exists():
        return reviewed, tests
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if fields[0] == "reviewed" and len(fields) == 2:
            reviewed = fields[1]
        elif fields[0] == "twin-test" and len(fields) == 3:
            tests.append((fields[1], fields[2]))
        else:
            raise ValueError(f"{path}:{number}: expected 'reviewed <commit>' or 'twin-test <original> <twin>'")
    return reviewed, tests


def write_reviewed(path, commit):
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [line for line in lines if not line.split("#", 1)[0].strip().startswith("reviewed ")]
    insert = next((index for index, line in enumerate(kept) if line.strip() and not line.startswith("#")), len(kept))
    kept.insert(insert, f"reviewed {commit}")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def changes(repo, since, to, originals):
    """{original: [(commit, date, subject)]} for the commits in since..to that touch each original, oldest first."""
    found = collections.OrderedDict((original, []) for original in originals)
    if not originals:
        return found
    log = git(repo, "log", "--reverse", "--format=\x1e%H\x1f%cs\x1f%s", "--name-only", f"{since}..{to}", "--", *originals)
    wanted = set(originals)
    for record in log.split("\x1e")[1:]:
        header, _, names = record.partition("\n")
        commit, date, subject = header.split("\x1f", 2)
        for name in names.split():
            if name in wanted:
                found[name].append((commit, date, subject))
    return found


def line_counts(repo, since, to, original):
    out = git(repo, "diff", "--numstat", f"{since}..{to}", "--", original).split()
    return (int(out[0]), int(out[1])) if len(out) >= 2 and out[0].isdigit() else (0, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--to", default="upstream/main")
    parser.add_argument("--since")
    parser.add_argument("--stat", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mark", metavar="REF")
    parser.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    args = parser.parse_args()
    ledger = args.repo / LEDGER.relative_to(REPO)
    try:
        if args.mark:
            commit = git(args.repo, "rev-parse", "--verify", f"{args.mark}^{{commit}}").strip()
            write_reviewed(ledger, commit)
            print(f"reviewed {commit}")
            return 0
        reviewed, _ = read_ledger(ledger)
        to = git(args.repo, "rev-parse", "--verify", f"{args.to}^{{commit}}").strip()
        since = args.since or reviewed or git(args.repo, "merge-base", "HEAD", to).strip()
        since = git(args.repo, "rev-parse", "--verify", f"{since}^{{commit}}").strip()
        pairs = twin_pairs(args.repo)
    except subprocess.CalledProcessError as error:
        parser.error(error.stderr.strip() or str(error))
    except ValueError as error:
        parser.error(str(error))
    twins = collections.defaultdict(list)
    for original, twin in pairs:
        twins[original].append(twin)
    found = changes(args.repo, since, to, sorted(twins))
    pending = {original: commits for original, commits in found.items() if commits}
    print(f"upstream {since[:10]}..{to[:10]}: {len(twins)} twinned original(s), {len(pending)} changed")
    for original, commits in pending.items():
        counts = ""
        if args.stat:
            added, removed = line_counts(args.repo, since, to, original)
            counts = f" (+{added} -{removed})"
        print(f"\n{original}{counts}\n  twin: {', '.join(twins[original])}")
        for commit, date, subject in commits:
            print(f"  {commit[:10]} {date} {subject}")
    missing = [original for original in twins if not (args.repo / original).exists()]
    if missing:
        print("\noriginals that no longer exist here (renamed or removed upstream?): " + " ".join(missing))
    if pending:
        print(f"\nport these changes to the twins, then: tools/mac/port_ledger.py --mark {to[:10]}")
    return 1 if args.check and (pending or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
