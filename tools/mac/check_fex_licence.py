#!/usr/bin/env python3
"""Fails when the FEX submodule could bring GPL-3.0 code into a GPL-2.0-only build.

    python3 tools/mac/check_fex_licence.py [repository root]

arvindfroi/FEX is based on willfaust/FEX, whose changes from 188e8224 (2026-08-28) onward are
GPL-3.0-only. Only commits whose whole history was audited may be pinned. The check fails when:
- the commit recorded in HEAD, staged in the index or checked out in 3rdparty/FEX is not in AUDITED;
- .gitmodules tracks a branch for 3rdparty/FEX, so `git submodule update --remote` could move the pin;
- a file is named like willfaust's GPL-3.0 markers (LICENSE-MADEIRA.md, COPYING.GPL-3.0);
- a LICENSE or COPYING file holds a GPL, LGPL or AGPL version 3 text;
- a file carries a GPL-3.0, LGPL-3.0 or AGPL-3.0 SPDX identifier or a "version 3 of the License" notice.
Every file of the checkout is scanned, built or not, except .git."""
import os
import re
import subprocess
import sys
from pathlib import Path

SUBMODULE = "3rdparty/FEX"

AUDITED = {
    "cad0c0969354e4f51462c5bf25f0f1acd5edce16":
        "arvindfroi/FEX anyps5-macos: FEX-Emu FEX-2607, willfaust commits up to ac555dd8, "
        "15 FEX-Emu backports, 6 arvindfroi commits; audited 2026-10-12",
}

MARKER_NAME = re.compile(r"^(license-madeira|copying\.(a|l)?gpl-?3)", re.I)
LICENCE_NAME = re.compile(r"^(licen[cs]e|copying)", re.I)
LICENCE_TEXT_V3 = re.compile(r"GNU\s+(AFFERO\s+|LESSER\s+)?GENERAL\s+PUBLIC\s+LICENSE\s+Version\s+3\b", re.I)
SPDX = re.compile(rb"SPDX-License-Identifier:([^\r\n]*)")
SPDX_ID = re.compile(r"[A-Za-z0-9.+-]+")
SPDX_V3 = re.compile(r"^(A|L)?GPL-3(\.0)?([-+].*)?$", re.I)
NOTICE_V3 = re.compile(rb"version\s+3\s+of\s+the\s+License", re.I)


def scan(tree):
    findings = []
    count = 0
    for directory, subdirectories, files in os.walk(tree):
        subdirectories[:] = sorted(name for name in subdirectories if name != ".git")
        for name in sorted(files):
            if name == ".git":
                continue
            path = Path(directory) / name
            shown = path.relative_to(tree).as_posix()
            count += 1
            if MARKER_NAME.match(name):
                findings.append(f"{shown}: GPL-3.0 marker file")
            if path.is_symlink() or not path.is_file():
                continue
            data = path.read_bytes()
            if LICENCE_NAME.match(name) and LICENCE_TEXT_V3.search(data.decode("latin-1")):
                findings.append(f"{shown}: GPL, LGPL or AGPL version 3 licence text")
            for expression in SPDX.findall(data):
                identifiers = SPDX_ID.findall(expression.decode("latin-1"))
                bad = [identifier for identifier in identifiers if SPDX_V3.match(identifier)]
                if bad:
                    findings.append(f"{shown}: SPDX identifier {' '.join(bad)}")
            if NOTICE_V3.search(data):
                findings.append(f"{shown}: \"version 3 of the License\" notice")
    return sorted(findings), count


def git(directory, *arguments):
    try:
        result = subprocess.run(["git", "-C", str(directory), *arguments], capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def gitlink(listing, field):
    if not listing:
        return None
    columns = listing.split("\t")[0].split()
    return columns[field] if len(columns) > field and columns[0] == "160000" else None


def pins(root):
    found = {}
    recorded = gitlink(git(root, "ls-tree", "HEAD", "--", SUBMODULE), 2)
    if recorded:
        found["recorded in HEAD"] = recorded
    staged = gitlink(git(root, "ls-files", "-s", "--", SUBMODULE), 1)
    if staged:
        found["staged in the index"] = staged
    checkout = root / SUBMODULE
    if (checkout / ".git").exists():
        top = git(checkout, "rev-parse", "--show-toplevel")
        if top and Path(top).resolve() == checkout.resolve():
            head = git(checkout, "rev-parse", "HEAD")
            if head:
                found["checked out"] = head
    return found


def tracked_branch(root):
    paths = git(root, "config", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$") or ""
    for line in paths.splitlines():
        key, _, value = line.partition(" ")
        if value.strip() == SUBMODULE:
            name = key[len("submodule."):-len(".path")]
            return git(root, "config", "--file", ".gitmodules", "--get", f"submodule.{name}.branch")
    return None


def check(root, audited=AUDITED):
    root = Path(root)
    findings = []
    checkout = root / SUBMODULE
    if not (checkout / "CMakeLists.txt").is_file():
        return [f"{SUBMODULE} is not checked out"], 0
    found = pins(root)
    if "recorded in HEAD" not in found and "staged in the index" not in found:
        findings.append(f"cannot read the commit of {SUBMODULE} from git in {root}")
    if "checked out" not in found:
        findings.append(f"cannot read the checked-out commit of {SUBMODULE}")
    for where, commit in found.items():
        if commit not in audited:
            findings.append(f"{SUBMODULE} commit {commit} ({where}) is not audited; audit its history and add it to AUDITED")
    branch = tracked_branch(root)
    if branch:
        findings.append(f".gitmodules tracks branch {branch} for {SUBMODULE}; remove it so the pin cannot drift")
    scanned, count = scan(checkout)
    findings.extend(f"{SUBMODULE}/{finding}" for finding in scanned)
    return findings, count


def main(arguments):
    root = Path(arguments[0]) if arguments else Path(__file__).resolve().parents[2]
    findings, count = check(root)
    for finding in findings:
        print(f"error: {finding}")
    if findings:
        return 1
    commit = pins(root)["checked out"]
    print(f"{SUBMODULE} {commit}: {AUDITED[commit]}; {count} files scanned, no GPL-3.0 markers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
