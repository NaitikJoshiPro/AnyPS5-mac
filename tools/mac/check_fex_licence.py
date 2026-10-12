#!/usr/bin/env python3
"""Fails when the vendored FEX tree could bring GPL-3.0 code into a GPL-2.0-only build.

    python3 tools/mac/check_fex_licence.py [repository root]

3rdparty/FEX is FEX-Emu/FEX added with `git subtree add --squash`, plus the slim macOS series applied with
`git am --directory=3rdparty/FEX` (DECISIONS.md D13). FEX-Emu/FEX is MIT at every revision. willfaust/FEX
relicensed his own later work to GPL-3.0 from 188e8224 (2026-08-28); his commits up to ac555dd8 are MIT.
The check fails when:
- 3rdparty/FEX in HEAD is not a vendored directory (a gitlink is the old submodule layout);
- the upstream revision that git subtree recorded last (git-subtree-split) is not in AUDITED_BASES;
- a commit that changes 3rdparty/FEX is authored by Will Faust without naming the audited commit it was
  ported from ("ported from arvindfroi/FEX commit <sha>", <sha> in AUDITED_SOURCES), or is authored by a
  contributor of his GPL-3.0 period;
- a submodule under 3rdparty/FEX is recorded at a commit that is not in AUDITED_NESTED, or tracks a branch;
- core/fex/CMakeLists.txt builds FEX without -DENABLE_GDB_SYMBOLS=OFF: FEX turns that option on whenever GDB's
  GPL-3.0 <gdb/jit-reader.h> is installed, and GDBJIT.cpp then includes it;
- a file is named like willfaust's GPL-3.0 markers (LICENSE-MADEIRA.md, COPYING.GPL-3.0);
- a LICENSE or COPYING file holds a GPL, LGPL or AGPL version 3 text;
- a file carries a GPL-3.0, LGPL-3.0 or AGPL-3.0 SPDX identifier or a "version 3 of the License" notice.
Every file under 3rdparty/FEX is scanned, built or not, including checked-out nested submodules, except .git."""
import os
import re
import subprocess
import sys
from pathlib import Path

VENDORED = "3rdparty/FEX"
RUNNER_CMAKE = "core/fex/CMakeLists.txt"

AUDITED_BASES = {
    "14c92681f4d62cf84d901460e0358de09c8847a7":
        "FEX-Emu FEX-2610, 2026-10-07: MIT; FEX-2607..FEX-2610 re-audited 2026-10-12, no Will Faust commits",
}

# Every commit of arvindfroi/FEX FEX-2607..cad0c096 that is not a FEX-Emu backport: Will Faust's 55 commits up
# to ac555dd8, his last MIT one, and arvindfroi's 6. Audited 2026-10-12 (.mac-work/research/fex-licence.md).
AUDITED_SOURCES = {
    "fce78cefd6ad213b270555238e7a4184fa42eb2e",  # 2026-03-07 Will Faust
    "1587b0084b12a62e7fa5adf5aac6bb8e46c37809",  # 2026-04-28 Will Faust
    "024a5851085af07c272dd203df7993e79a33f75a",  # 2026-04-28 Will Faust
    "344cc9c87c0bc167114e6e71207ecd87531400fb",  # 2026-04-28 Will Faust
    "a38fb5b4e2e89484a24b1cb2eff53a53685d77f2",  # 2026-04-28 Will Faust
    "f7a255603238fc6ab352f5c1775e17dd3a8e6e5a",  # 2026-04-29 Will Faust
    "a90e2448e6d84f008eab3cd783af75aba3e64f8a",  # 2026-04-29 Will Faust
    "2e7d471b37887c65d75435c204044644efa60058",  # 2026-04-29 Will Faust
    "16cae730631b213e07bc1c40d6a2e185d9f251ae",  # 2026-04-29 Will Faust
    "993d6e1e03b496dd9ef47b057ca9fe9d6c43f4ea",  # 2026-04-29 Will Faust
    "f37a518d366b04377515c452a82ed74d32d304fc",  # 2026-04-29 Will Faust
    "763d4573dd0e0b45fb2ae963000c7c5cd4a22d94",  # 2026-04-29 Will Faust
    "a52945e491f62e6575ed44d0d084c0136afba4a2",  # 2026-04-29 Will Faust
    "82097d0ae2c19cec8f45965dd849c0a01872c864",  # 2026-04-30 Will Faust
    "83e12849f53363402721893250d5e9ca9dd0e7fd",  # 2026-04-30 Will Faust
    "0dd45c3a29cfe09434e1a84c8250b3ef9cc1e18f",  # 2026-05-07 Will Faust
    "87b40c220c88a10204569be3bd01649f2cd6a135",  # 2026-05-08 Will Faust
    "4cc0b431fab857f0782934aecd9447e21cf00144",  # 2026-05-08 Will Faust
    "2c8a3fa32fd9a2f183e3b04069824f78269bbccb",  # 2026-05-08 Will Faust
    "db4f32768ea137148ea0b20072a797690b5dbd8d",  # 2026-05-11 Will Faust
    "981b08522dccc459290496ceb9700d74da85a35f",  # 2026-05-11 Will Faust
    "48f7eb4130d6f108bf2e7d943d112504f03ceef6",  # 2026-05-13 Will Faust
    "8f0243b30303aaeacb4a7f6d7e96f61b6e4fb316",  # 2026-05-13 Will Faust
    "707f213f5a7aae3796f2f866e9fe991b35dd0ca8",  # 2026-05-18 Will Faust
    "61f11e3cc726a14418f2414e358ce7baa61a021b",  # 2026-07-03 Will Faust
    "3228ed587bcfe52e5ce4e0d53bd77c17c6b1a974",  # 2026-07-03 Will Faust
    "b648df3374ed5ac554c74ad8f84233f89f359fd8",  # 2026-07-04 Will Faust
    "6084de07629ab14bf5ed8c4af051c02920b5d357",  # 2026-07-06 Will Faust
    "c78110486b5efdb855f922cb15cf254b8e855479",  # 2026-07-08 Will Faust
    "412d12118bcccab437d4c372a2f88c4cf1dfd1bc",  # 2026-07-08 Will Faust
    "acb015c250fb57644b5aae1fa7d0b0ad89b4187b",  # 2026-07-09 Will Faust
    "fa15626253b73d59fde6632d645838aee823dc13",  # 2026-07-27 Will Faust
    "53ab9a995985c60cafd27837ada4ff02f965b2f5",  # 2026-07-28 Will Faust
    "fdf361f0ec3e0f86b8313a8c3fad360146a1bd90",  # 2026-07-28 Will Faust
    "8c0e1a87abaef7af70d8e9cb696b85525f6cce8d",  # 2026-07-29 Will Faust
    "ceabf254afbdc3c4c943d238f8c3082a13595028",  # 2026-07-30 Will Faust
    "49a7b55ec0b1a175782ba7fb4292f688412cffb4",  # 2026-07-31 Will Faust
    "e05e7b7f61412d7313346751de88c553c05a2b40",  # 2026-07-31 Will Faust
    "b64c4c252a3b8e9b1ecdbb099bb15569b6aefa17",  # 2026-07-31 Will Faust
    "13488326b534a362d738b16a61f1c15be3949b81",  # 2026-07-31 Will Faust
    "b8368fe0a2d5de9e364cf01b9b81d906895069a7",  # 2026-08-01 Will Faust
    "90a30158427309fc550fc9b3d88e19fc5a977067",  # 2026-08-02 Will Faust
    "156653c0680fc0c0ef36725b3cc72c18518c2a7e",  # 2026-08-04 Will Faust
    "019bda8c0f2043bd4a0b2436ae38d4f5ceb93d02",  # 2026-08-04 Will Faust
    "b9e9a9762ba822f394423942dfb8a2d80c922298",  # 2026-08-04 Will Faust
    "93c640cfbe7300831d9716ff1ab1ba5174c47132",  # 2026-08-05 Will Faust
    "2e6bbec5f19d68db081242f27d6405d2b5a7ae7d",  # 2026-08-08 Will Faust
    "d6d49752eb15da0806e1c8f0bbd09f5b83fbd713",  # 2026-08-11 Will Faust
    "43659eb9fd4227bc42f9ae3f714b861f1f1ab237",  # 2026-08-12 Will Faust
    "4cc5afe2bf35edc1f7adf256bb48ed6e3d800c3e",  # 2026-08-12 Will Faust
    "04cbb90c715519136da771af3cc8f1dac9b821a6",  # 2026-08-14 Will Faust
    "c77105fa349c87f307f1256b1fe38e2fc2e25196",  # 2026-08-16 Will Faust
    "d21da240b746868cdc388617c3cb15c529b23f7f",  # 2026-08-17 Will Faust
    "b4bedfa60800cea8a2ad415f6e2e09390497ca49",  # 2026-08-21 Will Faust
    "ac555dd81fa84f86d1927476e152451005e78f49",  # 2026-08-25 Will Faust
    "d996d07ec9fefbe16a9f801ed4b27ff20be87567",  # 2026-10-09 Arvind Frøiland
    "52aa8a487af624ddb1963b48f5e6c73c85fa6dee",  # 2026-10-09 Arvind Frøiland
    "e941d55cfb2efa189c48ed34cb97523ed9932f2a",  # 2026-10-09 Arvind Frøiland
    "c254600b3ed35c1d1b1282ba5b8cc9182bcd3ba8",  # 2026-10-09 Arvind Frøiland
    "6eac57d5fb2f49dbefd7e95521828012588c2506",  # 2026-10-09 Arvind Frøiland
    "cad0c0969354e4f51462c5bf25f0f1acd5edce16",  # 2026-10-09 Arvind Frøiland
}

# Submodules of the vendored tree, at the commits FEX-2610 records, audited 2026-10-12.
AUDITED_NESTED = {
    f"{VENDORED}/External/fmt": {"c07e2aa4b130b4fd359a1c6455a5201b17af01fa"},
    f"{VENDORED}/External/range-v3": {"ca1388fb9da8e69314dda222dc7b139ca84e092f"},
    f"{VENDORED}/External/unordered_dense": {"3234af2c03549bc85656bfd3a86993bf1cd8aef1"},
    f"{VENDORED}/External/xxhash": {"e626a72bc2321cd320e953a0ccf1584cad60f363"},
}

WILL_FAUST = re.compile(r"will\s*faust|willfaust|willtechnoduck", re.I)
# Contributors to willfaust/FEX after the GPL-3.0 boundary (fex-licence.md section 2).
GPL3_CONTRIBUTORS = re.compile(r"125hz|spitefulowl|bahacan16", re.I)
PORTED_FROM = re.compile(r"ported from arvindfroi/FEX commit ([0-9a-f]{40})")
GDB_SYMBOLS_OFF = re.compile(r"^\s*-DENABLE_GDB_SYMBOLS(:BOOL)?=(OFF|FALSE|NO|0)\s*$", re.I | re.M)
SUBTREE_DIR = re.compile(rf"^git-subtree-dir: {re.escape(VENDORED)}/?$", re.M)
SUBTREE_SPLIT = re.compile(r"^git-subtree-split: ([0-9a-f]{40})$", re.M)

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


def entry_mode(root, revision):
    listing = git(root, "ls-tree", revision, "--", VENDORED)
    return listing.split()[0] if listing else None


def recorded_base(root):
    log = git(root, "log", "--format=%H%x00%B%x1e", "-E", f"--grep=^git-subtree-dir: {VENDORED}/?$", "HEAD")
    for record in (log or "").split("\x1e"):
        commit, _, message = record.strip().partition("\x00")
        if commit and SUBTREE_DIR.search(message):
            split = SUBTREE_SPLIT.search(message)
            if split:
                return commit, split.group(1)
    return None, None


def series(root):
    log = git(root, "log", "--no-merges", "--format=%H%x00%an%x00%ae%x00%B%x1e", "HEAD", "--", VENDORED)
    for record in (log or "").split("\x1e"):
        fields = record.strip().split("\x00", 3)
        if len(fields) == 4:
            yield fields


def gitlinks(root):
    found = {}
    for arguments, field in ((("ls-tree", "-r", "HEAD", "--", VENDORED), 2), (("ls-files", "-s", "--", VENDORED), 1)):
        for line in (git(root, *arguments) or "").splitlines():
            columns, _, path = line.partition("\t")
            columns = columns.split()
            if columns and columns[0] == "160000":
                found.setdefault(path, set()).add(columns[field])
    return found


def tracked_branches(root):
    branches = {}
    paths = git(root, "config", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$") or ""
    for line in paths.splitlines():
        key, _, value = line.partition(" ")
        value = value.strip()
        if value == VENDORED or value.startswith(VENDORED + "/"):
            name = key[len("submodule."):-len(".path")]
            branch = git(root, "config", "--file", ".gitmodules", "--get", f"submodule.{name}.branch")
            if branch:
                branches[value] = branch
    return branches


def check(root, bases=AUDITED_BASES, sources=AUDITED_SOURCES, nested=AUDITED_NESTED):
    root = Path(root)
    findings = []
    tree = root / VENDORED
    if not (tree / "CMakeLists.txt").is_file():
        return [f"{VENDORED} is not checked out"], 0, None
    mode = entry_mode(root, "HEAD")
    if mode == "160000":
        findings.append(f"{VENDORED} is a submodule in HEAD; D13 vendors FEX-Emu FEX-2610 with git subtree")
    elif mode != "040000":
        findings.append(f"cannot read {VENDORED} from HEAD in {root}")
    squash, base = recorded_base(root)
    if base is None:
        findings.append(f"no git-subtree-split recorded for {VENDORED}; add FEX with `git subtree add --squash`")
    elif base not in bases:
        findings.append(f"{VENDORED} upstream base {base} (recorded by {squash}) is not audited; "
                        "audit FEX-Emu's history up to it and add it to AUDITED_BASES")
    ported = 0
    for commit, name, email, message in series(root):
        author = f"{name} <{email}>"
        if GPL3_CONTRIBUTORS.search(author):
            findings.append(f"commit {commit} by {author} changes {VENDORED}; "
                            "willfaust/FEX contributors after 188e8224 are GPL-3.0")
            continue
        origin = PORTED_FROM.search(message)
        if origin:
            if origin.group(1) not in sources:
                findings.append(f"commit {commit} is ported from {origin.group(1)}, which is not in AUDITED_SOURCES")
                continue
            ported += 1
        elif WILL_FAUST.search(author):
            findings.append(f"commit {commit} by {author} changes {VENDORED} without naming an audited source commit")
    for path, commits in sorted(gitlinks(root).items()):
        for commit in sorted(commits - nested.get(path, set())):
            findings.append(f"submodule {path} at {commit} is not audited; add it to AUDITED_NESTED after auditing it")
    for path, branch in sorted(tracked_branches(root).items()):
        findings.append(f".gitmodules tracks branch {branch} for {path}; remove it so the pin cannot drift")
    runner = root / RUNNER_CMAKE
    if runner.is_file() and not GDB_SYMBOLS_OFF.search(runner.read_text(encoding="utf-8", errors="replace")):
        findings.append(f"{RUNNER_CMAKE} does not pass -DENABLE_GDB_SYMBOLS=OFF; FEX would include GDB's GPL-3.0 "
                        "<gdb/jit-reader.h> wherever it is installed")
    scanned, count = scan(tree)
    findings.extend(f"{VENDORED}/{finding}" for finding in scanned)
    summary = None
    if base in bases:
        summary = f"{VENDORED}: FEX-Emu {base} ({bases[base]}) + {ported} ported commits"
    return findings, count, summary


def main(arguments):
    root = Path(arguments[0]) if arguments else Path(__file__).resolve().parents[2]
    findings, count, summary = check(root)
    for finding in findings:
        print(f"error: {finding}")
    if findings:
        return 1
    print(f"{summary}; {count} files scanned, no GPL-3.0 markers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
