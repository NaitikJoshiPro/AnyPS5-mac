import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mac"))
import check_fex_licence

GIT_ENVIRONMENT = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
                       GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
                       GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid")

MIT = "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
GPL2 = "                    GNU GENERAL PUBLIC LICENSE\n                       Version 2, June 1991\n"
GPL3 = "                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n"
LGPL3 = "                   GNU LESSER GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n"


def write(root, relative, text):
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), "-c", "init.defaultBranch=main", *arguments],
                          env=GIT_ENVIRONMENT, capture_output=True, text=True, check=True).stdout.strip()


def commit(directory, message):
    git(directory, "add", "-A")
    git(directory, "commit", "-q", "--no-verify", "--no-gpg-sign", "-m", message)
    return git(directory, "rev-parse", "HEAD")


BASE = "14c92681f4d62cf84d901460e0358de09c8847a7"
SOURCE = "ac555dd81fa84f86d1927476e152451005e78f49"
FMT = "c07e2aa4b130b4fd359a1c6455a5201b17af01fa"
VENDORED = check_fex_licence.VENDORED


def vendored(root, base=BASE):
    """A repository with FEX vendored the way `git subtree add --squash` records it."""
    git(root, "init", "-q")
    write(root, "README.md", "AnyPS5\n")
    commit(root, "superproject")
    write(Path(root) / VENDORED, "CMakeLists.txt", "project(FEX)\n")
    write(Path(root) / VENDORED, "LICENSE", MIT)
    commit(root, f"Squashed '{VENDORED}/' content from commit {base[:9]}\n\n"
                 f"git-subtree-dir: {VENDORED}\ngit-subtree-split: {base}\n")


def commit_as(root, author, message, relative="FEXCore/x.cpp"):
    name, _, email = author.partition(" <")
    write(Path(root) / VENDORED, relative, f"// SPDX-License-Identifier: MIT\n// {message}\n")
    git(root, "add", "-A")
    subprocess.run(["git", "-C", str(root), "commit", "-q", "--no-verify", "--no-gpg-sign", "--author",
                    f"{name} <{email.rstrip('>')}>", "-m", message], env=GIT_ENVIRONMENT, check=True)
    return git(root, "rev-parse", "HEAD")


def add_gitlink(root, path, sha, branch=None):
    modules = Path(root) / ".gitmodules"
    text = modules.read_text() if modules.exists() else ""
    text += f'[submodule "{path}"]\n\tpath = {path}\n\turl = https://example.invalid/x.git\n'
    if branch:
        text += f"\tbranch = {branch}\n"
    modules.write_text(text)
    git(root, "update-index", "--add", "--cacheinfo", f"160000,{sha},{path}")
    git(root, "add", ".gitmodules")
    git(root, "commit", "-q", "--no-verify", "--no-gpg-sign", "-m", f"add {path}")


class ScanTests(unittest.TestCase):
    def test_permissive_and_gpl2_compatible_identifiers_pass(self):
        with tempfile.TemporaryDirectory() as tree:
            write(tree, "LICENSE", MIT)
            write(tree, "External/xxhash/cli/COPYING", GPL2)
            write(tree, "FEXCore/a.cpp", "// SPDX-License-Identifier: MIT\n")
            write(tree, "FEXCore/b.h", "/* SPDX-License-Identifier: GPL-2.0-only */\n")
            write(tree, "Source/Windows/c.h", "// SPDX-License-Identifier: LGPL-2.1-or-later\n")
            write(tree, "External/d.h", "// SPDX-License-Identifier: (MIT OR CC0-1.0)\n")
            write(tree, "External/e.cpp", "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n")
            write(tree, "FEXCore/f.h", "/* iOS-Madeira 2026-05-18: callret_sp_base mirrored */\n")
            self.assertEqual(check_fex_licence.scan(Path(tree)), ([], 8))

    def test_willfaust_marker_files_fail_anywhere(self):
        with tempfile.TemporaryDirectory() as tree:
            write(tree, "LICENSE-MADEIRA.md", "Only contributions made from 2026-08-28 onward are GPL-only\n")
            write(tree, "Data/COPYING.GPL-3.0", "")
            findings, _ = check_fex_licence.scan(Path(tree))
            self.assertEqual(findings, ["Data/COPYING.GPL-3.0: GPL-3.0 marker file",
                                        "LICENSE-MADEIRA.md: GPL-3.0 marker file"])

    def test_version_3_spdx_identifiers_fail(self):
        identifiers = ["GPL-3.0", "GPL-3.0-only", "GPL-3.0-or-later", "GPL-3.0+", "LGPL-3.0-only",
                       "AGPL-3.0-or-later", "GPL-3.0-only WITH GCC-exception-3.1"]
        with tempfile.TemporaryDirectory() as tree:
            for index, identifier in enumerate(identifiers):
                write(tree, f"FEXCore/{index}.cpp", f"// SPDX-License-Identifier: {identifier}\n")
            write(tree, "FEXCore/dual.cpp", "// SPDX-License-Identifier: MIT OR GPL-3.0-only\n")
            findings, count = check_fex_licence.scan(Path(tree))
            self.assertEqual(count, len(identifiers) + 1)
            self.assertEqual(len(findings), len(identifiers) + 1)
            self.assertIn("FEXCore/dual.cpp: SPDX identifier GPL-3.0-only", findings)
            self.assertIn("FEXCore/3.cpp: SPDX identifier GPL-3.0+", findings)

    def test_version_3_licence_texts_and_notices_fail(self):
        with tempfile.TemporaryDirectory() as tree:
            write(tree, "LICENSE", GPL3)
            write(tree, "External/x/COPYING.LESSER", LGPL3)
            write(tree, "FEXCore/notice.cpp", " * the Free Software Foundation, either version 3 of the License, or\n")
            findings, _ = check_fex_licence.scan(Path(tree))
            self.assertEqual(findings, ["External/x/COPYING.LESSER: GPL, LGPL or AGPL version 3 licence text",
                                        "FEXCore/notice.cpp: \"version 3 of the License\" notice",
                                        "LICENSE: GPL, LGPL or AGPL version 3 licence text"])

    def test_git_metadata_is_not_scanned(self):
        with tempfile.TemporaryDirectory() as tree:
            write(tree, ".git/COPYING.GPL-3.0", GPL3)
            write(tree, "External/fmt/.git", "gitdir: ../../.git/modules/External/fmt\n")
            write(tree, "LICENSE", MIT)
            self.assertEqual(check_fex_licence.scan(Path(tree)), ([], 1))


class VendoredTests(unittest.TestCase):
    def check(self, root):
        return check_fex_licence.check(Path(root), {BASE: "test base"}, {SOURCE}, {f"{VENDORED}/External/fmt": {FMT}})

    def test_audited_base_and_series_pass(self):
        with tempfile.TemporaryDirectory() as root:
            vendored(root)
            commit_as(root, "Will Faust <willtechnoduck@gmail.com>",
                      f"iOS port\n\n(ported from arvindfroi/FEX commit {SOURCE}\n to FEX-Emu FEX-2610)")
            commit_as(root, "Arvind Fr\u00f8iland <arvind@example.invalid>", "Apple: keep the JIT off x18", "a.cpp")
            commit_as(root, "t <t@example.invalid>", "fix(fex): our own change", "b.cpp")
            add_gitlink(root, f"{VENDORED}/External/fmt", FMT)
            findings, count, summary = self.check(root)
            self.assertEqual(findings, [])
            self.assertEqual(count, 5)
            self.assertEqual(summary, f"{VENDORED}: FEX-Emu {BASE} (test base) + 1 ported commits")

    def test_unaudited_base_fails(self):
        with tempfile.TemporaryDirectory() as root:
            vendored(root, base="0" * 40)
            findings, _, _ = self.check(root)
            self.assertEqual(len(findings), 1)
            self.assertIn(f"upstream base {'0' * 40}", findings[0])

    def test_latest_subtree_record_wins(self):
        with tempfile.TemporaryDirectory() as root:
            vendored(root)
            write(Path(root) / VENDORED, "Readme.md", "FEX\n")
            commit(root, f"Squashed '{VENDORED}/' changes from {BASE[:9]}..{'1' * 9}\n\n"
                         f"git-subtree-dir: {VENDORED}\ngit-subtree-split: {'1' * 40}\n")
            findings, _, _ = self.check(root)
            self.assertEqual(len(findings), 1)
            self.assertIn(f"upstream base {'1' * 40}", findings[0])

    def test_missing_subtree_record_fails(self):
        with tempfile.TemporaryDirectory() as root:
            git(root, "init", "-q")
            write(Path(root) / VENDORED, "CMakeLists.txt", "project(FEX)\n")
            commit(root, "copied FEX by hand")
            findings, _, _ = self.check(root)
            self.assertEqual(findings, [f"no git-subtree-split recorded for {VENDORED}; add FEX with `git subtree add --squash`"])

    def test_submodule_layout_fails(self):
        with tempfile.TemporaryDirectory() as root:
            fex = Path(root) / VENDORED
            write(fex, "CMakeLists.txt", "project(FEX)\n")
            git(fex, "init", "-q")
            commit(fex, "fex")
            git(root, "init", "-q")
            commit(root, f"superproject\n\ngit-subtree-dir: {VENDORED}\ngit-subtree-split: {BASE}\n")
            findings, _, _ = self.check(root)
            self.assertIn(f"{VENDORED} is a submodule in HEAD; D13 vendors FEX-Emu FEX-2610 with git subtree", findings)

    def test_will_faust_commit_without_audited_source_fails(self):
        with tempfile.TemporaryDirectory() as root:
            vendored(root)
            plain = commit_as(root, "Will Faust <willtechnoduck@gmail.com>", "Madeira: something new")
            other = commit_as(root, "Will Faust <willtechnoduck@gmail.com>",
                              f"later work\n\n(ported from arvindfroi/FEX commit {'2' * 40})", "c.cpp")
            findings, _, _ = self.check(root)
            self.assertEqual(findings, [
                f"commit {other} is ported from {'2' * 40}, which is not in AUDITED_SOURCES",
                f"commit {plain} by Will Faust <willtechnoduck@gmail.com> changes {VENDORED} "
                "without naming an audited source commit"])

    def test_gpl3_period_contributor_fails_even_with_a_source(self):
        with tempfile.TemporaryDirectory() as root:
            vendored(root)
            bad = commit_as(root, "spitefulowl <owl@example.invalid>",
                            f"FEXCore/iOS: diagnostics\n\n(ported from arvindfroi/FEX commit {SOURCE})")
            findings, _, _ = self.check(root)
            self.assertEqual(findings, [f"commit {bad} by spitefulowl <owl@example.invalid> changes {VENDORED}; "
                                        "willfaust/FEX contributors after 188e8224 are GPL-3.0"])

    def test_unaudited_or_tracking_nested_submodule_fails(self):
        with tempfile.TemporaryDirectory() as root:
            vendored(root)
            add_gitlink(root, f"{VENDORED}/External/fmt", "3" * 40)
            add_gitlink(root, f"{VENDORED}/External/rpmalloc", "4" * 40, branch="ios-madeira")
            findings, _, _ = self.check(root)
            self.assertEqual(findings, [
                f"submodule {VENDORED}/External/fmt at {'3' * 40} is not audited; add it to AUDITED_NESTED after auditing it",
                f"submodule {VENDORED}/External/rpmalloc at {'4' * 40} is not audited; add it to AUDITED_NESTED after auditing it",
                f".gitmodules tracks branch ios-madeira for {VENDORED}/External/rpmalloc; remove it so the pin cannot drift"])

    def test_marker_in_the_vendored_tree_fails(self):
        with tempfile.TemporaryDirectory() as root:
            vendored(root)
            write(Path(root) / VENDORED, "LICENSE-MADEIRA.md", "")
            findings, _, _ = self.check(root)
            self.assertEqual(findings, [f"{VENDORED}/LICENSE-MADEIRA.md: GPL-3.0 marker file"])

    def test_runner_must_switch_off_gdb_symbols(self):
        with tempfile.TemporaryDirectory() as root:
            vendored(root)
            write(root, check_fex_licence.RUNNER_CMAKE, "ExternalProject_Add(fexcore\n    CMAKE_ARGS\n        -DENABLE_LTO=OFF)\n")
            findings, _, _ = self.check(root)
            self.assertEqual(findings, [f"{check_fex_licence.RUNNER_CMAKE} does not pass -DENABLE_GDB_SYMBOLS=OFF; FEX would "
                                        "include GDB's GPL-3.0 <gdb/jit-reader.h> wherever it is installed"])
            write(root, check_fex_licence.RUNNER_CMAKE,
                  "ExternalProject_Add(fexcore\n    CMAKE_ARGS\n        -DENABLE_GDB_SYMBOLS=OFF\n        -DENABLE_LTO=OFF)\n")
            self.assertEqual(self.check(root)[0], [])

    def test_missing_tree_fails(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(check_fex_licence.check(Path(root)), ([f"{VENDORED} is not checked out"], 0, None))


if __name__ == "__main__":
    unittest.main()
