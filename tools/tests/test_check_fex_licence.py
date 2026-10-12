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


def superproject(root, branch=None):
    fex = Path(root) / check_fex_licence.SUBMODULE
    write(fex, "CMakeLists.txt", "project(FEX)\n")
    write(fex, "LICENSE", MIT)
    git(fex, "init", "-q")
    pinned = commit(fex, "fex")
    modules = f'[submodule "{check_fex_licence.SUBMODULE}"]\n\tpath = {check_fex_licence.SUBMODULE}\n\turl = https://example.invalid/FEX.git\n'
    if branch:
        modules += f"\tbranch = {branch}\n"
    write(root, ".gitmodules", modules)
    git(root, "init", "-q")
    commit(root, "superproject")
    return fex, pinned


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


class PinTests(unittest.TestCase):
    def test_audited_pin_passes(self):
        with tempfile.TemporaryDirectory() as root:
            _, pinned = superproject(root)
            findings, count = check_fex_licence.check(Path(root), {pinned: "test"})
            self.assertEqual(findings, [])
            self.assertEqual(count, 2)

    def test_unaudited_pin_fails_for_each_place_it_is_read(self):
        with tempfile.TemporaryDirectory() as root:
            _, pinned = superproject(root)
            findings, _ = check_fex_licence.check(Path(root), {})
            self.assertEqual(len(findings), 3)
            for where in ("recorded in HEAD", "staged in the index", "checked out"):
                self.assertIn(f"commit {pinned} ({where}) is not audited", "\n".join(findings))

    def test_checkout_moved_away_from_the_pin_fails(self):
        with tempfile.TemporaryDirectory() as root:
            fex, pinned = superproject(root)
            write(fex, "FEXCore/new.cpp", "// SPDX-License-Identifier: MIT\n")
            moved = commit(fex, "unaudited")
            findings, _ = check_fex_licence.check(Path(root), {pinned: "test"})
            self.assertEqual(findings, [f"{check_fex_licence.SUBMODULE} commit {moved} (checked out) is not audited; "
                                        "audit its history and add it to AUDITED"])

    def test_tracked_branch_fails(self):
        with tempfile.TemporaryDirectory() as root:
            _, pinned = superproject(root, branch="anyps5-macos")
            findings, _ = check_fex_licence.check(Path(root), {pinned: "test"})
            self.assertEqual(findings, [f".gitmodules tracks branch anyps5-macos for {check_fex_licence.SUBMODULE}; "
                                        "remove it so the pin cannot drift"])

    def test_marker_in_an_audited_checkout_fails(self):
        with tempfile.TemporaryDirectory() as root:
            fex, pinned = superproject(root)
            write(fex, "LICENSE-MADEIRA.md", "")
            findings, _ = check_fex_licence.check(Path(root), {pinned: "test"})
            self.assertEqual(findings, [f"{check_fex_licence.SUBMODULE}/LICENSE-MADEIRA.md: GPL-3.0 marker file"])

    def test_missing_checkout_fails(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(check_fex_licence.check(Path(root)),
                             ([f"{check_fex_licence.SUBMODULE} is not checked out"], 0))


if __name__ == "__main__":
    unittest.main()
