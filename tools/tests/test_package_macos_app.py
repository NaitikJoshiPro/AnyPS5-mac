import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_macos_app


class PackageMacosAppTests(unittest.TestCase):
    def test_guest_modules_resolve_from_the_executable_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relinked = root / "relinked"
            (relinked / "app0" / "sce_module").mkdir(parents=True)
            (relinked / "eboot").write_text('#!/bin/sh\npwd -P > "$HOME/where"\ncat app0/sce_sys/param.json > "$HOME/param"\ntest -d download0 && exit 7\nexit 1\n')
            (relinked / "app0" / "sce_module" / "libc.prx.guest.prx").write_bytes(b"module")
            game = root / "game"
            (game / "sce_sys").mkdir(parents=True)
            (game / "sce_sys" / "param.json").write_text(json.dumps({"titleId": "PPSA00000"}))
            libs = root / "libs"
            libs.mkdir()
            (libs / "libc.prx").write_bytes(b"prx")
            vulkan = root / "vulkan"
            (vulkan / "lib").mkdir(parents=True)
            (vulkan / "lib" / "libvulkan.1.dylib").write_bytes(b"loader")
            (vulkan / "lib" / "libMoltenVK.dylib").write_bytes(b"driver")
            (vulkan / "share" / "vulkan" / "icd.d").mkdir(parents=True)
            (vulkan / "share" / "vulkan" / "icd.d" / "MoltenVK_icd.json").write_text(json.dumps({"ICD": {"library_path": "x"}}))
            bundle = root / "Title.app"

            package_macos_app.package(relinked, game, libs, vulkan, None, bundle)

            module = bundle / "Contents" / "MacOS" / "app0" / "sce_module" / "libc.prx.guest.prx"
            self.assertEqual(module.read_bytes(), b"module")
            self.assertEqual(module.resolve(), (bundle / "Contents" / "Resources" / "game" / "app0" / "sce_module" / "libc.prx.guest.prx").resolve())
            self.assertTrue((bundle / "Contents" / "MacOS" / "app0").is_symlink())
            self.assertFalse((bundle / "Contents" / "MacOS" / "app0").readlink().is_absolute())

            home = root / "home"
            home.mkdir()
            launched = subprocess.run([str(bundle / "Contents" / "MacOS" / "launch")], env={**os.environ, "HOME": str(home)}, capture_output=True, timeout=30)
            self.assertEqual(launched.returncode, 7, launched.stderr)
            run = home / "Library" / "Application Support" / "AnyPS5" / "PPSA00000"
            self.assertEqual(Path((home / "where").read_text().strip()), run.resolve())
            self.assertEqual(json.loads((home / "param").read_text()), {"titleId": "PPSA00000"})
            self.assertEqual(len(list((bundle / "Contents" / "Resources" / "game").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
