"""check_no_x86.py on small translation units preprocessed for arm64: what it reports and its exit status."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parent / "check_no_x86.py"
COMPILER = os.environ.get("CXX") or shutil.which("clang++")

FLAGGED = {
    "header.cpp": "#include <immintrin.h>\nint f() { return 0; }\n",
    "intrinsic.cpp": "unsigned f() { return _mm_getcsr(); }\n",
    "cpu.cpp": "bool f() { return __builtin_cpu_supports(\"avx2\"); }\n",
    "assembly.cpp": "asm(\".text\\n\"\n    \"f:\\n\"\n    \"    movq %rdi, %rax\\n\"\n    \"    ret\\n\");\n",
    "clobber.cpp": "void f() { asm volatile(\"\" ::: \"xmm0\"); }\n",
    "pause.cpp": "void f() { asm volatile(\"pause\" ::: \"memory\"); }\n",
    "trap.cpp": "void f() { asm volatile(\"int3\"); }\n",
    "file.S": ".text\nf:\n    movq %rdi, %rax\n    ret\n",
}

CLEAN = {
    "guarded.cpp": "#if defined(__x86_64__)\n#include <immintrin.h>\nvoid f() { asm(\"movq %rdi, %rax\"); _mm_pause(); }\n#endif\n",
    "neutral.cpp": "void f() { asm volatile(\"\" ::: \"memory\"); asm volatile(\"yield\"); asm volatile(\"dmb ish\" ::: \"memory\"); }\n"
                   "extern \"C\" int g() asm(\"renamed\");\n",
    "arm64.S": ".text\nf:\n    stp x29, x30, [sp, #-16]!\n    ldp x29, x30, [sp], #16\n    ret\n",
}

BROKEN = {
    "broken.cpp": "#include \"missing.hpp\"\n",
}


@unittest.skipUnless(COMPILER, "no clang++ to preprocess for arm64")
class CheckNoX86Tests(unittest.TestCase):
    def run_tool(self, sources):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            commands = []
            for name, text in sources.items():
                (root / name).write_text(text)
                commands.append({"directory": str(root), "file": str(root / name),
                                 "command": f"{COMPILER} --target=aarch64-unknown-linux-gnu -std=c++20 -o {name}.o -c {root / name}"})
            build = root / "build"
            build.mkdir()
            (build / "compile_commands.json").write_text(json.dumps(commands))
            result = subprocess.run([sys.executable, "-I", str(TOOL), str(build), "--repo", str(root), "--jobs", "2"],
                                    capture_output=True, text=True, timeout=120)
            return result.returncode, result.stdout.replace(str(root) + os.sep, ""), result.stderr

    def test_reports_every_x86_construct(self):
        status, output, _ = self.run_tool({**FLAGGED, **CLEAN})
        self.assertEqual(status, 1, output)
        for name in FLAGGED:
            with self.subTest(name=name):
                self.assertRegex(output, rf"(?m)^{name}:\d+: ")
        for name in CLEAN:
            with self.subTest(name=name):
                self.assertNotRegex(output, rf"(?m)^{name}:")
        self.assertIn(f"{len(FLAGGED) + len(CLEAN)} translation unit(s) scanned", output)

    def test_clean_code_passes(self):
        status, output, _ = self.run_tool(CLEAN)
        self.assertEqual(status, 0, output)
        self.assertIn("3 translation unit(s) scanned, 0 finding(s), 0 not preprocessed", output)

    def test_unpreprocessed_unit_fails(self):
        status, output, errors = self.run_tool({**CLEAN, **BROKEN})
        self.assertEqual(status, 2, output)
        self.assertIn("cannot preprocess broken.cpp", errors)


if __name__ == "__main__":
    unittest.main()
