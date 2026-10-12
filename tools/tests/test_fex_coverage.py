import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mac"))
import fex_coverage


def module(file_id, text, jit=(), hit=(), **extra):
    return {"path": f"/t/{file_id}.elf", "file_id": file_id, "size": 0x10000, "text": [list(r) for r in text],
            "text_bytes": sum(size for _, size in text), "jit_blocks": len(jit), "cache_hit_blocks": len(hit),
            "jit": [list(r) for r in jit], "cache_hit": [list(r) for r in hit], **extra}


def report(mode, modules):
    return {"version": 1, "mode": mode, "blocks_outside_modules": {"jit": 0, "cache_hit": 0}, "modules": modules}


class Summary(unittest.TestCase):
    def test_static_and_run(self):
        precompile = report("precompile", [module("a", [(0x1000, 0x100)], jit=[(0x1000, 0x40), (0x1020, 0x40)],
                                                  seeds={"total": 2}, fde=[(0x1000, 0x80), (0x10f0, 0x100)])])
        run = report("run", [module("a", [(0x1000, 0x100)], jit=[(0x1080, 0x20)], hit=[(0x1000, 0x30), (0x1050, 0x10)])])
        binary = fex_coverage.summarise(precompile, run)["binaries"][0]
        self.assertEqual(binary["text_bytes"], 0x100)
        self.assertEqual(binary["fde_bytes"], 0x90)        # the second FDE is clipped to the text
        self.assertEqual(binary["aot_bytes"], 0x60)        # [0x1000, 0x1060) merged
        self.assertEqual(binary["aot_pct"], 37.5)
        self.assertEqual(binary["executed_bytes"], 0x60)   # 0x30 + 0x10 + 0x20
        self.assertEqual(binary["executed_aot_bytes"], 0x40)
        self.assertEqual(binary["jit_only_bytes"], 0x20)
        self.assertEqual(binary["jit_only_pct"], 33.3)
        self.assertEqual(binary["run_jit_blocks"], 1)

    def test_modules_match_by_file_id(self):
        precompile = report("precompile", [module("a", [(0, 0x10)]), module("b", [(0, 0x10)], jit=[(0, 0x10)])])
        run = report("run", [module("b", [(0, 0x10)], hit=[(0, 0x8)])])
        a, b = fex_coverage.summarise(precompile, run)["binaries"]
        self.assertNotIn("executed_bytes", a)
        self.assertEqual(b["executed_aot_pct"], 100.0)

    def test_schema_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            path.write_text(json.dumps(report("run", [module("a", [(0, 1)])])))
            with self.assertRaises(ValueError):
                fex_coverage.load(path, "precompile")
            broken = report("precompile", [module("a", [(0, 1)])])
            del broken["modules"][0]["text"]
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                fex_coverage.load(path, "precompile")
            path.write_text(json.dumps({**report("precompile", []), "version": 2}))
            with self.assertRaises(ValueError):
                fex_coverage.load(path, "precompile")


if __name__ == "__main__":
    unittest.main()
