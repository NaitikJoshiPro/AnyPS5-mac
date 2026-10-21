import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gen_bridge_signatures.py"
SOURCES = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(TOOL.parent))
import gen_bridge_signatures as generator

POINTER = {"kind": "pointer", "size": 8, "align": 8}
FLOAT = {"kind": "float", "size": 4, "align": 4}
DOUBLE = {"kind": "double", "size": 8, "align": 8}
LONG_DOUBLE = {"kind": "long double", "size": 8, "align": 8}
X87 = {"kind": "x87", "size": 16, "align": 16}
INT128 = {"kind": "int128", "size": 16, "align": 16, "signed": False}
VOID = {"kind": "void"}
EMPTY = {"kind": "empty", "size": 1, "align": 1}
APPLE = sys.platform == "darwin" and shutil.which("xcrun") is not None


def integer(size, signed=False):
    return {"kind": "int", "size": size, "align": size, "signed": signed}


def record(size, align, leaves, indirect=False):
    return {"kind": "record", "size": size, "align": align, "indirect": indirect, "union": False,
            "leaves": [[offset, kind, length, length] for offset, kind, length in leaves]}


def signature(result, *parameters, variadic=False):
    return {"variadic": variadic, "result": result, "parameters": list(parameters)}


def reasons(result, *parameters, variadic=False):
    return generator.classify(signature(result, *parameters, variadic=variadic))[2]


POSITION = record(12, 4, [(0, "float", 4), (4, "float", 4), (8, "float", 4)])
WIDE = record(24, 8, [(0, "int", 8), (8, "int", 8), (16, "int", 8)])
OWNER = record(8, 8, [(0, "int", 8)], indirect=True)


class ModelTests(unittest.TestCase):
    def test_positional_shapes(self):
        self.assertEqual(reasons(integer(8), POINTER), [])
        self.assertEqual(reasons(VOID, *[POINTER] * 8, integer(1)), [])
        self.assertEqual(reasons(VOID, *[POINTER] * 11), [])
        self.assertEqual(reasons(record(16, 8, [(0, "int", 8), (8, "int", 8)]), integer(8), integer(8)), [])
        self.assertEqual(reasons(record(16, 8, [(0, "double", 8), (8, "double", 8)]), record(16, 8, [(0, "double", 8), (8, "double", 8)])), [])
        self.assertEqual(reasons(VOID, record(4, 4, [(0, "float", 4)]), FLOAT), [])
        self.assertEqual(reasons(VOID, OWNER, POINTER), [])
        self.assertEqual(reasons(INT128, INT128, INT128), [])
        self.assertEqual(reasons(VOID, integer(4), EMPTY, integer(4)), [])

    def test_stack_arguments(self):
        release = [POINTER, integer(1), integer(2), integer(1), integer(1), POINTER, integer(1), integer(8), integer(2), integer(2), integer(1), integer(4)]
        self.assertEqual(reasons(POINTER, *release), ["stack-layout"])
        self.assertEqual(reasons(VOID, *[POINTER] * 8, integer(1), integer(1)), ["stack-layout"])
        self.assertEqual(reasons(VOID, *[DOUBLE] * 9), ["stack-layout"])
        self.assertEqual(reasons(VOID, *[POINTER] * 17), ["stack-layout"])

    def test_structs(self):
        self.assertEqual(reasons(integer(4), POINTER, POSITION, FLOAT, POINTER), ["register-shift", "struct-registers"])
        self.assertEqual(reasons(VOID, record(8, 4, [(0, "float", 4), (4, "float", 4)])), ["struct-registers"])
        self.assertEqual(reasons(VOID, record(16, 8, [(0, "double", 8), (8, "int", 4)])), ["struct-registers"])
        self.assertEqual(reasons(VOID, WIDE), ["struct-memory"])
        self.assertEqual(reasons(WIDE), ["sret"])
        self.assertEqual(reasons(OWNER), ["sret"])
        self.assertEqual(reasons(OWNER, POINTER), ["register-shift", "sret"])
        self.assertEqual(reasons(POSITION), ["struct-result"])

    def test_x87_and_long_double(self):
        self.assertEqual(reasons(LONG_DOUBLE, LONG_DOUBLE), ["long-double"])
        self.assertEqual(reasons(LONG_DOUBLE, LONG_DOUBLE, POINTER), ["long-double"])
        self.assertEqual(reasons(X87, POINTER, POINTER), ["x87-result"])
        self.assertEqual(reasons(integer(4), X87), ["x87-argument"])

    def test_int128_and_variadic(self):
        # Apple's arm64 convention does not round __int128 up to an even register: x1:x2 like rsi:rdx.
        self.assertEqual(reasons(VOID, integer(4), INT128), [])
        # With one register left the guest moves it to the stack; arm64 still has x5:x6.
        self.assertEqual(reasons(VOID, *[integer(4)] * 5, INT128), ["int128"])
        self.assertEqual(reasons(integer(4), POINTER, variadic=True), ["variadic"])

    def test_layouts(self):
        guest, host, _ = generator.classify(signature(integer(4), POINTER, POSITION, FLOAT))
        self.assertEqual(guest["parameters"][1], [["xmm0+0", 0, 8, "value"], ["xmm1+0", 8, 4, "value"]])
        self.assertEqual(host["parameters"][1], [["v0+0", 0, 4, "value"], ["v1+0", 4, 4, "value"], ["v2+0", 8, 4, "value"]])
        guest, host, _ = generator.classify(signature(WIDE, WIDE))
        self.assertEqual((guest["hidden"], guest["parameters"][0]), (True, [["stack+0", 0, 24, "value"]]))
        self.assertEqual((host["hidden"], host["parameters"][0]), (True, [["x0", 0, 8, "reference"]]))
        guest, host, _ = generator.classify(signature(VOID, *[POINTER] * 8, integer(2), integer(2)))
        self.assertEqual([guest["parameters"][8], guest["parameters"][9]], [[["stack+16", 0, 2, "value"]], [["stack+24", 0, 2, "value"]]])
        self.assertEqual([host["parameters"][8], host["parameters"][9]], [[["stack+0", 0, 2, "value"]], [["stack+2", 0, 2, "value"]]])


HEADER = "\n".join([
    "#pragma once",
    "#include <cstdint>",
    "struct alignas(16) X87Extended { std::uint64_t significand; std::uint16_t signExponent; };",
    "using GuestLongDouble = X87Extended;",
    "struct Position { float x, y, z; };",
    "struct Owner { Owner(); Owner(const Owner&); ~Owner(); void* data; };",
    "struct Wide { long a, b, c; };",
    "typedef int (*Compare)(const void*, const void*);",
    "",
])

SOURCE = "\n".join([
    '#include "Probe.hpp"',
    "#include <cstddef>",
    "",
    "#define DEFINE_PROBE(name) int name(int value) { return value; }",
    "",
    'extern "C" {',
    "int ProbePlain_nid_postfix(const char* text) { return text != nullptr; }",
    "int ProbeDeclared_nid_postfix(int value);",
    "static int ProbeStatic(int value) { return value; }",
    '__attribute__((visibility("hidden"))) int ProbeHidden(int value) { return value; }',
    "int ProbeHost_nid_no_patch(int value) { return value; }",
    "int ProbeVariadic_nid_postfix(const char* format, ...) { return 0; }",
    "GuestLongDouble ProbeX87_nid_postfix(const char* text, char** end) { return {}; }",
    "long double ProbeLongDouble_nid_postfix(long double value) { return value; }",
    "Owner ProbeOwner_nid_postfix() { return {}; }",
    "Wide ProbeWide_nid_postfix(Wide value) { return value; }",
    "int ProbeCoefficients(void* handle, Position position, float spread) { return 0; }",
    "void ProbeSort_nid_postfix(void* base, std::size_t count, std::size_t size, Compare compare) {}",
    "void ProbeTable(void* table[]) {}",
    "int ProbeComment(int value) /* { */ { return value; }",
    "std::uint32_t* ProbeReleaseMem(void* buffer, std::uint8_t a, std::uint16_t b, std::uint8_t c, std::uint8_t d, const void* e,",
    "                               std::uint8_t f, std::uint64_t g, std::uint16_t h, std::uint16_t i, std::uint8_t j, std::uint32_t k) {",
    "    return nullptr;",
    "}",
    "DEFINE_PROBE(ProbeMacro)",
    "}",
    "",
    "namespace Probe {",
    'extern "C" int ProbeNamespaced(int value) { return value; }',
    "}",
    "",
    "int ProbeCpp(int value) { return value; }",
    "",
])

EXPECTED = ["ProbeCoefficients", "ProbeComment", "ProbeLongDouble_nid_postfix", "ProbeMacro", "ProbeNamespaced",
            "ProbeOwner_nid_postfix", "ProbePlain_nid_postfix", "ProbeReleaseMem", "ProbeSort_nid_postfix", "ProbeTable",
            "ProbeVariadic_nid_postfix", "ProbeWide_nid_postfix", "ProbeX87_nid_postfix"]


def run(*arguments):
    return subprocess.run([sys.executable, str(TOOL), *map(str, arguments)], capture_output=True, text=True)


@unittest.skipUnless(APPLE, "needs Apple clang and its libclang")
class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        library = self.root / "core" / "libs" / "prx" / "libprobe"
        (library / "tests").mkdir(parents=True)
        (library / "Probe.hpp").write_text(HEADER)
        (library / "Export.cpp").write_text(SOURCE)
        (library / "Internal.cpp").write_text('extern "C" int ProbeInternal(int value) { return value; }\n')
        (library / "tests" / "Test.cpp").write_text('extern "C" int ProbeTestOnly(int value) { return value; }\n')
        build = self.root / "build"
        build.mkdir()
        entries = []
        for name, define in (("Export.cpp", "-Dlibprobe_EXPORTS"), ("Internal.cpp", ""), ("tests/Test.cpp", "-Dlibprobe_EXPORTS")):
            output = f"core/libs/prx/libprobe/CMakeFiles/libprobe.dir/{name}.o"
            command = f"/usr/bin/c++ {define} -I{library} -std=gnu++20 -arch arm64 -o {output} -c {library / name}"
            entries.append({"directory": str(build), "command": command, "file": str(library / name), "output": output})
        self.commands = build / "compile_commands.json"
        self.commands.write_text(json.dumps(entries))
        self.json = build / "BridgeSignatures.json"
        self.inc = build / "BridgeSignatures.inc"

    def tearDown(self):
        self.directory.cleanup()

    def generate(self, *extra):
        result = run("generate", "--compile-commands", self.commands, "--root", self.root, "--cache", self.root / "build" / "cache",
                     "--json", self.json, "--inc", self.inc, "--jobs", 2, *extra)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout, json.loads(self.json.read_text())

    def test_exports_and_classification(self):
        output, document = self.generate()
        self.assertIn("(1 parsed, 0 cached)", output)
        exports = {export["name"]: export for export in document["exports"]}
        self.assertEqual(sorted(exports), EXPECTED)
        self.assertEqual((document["hostOnly"], document["problems"]), (1, []))
        self.assertEqual(exports["ProbeReleaseMem"]["reasons"], ["stack-layout"])
        self.assertEqual(exports["ProbeCoefficients"]["reasons"], ["register-shift", "struct-registers"])
        self.assertEqual(exports["ProbeOwner_nid_postfix"]["reasons"], ["sret"])
        self.assertTrue(exports["ProbeOwner_nid_postfix"]["result"]["indirect"])
        self.assertEqual(exports["ProbeWide_nid_postfix"]["reasons"], ["sret", "struct-memory"])
        self.assertEqual(exports["ProbeX87_nid_postfix"]["reasons"], ["x87-result"])
        self.assertEqual(exports["ProbeX87_nid_postfix"]["result"]["kind"], "x87")
        self.assertEqual(exports["ProbeLongDouble_nid_postfix"]["reasons"], ["long-double"])
        self.assertEqual((exports["ProbeVariadic_nid_postfix"]["variadic"], len(exports["ProbeVariadic_nid_postfix"]["parameters"])), (True, 1))
        self.assertEqual(exports["ProbeTable"]["parameters"][0]["kind"], "pointer")
        callback = exports["ProbeSort_nid_postfix"]["parameters"][3]["callback"]
        self.assertEqual((callback["result"]["kind"], [value["kind"] for value in callback["parameters"]], callback["reasons"]),
                         ("int", ["pointer", "pointer"], []))
        self.assertEqual([leaf[1] for leaf in exports["ProbeCoefficients"]["parameters"][1]["leaves"]], ["float"] * 3)
        self.assertEqual(exports["ProbeMacro"]["location"], "core/libs/prx/libprobe/Export.cpp:25")

    def test_cache_and_full_parse(self):
        self.generate()
        output, _ = self.generate()
        self.assertIn("(0 parsed, 1 cached)", output)
        header = self.root / "core" / "libs" / "prx" / "libprobe" / "Probe.hpp"
        header.write_text(HEADER + "struct Unused {};\n")
        output, _ = self.generate()
        self.assertIn("(1 parsed, 0 cached)", output)
        _, document = self.generate("--full-parse")
        self.assertEqual(sorted(export["name"] for export in document["exports"]), EXPECTED)

    def test_check(self):
        self.generate()
        bridge = self.root / "Bridge.cpp"

        def check(variadic, x87):
            bridge.write_text(f"constexpr VariadicExport VariadicExports[] = {{\n{variadic}\n}};\n"
                              f"constexpr std::string_view X87Results[] = {{{x87}}};\n")
            return run("check", "--json", self.json, "--bridge", bridge)

        passed = check('    {"ProbeVariadic_nid_postfix", GuestList, 1},', '"ProbeX87_nid_postfix"')
        self.assertEqual(passed.returncode, 0, passed.stdout)
        self.assertIn("stack-layout (1)", passed.stdout)
        failed = check('    {"ProbeVariadic_nid_postfix", GuestList, 2},', '"ProbeX87_nid_postfix"')
        self.assertIn("has 1 fixed arguments, VariadicExports says 2", failed.stdout)
        failed = check('    {"ProbeGone_nid_postfix", Integers, 2},', '"ProbeX87_nid_postfix"')
        self.assertIn("ProbeVariadic_nid_postfix is variadic with 1 fixed arguments but not in VariadicExports", failed.stdout)
        self.assertIn("VariadicExports lists ProbeGone_nid_postfix", failed.stdout)
        failed = check('    {"ProbeVariadic_nid_postfix", GuestList, 1},', '"strtold_nid_postfix"')
        self.assertIn("ProbeX87_nid_postfix returns GuestLongDouble but is not in X87Results", failed.stdout)
        self.assertNotEqual(failed.returncode, 0)

    def test_compiler_agrees_and_table_compiles(self):
        self.generate()
        result = run("verify", "--json", self.json, "--work", self.root / "build" / "verify", "--compile-commands", self.commands,
                     "--root", self.root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("4 exports compiled from library units compared, 0 unverified, 0 differ", result.stdout)
        compiled = subprocess.run(["xcrun", "clang++", "-std=c++20", "-fsyntax-only", "-DAPS5_FEX_BRIDGE_SIGNATURES",
                                   f"-I{SOURCES}", f"-I{self.inc.parent}", str(SOURCES / "BridgeSignatures.cpp")],
                                  capture_output=True, text=True)
        self.assertEqual(compiled.returncode, 0, compiled.stderr)

    def test_complex_long_double_is_unclassified(self):
        source = self.root / "core" / "libs" / "prx" / "libprobe" / "Export.cpp"
        source.write_text(SOURCE + 'extern "C" __complex__ long double ProbeComplex_nid_postfix(long double value) { return value; }\n')
        _, document = self.generate()
        self.assertTrue(any("ProbeComplex_nid_postfix: cannot classify _Complex long double" in problem
                            for problem in document["problems"]), document["problems"])

    def test_unparsable_unit_fails(self):
        source = self.root / "core" / "libs" / "prx" / "libprobe" / "Export.cpp"
        source.write_text(SOURCE + "UnknownType ProbeBroken();\n")
        result = run("generate", "--compile-commands", self.commands, "--root", self.root, "--cache", self.root / "build" / "cache",
                     "--json", self.json, "--inc", self.inc)
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(self.json.read_text())
        self.assertTrue(any("unknown type name 'UnknownType'" in problem for problem in document["problems"]), document["problems"])
        bridge = self.root / "Bridge.cpp"
        bridge.write_text('VariadicExports[] = {{"ProbeVariadic_nid_postfix", GuestList, 1}};\nX87Results[] = {"ProbeX87_nid_postfix"};\n')
        self.assertEqual(run("check", "--json", self.json, "--bridge", bridge).returncode, 1)


if __name__ == "__main__":
    unittest.main()
