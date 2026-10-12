"""Tests for asm/prepare_asm_tests.py on synthetic inputs.

    python3 test_prepare_asm_tests.py

The end-to-end cases assemble with nasm and are skipped when it is missing."""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "asm"))
import prepare_asm_tests as prepare

DELTA = prepare.DELTA
SCRATCH = prepare.Ranges([prepare.POINTER_WINDOW])


def analysed(source, ranges=SCRATCH):
    return prepare.analyse(source.strip("\n").splitlines(), ranges)


def moved(source, ranges=SCRATCH):
    return [(origin.family, origin.value) for origin in analysed(source, ranges).origins]


class Numbers(unittest.TestCase):
    def test_nasm_number_forms(self):
        self.assertEqual(prepare.nasm_number("0xe000_0000"), 0xE000_0000)
        self.assertEqual(prepare.nasm_number("0E0000000h"), 0xE000_0000)
        self.assertEqual(prepare.nasm_number("0b101"), 5)
        self.assertEqual(prepare.nasm_number("101b"), 5)
        self.assertEqual(prepare.nasm_number("17q"), 15)
        self.assertEqual(prepare.nasm_number("1_000"), 1000)
        self.assertIsNone(prepare.nasm_number("0xzz"))

    def test_constant_expressions(self):
        self.assertEqual(prepare.constant("0xe000_0000 + 8 * 3"), 0xE000_0018)
        self.assertEqual(prepare.constant("qword (4096 / 8) - 1"), 511)
        self.assertEqual(prepare.constant("-1"), (1 << 64) - 1)
        self.assertEqual(prepare.constant("1 << 4 | 1"), 17)
        self.assertIsNone(prepare.constant("label + 4"))
        self.assertIsNone(prepare.constant("[rax]"))
        self.assertIsNone(prepare.constant("rax"))
        self.assertIsNone(prepare.constant("'ab'"))


class Lines(unittest.TestCase):
    def test_labels_prefixes_and_operands(self):
        line = prepare.parse_line(".loop: ..@3.next: lock add qword [rdx + 8 * 2], rax ; comment, with comma")
        self.assertEqual(line.head, ".loop: ..@3.next: lock ")
        self.assertEqual(line.prefixes, ["lock"])
        self.assertEqual(line.mnemonic, "add")
        self.assertEqual(line.operands, ["qword [rdx + 8 * 2]", "rax"])

    def test_directives_and_times(self):
        self.assertIsNone(prepare.parse_line("[bits 64]"))
        self.assertIsNone(prepare.parse_line("%line 3+1 test.asm"))
        self.assertIsNone(prepare.parse_line("   "))
        self.assertEqual(prepare.parse_line("data: dq 1, 2").mnemonic, "dq")
        self.assertEqual(prepare.parse_line("value equ 5").mnemonic, "equ")
        nested = prepare.parse_line("times (((16) - (($-$$) % (16))) % (16)) nop")
        self.assertEqual(nested.mnemonic, "nop")
        repeated = prepare.parse_line("times 4 rep movsb")
        self.assertEqual((repeated.prefixes, repeated.mnemonic), (["rep"], "movsb"))


class Analysis(unittest.TestCase):
    def test_pointer_is_moved_and_value_is_not(self):
        self.assertEqual(moved("""
mov rdx, 0xe0000000
mov rax, 0xe0000010
mov [rdx], rax
"""), [("rdx", 0xE000_0000)])

    def test_32_bit_mov_of_an_address_is_widened(self):
        source = "start: mov edx, 0xe0000000\nmov [rdx + 8], rax"
        analysis = analysed(source)
        self.assertEqual([(origin.family, origin.value) for origin in analysis.origins], [("rdx", 0xE000_0000)])
        self.assertEqual(prepare.rewrite(source.splitlines(), analysis)[0], f"start: mov rdx, {0xE000_0000 + DELTA:#x}")

    def test_pointer_follows_copies_arithmetic_and_the_stack(self):
        self.assertEqual(moved("""
mov rax, 0xe0000000
mov rbx, rax
add rbx, 16
push rbx
pop rsi
lodsq
"""), [("rax", 0xE000_0000)])

    def test_string_and_stack_operations_use_their_registers(self):
        self.assertEqual(moved("mov rdi, 0xe0000100\nrep stosb\nmov rsp, 0xe0001000\npush rax"),
                         [("rdi", 0xE000_0100), ("rsp", 0xE000_1000)])

    def test_compared_and_subtracted_addresses_move_together(self):
        self.assertEqual(moved("""
mov rsp, 0xe0001000
mov rbx, 0xe0000ff8
push rax
cmp rsp, rbx
mov rsi, 0xf0000000
mov rcx, 0xf0001000
sub rcx, rsi
lodsb
""", prepare.Ranges([prepare.POINTER_WINDOW, (0xF000_0000, 0xF000_1000)])),
            [("rsp", 0xE000_1000), ("rbx", 0xE000_0FF8), ("rsi", 0xF000_0000), ("rcx", 0xF000_1000)])

    def test_unscaled_base_moves_when_only_the_sum_is_an_address(self):
        ranges = prepare.Ranges([prepare.POINTER_WINDOW, (0x1_1000_0000, 0x1_1000_1000)])
        self.assertEqual(moved("mov r14, 0x1000_0000\nmov r15, 0x2000_0000\nmov rax, [r14 + (r15 * 8)]", ranges),
                         [("r14", 0x1000_0000)])

    def test_segment_relative_registers_are_offsets(self):
        analysis = analysed("mov rdx, 0xe0000008\nwrfsbase rdx\nmov r8, 0xe0000010\na32 mov rax, qword [fs:r8d]")
        self.assertEqual(analysis.problems, [])
        self.assertEqual([origin.family for origin in analysis.origins], ["rdx"])

    def test_unmovable_uses_are_reported(self):
        cases = {
            "mov rdi, 0xe0000000\nmov edi, edi\nstosb": "truncated to 32 bits",
            "mov rsi, 0xe0000000\na32 lodsb": "truncated to 32 bits",
            "mov rsi, 0x1_0008_0000\na32 lodsq": "32-bit address 0x80000",
            "mov rax, [abs 0xe0000000]": "absolute address 0xe0000000",
            "lea edi, [rel target]\ntarget: hlt": "32-bit register",
            "dq 0xe0000008": "keeps the address",
            "db 0x67\ndb 0xac\nhlt": "0x67 address-size prefix",
            "mov ax, 0xe000": None,
        }
        ranges = prepare.Ranges([prepare.POINTER_WINDOW, (0x8_0000, 0x8_1000)])
        for source, problem in cases.items():
            with self.subTest(source=source):
                problems = analysed(source, ranges).problems
                if problem is None:
                    self.assertEqual(problems, [])
                else:
                    self.assertTrue(any(problem in found for found in problems), problems)


class Lists(unittest.TestCase):
    def setUp(self):
        self.directory = pathlib.Path(tempfile.mkdtemp(prefix="aps5-lists-"))

    def tearDown(self):
        shutil.rmtree(self.directory)

    def write(self, name, text):
        (self.directory / name).write_text(text)
        return self.directory / name

    def test_our_list_needs_reasons_and_known_qualifiers(self):
        path = self.write("Known_Failures_macos", "# comment\nA/b.asm: reason one\nA/c.asm [full]: only full\n"
                                                  "A/d.asm [default, nasm3]: assembler\nA/e.asm [nasm2]: old assembler\n")
        entries = prepare.read_our_list(path, 3)
        self.assertEqual(entries["A/b.asm"], [("*", "Known_Failures_macos: reason one")])
        self.assertEqual(entries["A/c.asm"], [("full", "Known_Failures_macos: only full")])
        self.assertEqual(entries["A/d.asm"], [("default", "Known_Failures_macos: assembler")])
        self.assertEqual(entries["A/e.asm"], [])
        self.assertEqual(prepare.read_our_list(path, 2)["A/d.asm"], [])
        for text in ("A/b.asm\n", "A/b.asm:\n", "A/b.asm [fast]: reason\n"):
            with self.subTest(text=text), self.assertRaises(prepare.PrepareError):
                prepare.read_our_list(self.write("Known_Failures_macos", text), 3)
        with self.assertRaises(prepare.PrepareError):
            prepare.read_our_list(self.write("Disabled_Tests_macos", "A/b.asm [full]: reason\n"), 3)

    def test_fex_list_reasons_come_from_comments(self):
        path = self.write("Disabled_Tests", "# Can't test this\nTest_Primary/E8.asm\n\nTest_X/y.asm\n"
                                            "jit_1/Test_64Bit_Z/w.asm\nhost/Test_64Bit_Z/v.asm\nNo_prefix.asm\n")
        entries = prepare.read_fex_list(path)
        self.assertEqual(entries["Primary/E8.asm"], [("*", "FEX Disabled_Tests: Can't test this")])
        self.assertEqual(entries["X/y.asm"], [("*", "FEX Disabled_Tests: no reason given")])
        self.assertEqual(entries["Z/w.asm"], [("full", "FEX Disabled_Tests: no reason given")])
        self.assertNotIn("Z/v.asm", entries)
        self.assertNotIn("No_prefix.asm", entries)


class MemoryData(unittest.TestCase):
    def test_both_json_forms(self):
        self.assertEqual(prepare.memory_bytes("88 77 66"), bytes([0x88, 0x77, 0x66]))
        self.assertEqual(prepare.memory_bytes("0x4142434445464748"), bytes.fromhex("4847464544434241"))


TEST = """%ifdef CONFIG
{
  "RegData": {
    "RAX": "0x1",
    "RDI": "0xe0000008",
    "R14": "0x10028",
    "XMM0": ["0x1", "0x2", "0x3", "0x4"]
  },
  "MemoryRegions": {"0x100000000": "4096"},
  "MemoryData": {"0x100000000": "0x0102"},
  "Env": {"FEX_X87REDUCEDPRECISION": "1"},
  "HostFeatures": ["AVX"]
}
%endif

mov edi, 0xe0000000
mov qword [rdi], 1
add rdi, 8
mov rax, [rdi - 8]
mov rsi, 0x1_0000_0000
mov rbx, [rsi]
lea r14, [rel .end]
.end:
hlt
"""


@unittest.skipUnless(os.environ.get("NASM") or shutil.which("nasm"), "nasm is not installed")
class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.directory = pathlib.Path(tempfile.mkdtemp(prefix="aps5-prepare-"))
        self.asm = self.directory / "ASM"
        (self.asm / "Includes").mkdir(parents=True)
        (self.asm / "Group").mkdir()
        (self.asm / "Group" / "moved.asm").write_text(TEST)
        (self.asm / "Group" / "plain.asm").write_text('%ifdef CONFIG\n{"RegData": {"RAX": "0x2"}}\n%endif\nmov rax, 2\nhlt\n')
        (self.asm / "Group" / "skipped.asm").write_text("hlt\n")
        (self.asm / "Group" / "unmovable.asm").write_text("mov rsi, 0xe0000000\na32 lodsb\nhlt\n")
        (self.asm / "Disabled_Tests").write_text("# Not here\nTest_Group/skipped.asm\n")
        self.lists = self.directory / "lists"
        self.lists.mkdir()
        (self.lists / "Known_Failures_macos").write_text("Group/plain.asm [full]: an example\n")
        self.output = self.directory / "out"

    def tearDown(self):
        shutil.rmtree(self.directory)

    def run_prepare(self):
        result = subprocess.run([sys.executable, prepare.__file__, str(self.asm), str(self.lists), str(self.output)],
                                capture_output=True, text=True)
        return result.returncode, result.stdout + result.stderr

    def config(self, name):
        return (self.output / "Group" / (name + ".asm.cfg")).read_text().splitlines()

    def test_prepares_moves_and_expectations(self):
        code, output = self.run_prepare()
        self.assertEqual(code, 0, output)
        self.assertIn("prepared 4 tests", output)
        lines = self.config("moved")
        self.assertEqual(lines[0], prepare.FORMAT)
        self.assertIn("relocations 2", lines)
        self.assertIn(f"code {prepare.CODE_BASE + DELTA:#x}", lines)
        self.assertIn(f"map {0xE000_0000 + DELTA:#x} {prepare.SCRATCH_SIZE:#x}", lines)
        self.assertIn(f"map {0x1_0000_0000 + DELTA:#x} {prepare.PAGE:#x}", lines)
        self.assertIn(f"memory {0x1_0000_0000 + DELTA:#x} 0201", lines)
        self.assertIn("option FEX_X87REDUCEDPRECISION 1", lines)
        self.assertIn("feature AVX", lines)
        self.assertIn("reg XMM0 0x1 0x2 0x3 0x4", lines)
        self.assertIn(f"alt RDI 0 {0xE000_0008 + DELTA:#x}", lines)
        binary = (self.output / "Group" / "moved.asm.bin").read_bytes()
        self.assertIn((0xE000_0000 + DELTA).to_bytes(8, "little"), binary)
        self.assertIn((0x1_0000_0000 + DELTA).to_bytes(8, "little"), binary)
        end = binary.index(bytes([0xF4]))
        self.assertIn(f"alt R14 0 {DELTA + prepare.CODE_BASE + end:#x}", lines)
        self.assertEqual(end, 0x28 + 5)
        instructions = next(line for line in lines if line.startswith("instructions ")).split()[1:]
        self.assertIn("lea", instructions)
        self.assertEqual(self.config("skipped")[2], "skip disabled by FEX Disabled_Tests: Not here")
        self.assertTrue(self.config("unmovable")[-1].startswith("fail uses low addresses that cannot be moved"))
        self.assertIn("known_failure full Known_Failures_macos: an example", self.config("plain"))
        self.assertFalse((self.output / "Group" / "skipped.asm.bin").exists())

    def test_second_run_reuses_unchanged_outputs(self):
        self.assertEqual(self.run_prepare()[0], 0)
        config = self.output / "Group" / "moved.asm.cfg"
        before = config.stat().st_mtime_ns
        self.assertEqual(self.run_prepare()[0], 0)
        self.assertEqual(config.stat().st_mtime_ns, before)
        (self.lists / "Known_Failures_macos").write_text("Group/moved.asm: now listed\n")
        self.assertEqual(self.run_prepare()[0], 0)
        self.assertIn("known_failure * Known_Failures_macos: now listed", config.read_text().splitlines())

    def test_listing_a_missing_test_fails(self):
        (self.lists / "Disabled_Tests_macos").write_text("Group/gone.asm: removed upstream\n")
        code, output = self.run_prepare()
        self.assertEqual(code, 1)
        self.assertIn("Group/gone.asm is listed", output)


if __name__ == "__main__":
    unittest.main()
