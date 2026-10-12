import contextlib
import io
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unself

SIGNED = 0x4
DIGESTS = 0x10000
EXECUTABLE = bytes.fromhex("4f153d1d")
MODULE = bytes.fromhex("5414f5ee")

TEXT = bytes((index * 7 + 3) & 0xFF for index in range(0x123))
DATA = bytes((index * 13 + 1) & 0xFF for index in range(0x80))
VERSION = b"\x00\x00\x16\x00\x08libc:\x02\x00\x00\t\x00\x00\x00\x01\x02\x00\x00\t\x00\x00\x00\x01"

PROGRAM_HEADERS = [
    (unself.PT_LOAD, 5, 0x1000, 0x0, 0x0, len(TEXT), len(TEXT), 0x4000),
    (unself.PT_LOAD, 6, 0x2000, 0x4000, 0x4000, len(DATA), 0x200, 0x4000),
    (2, 6, 0x2010, 0x4010, 0x4010, 0x20, 0x20, 0x8),
    (unself.PT_SCE_VERSION, 0, 0x2080, 0, 0, len(VERSION), len(VERSION), 0x10),
    (unself.PT_NOTE, 0, 0x20A0, 0, 0, 0x18, 0, 0x4),
]
SEGMENTS = [(0, TEXT), (1, DATA)]
ELF_SIZE = 0x20B8


def elf_head(machine=unself.EM_X86_64, shoff=0x3000, shnum=5, headers=PROGRAM_HEADERS):
    ident = b"\x7fELF" + bytes([2, 1, 1, 9]) + bytes(8)
    header = unself.ELF_HEADER.pack(ident, 0xFE10, machine, 1, 0x1040, 0x40, shoff, 0, 0x40,
                                    unself.PROGRAM_HEADER.size, len(headers), 0x40, shnum, shnum - 1 if shnum else 0)
    return header + b"".join(unself.PROGRAM_HEADER.pack(*fields) for fields in headers)


def align(value, alignment=0x10):
    return (value + alignment - 1) & ~(alignment - 1)


def build(magic=EXECUTABLE, segments=SEGMENTS, extra=0, version=VERSION, head=None, sizes=None, ids=None):
    head = elf_head() if head is None else head
    count = 2 * len(segments)
    position = align(unself.HEADER.size + count * unself.ENTRY.size + len(head)) + 0x100
    entries, payload = [], []
    for number, (segment, data) in enumerate(segments):
        digest = bytes([0xD0 + number]) * 0x20
        entries.append(((2 * number + 1) << 20 | SIGNED | DIGESTS, position, len(digest), len(digest)))
        payload.append((position, digest))
        position += len(digest)
        segment = ids[number] if ids else segment
        size = sizes[number] if sizes else len(data)
        entries.append((segment << 20 | 2 << 12 | unself.BLOCKED | SIGNED | extra, position, size, size))
        payload.append((position, data))
        position = align(position + len(data))
    end = max(offset + len(data) for offset, data in payload)
    image = bytearray(end)
    image[:unself.HEADER.size] = unself.HEADER.pack(magic, 0, 1, 1, 0x12, 0x101, 0x560, 0x510, align(end), count, 0x22, 0)
    for index, fields in enumerate(entries):
        unself.ENTRY.pack_into(image, unself.HEADER.size + index * unself.ENTRY.size, *fields)
    start = unself.HEADER.size + count * unself.ENTRY.size
    image[start:start + len(head)] = head
    for offset, data in payload:
        image[offset:offset + len(data)] = data
    return bytes(image) + version


def expected_elf():
    output = bytearray(ELF_SIZE)
    head = elf_head(shoff=0, shnum=0)
    output[:len(head)] = head
    output[0x1000:0x1000 + len(TEXT)] = TEXT
    output[0x2000:0x2000 + len(DATA)] = DATA
    output[0x2080:0x2080 + len(VERSION)] = VERSION
    return bytes(output)


class UnwrapTests(unittest.TestCase):
    def test_executable_self_becomes_the_plain_elf(self):
        elf, notes = unself.unself(build())
        self.assertEqual(elf, expected_elf())
        self.assertEqual(notes, ["section headers are not stored in a SELF: cleared e_shoff 0x3000 and e_shnum 5",
                                 "program header 4 (PT_NOTE, 0x18 bytes at 0x20a0) is not stored in the SELF: zero-filled"])

    def test_module_variant_has_the_same_layout(self):
        self.assertEqual(unself.unself(build(magic=MODULE))[0], expected_elf())

    def test_missing_version_data_is_zero_filled_and_reported(self):
        elf, notes = unself.unself(build(version=b""))
        self.assertEqual(elf[0x2080:0x2080 + len(VERSION)], bytes(len(VERSION)))
        self.assertIn("program header 3 (PT_SCE_VERSION, 0x1a bytes) is not appended to the SELF: zero-filled", notes)

    def test_version_data_after_padding_is_taken_from_the_end(self):
        elf, _ = unself.unself(build(version=bytes(6) + VERSION))
        self.assertEqual(elf, expected_elf())

    def test_short_version_data_fails(self):
        with self.assertRaisesRegex(unself.UnselfError, "PT_SCE_VERSION needs 0x1a"):
            unself.unself(build(version=VERSION[:5]))

    def test_encrypted_segment_fails(self):
        with self.assertRaisesRegex(unself.UnselfError, "segment 0 \\(entry 1\\) is encrypted"):
            unself.unself(build(extra=unself.ENCRYPTED))

    def test_compressed_segment_fails(self):
        with self.assertRaisesRegex(unself.UnselfError, "segment 0 \\(entry 1\\) is compressed"):
            unself.unself(build(extra=unself.COMPRESSED))

    def test_unknown_magic_fails(self):
        with self.assertRaisesRegex(unself.UnselfError, "not a SELF: magic 7f 45 4c 46"):
            unself.unself(expected_elf())

    def test_other_architectures_fail(self):
        with self.assertRaisesRegex(unself.UnselfError, "e_machine 183, expected 62"):
            unself.unself(build(head=elf_head(machine=183)))

    def test_size_mismatch_fails(self):
        with self.assertRaisesRegex(unself.UnselfError, "entry 3 holds 0x7f bytes, program header 1 needs 0x80"):
            unself.unself(build(sizes=[len(TEXT), len(DATA) - 1]))

    def test_segment_beyond_the_program_headers_fails(self):
        with self.assertRaisesRegex(unself.UnselfError, "entry 3 holds segment 9, the ELF has 5 program headers"):
            unself.unself(build(ids=[0, 9]))

    def test_duplicate_segment_fails(self):
        with self.assertRaisesRegex(unself.UnselfError, "entries 1 and 3 both hold segment 0"):
            unself.unself(build(segments=[(0, TEXT), (0, TEXT)]))

    def test_missing_loadable_segment_fails(self):
        with self.assertRaisesRegex(unself.UnselfError, "program header 1 \\(PT_LOAD, 0x80 bytes at 0x2000\\) has no data"):
            unself.unself(build(segments=[(0, TEXT)]))

    def test_truncated_self_fails(self):
        data = build(version=b"")
        with self.assertRaisesRegex(unself.UnselfError, "truncated SELF: entry 3 ends"):
            unself.unself(data[:-0x10])
        with self.assertRaisesRegex(unself.UnselfError, "truncated SELF"):
            unself.unself(data[:0x60])


class CommandLineTests(unittest.TestCase):
    def run_main(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = unself.main([str(argument) for argument in arguments])
        return status, output.getvalue(), errors.getvalue()

    def test_file_mode_writes_the_elf(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "eboot.bin", Path(directory) / "eboot.elf"
            source.write_bytes(build())
            status, output, errors = self.run_main(source, destination)
            self.assertEqual((status, errors), (0, ""))
            self.assertIn("SELF unwrapped, 0x", output)
            self.assertEqual(destination.read_bytes(), expected_elf())

    def test_plain_elf_is_copied_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "eboot.bin", Path(directory) / "copy.bin"
            source.write_bytes(expected_elf())
            status, output, _ = self.run_main(source, destination)
            self.assertEqual(status, 0)
            self.assertIn("already an ELF", output)
            self.assertEqual(destination.read_bytes(), expected_elf())

    def test_errors_are_reported_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "eboot.bin", Path(directory) / "eboot.elf"
            source.write_bytes(build(extra=unself.ENCRYPTED))
            status, _, errors = self.run_main(source, destination)
            self.assertEqual(status, 1)
            self.assertIn("is encrypted", errors)
            self.assertFalse(destination.exists())
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["eboot.bin"])

    def test_tree_mode_unwraps_selfs_and_copies_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            title, output = Path(directory) / "title", Path(directory) / "out" / "title"
            (title / "sce_module").mkdir(parents=True)
            (title / "sce_sys").mkdir()
            (title / "eboot.bin").write_bytes(build())
            (title / "sce_module" / "libc.prx").write_bytes(build(magic=MODULE))
            (title / "sce_sys" / "param.json").write_text('{"titleId": "TEST00000"}', encoding="utf-8")
            (title / "a").write_bytes(b"\x4f\x15")
            status, output_text, errors = self.run_main(title, output)
            self.assertEqual((status, errors), (0, ""))
            self.assertEqual(output_text.count("unwrapped"), 2)
            self.assertEqual((output / "eboot.bin").read_bytes(), expected_elf())
            self.assertEqual((output / "sce_module" / "libc.prx").read_bytes(), expected_elf())
            self.assertEqual((output / "sce_sys" / "param.json").read_bytes(), (title / "sce_sys" / "param.json").read_bytes())
            self.assertEqual((output / "a").read_bytes(), b"\x4f\x15")
            self.assertEqual((title / "eboot.bin").read_bytes(), build())
            self.assertEqual(sorted(path.name for path in output.parent.iterdir()), ["title"])

    def test_tree_mode_refuses_existing_destinations_links_and_nesting(self):
        with tempfile.TemporaryDirectory() as directory:
            title = Path(directory) / "title"
            title.mkdir()
            (title / "eboot.bin").write_bytes(build())
            existing = Path(directory) / "existing"
            existing.mkdir()
            self.assertEqual(self.run_main(title, existing)[0], 1)
            self.assertIn("is inside", self.run_main(title, title / "out")[2])
            os.symlink("/etc/hosts", title / "link")
            status, _, errors = self.run_main(title, Path(directory) / "out")
            self.assertEqual(status, 1)
            self.assertIn("link in", errors)
            self.assertIn("is a symbolic link", errors)
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["existing", "title"])


if __name__ == "__main__":
    unittest.main()
