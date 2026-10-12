#!/usr/bin/env python3
"""Unwrap fake-signed SELF containers (fSELF) into the plain ELF they carry.

    python3 tools/unself.py <SELF file> <output ELF>
    python3 tools/unself.py <title directory> <new output directory>

Layout of a SELF, little-endian: a 0x20-byte header (magic 4F 15 3D 1D, or 54 14 F5 EE for the PS5
module variant; entry count at 0x18), one 0x20-byte entry per stored range (properties, file
offset, file size, memory size), then the ELF header and its program headers. An entry with the
blocked flag (0x800) holds the file data of program header (properties >> 20) & 0xFFF; the other
entries hold digests. The PT_SCE_VERSION data has no entry and is appended after the last entry.

The output is the ELF header and program headers, then each stored segment at its p_offset. A SELF
does not store section headers, so e_shoff, e_shnum and e_shstrndx are cleared. Non-loadable
PT_NOTE and PT_SCE_COMMENT ranges without stored data are left zero-filled and reported. Encrypted
or compressed segments (retail SELFs, which need a console to decrypt) are refused.

In directory mode the tree is copied, every file starting with a SELF magic is unwrapped and every
other file is copied unchanged. Symbolic links are refused."""
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

MAGICS = {bytes.fromhex("4f153d1d"): "SELF", bytes.fromhex("5414f5ee"): "PS5 module SELF"}
HEADER = struct.Struct("<4sBBBBIHHQHHI")
ENTRY = struct.Struct("<QQQQ")
ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")

BLOCKED = 0x800
ENCRYPTED = 0x2
COMPRESSED = 0x8
EM_X86_64 = 62
PT_LOAD = 1
PT_NOTE = 4
PT_SCE_COMMENT = 0x6FFFFF00
PT_SCE_VERSION = 0x6FFFFF01
NAMES = {0: "PT_NULL", 1: "PT_LOAD", 2: "PT_DYNAMIC", 3: "PT_INTERP", 4: "PT_NOTE", 6: "PT_PHDR", 7: "PT_TLS",
         0x6474E550: "PT_GNU_EH_FRAME", 0x6474E551: "PT_GNU_STACK", 0x6474E552: "PT_GNU_RELRO",
         0x61000000: "PT_SCE_DYNLIBDATA", 0x61000001: "PT_SCE_PROCPARAM", 0x61000002: "PT_SCE_MODULE_PARAM",
         0x61000010: "PT_SCE_RELRO", PT_SCE_COMMENT: "PT_SCE_COMMENT", PT_SCE_VERSION: "PT_SCE_VERSION"}


class UnselfError(ValueError):
    pass


def is_self(data):
    return data[:4] in MAGICS


def type_name(kind):
    return NAMES.get(kind, f"type {kind:#x}")


def covered(ranges, start, end):
    position = start
    for low, high in sorted(ranges):
        if low > position:
            break
        position = max(position, high)
        if position >= end:
            return True
    return position >= end


def unself(data):
    if len(data) < HEADER.size or not is_self(data):
        raise UnselfError(f"not a SELF: magic {data[:4].hex(' ')}, expected 4f 15 3d 1d or 54 14 f5 ee")
    count = HEADER.unpack_from(data)[9]
    elf_start = HEADER.size + count * ENTRY.size
    if len(data) < elf_start + ELF_HEADER.size:
        raise UnselfError(f"truncated SELF: {len(data):#x} bytes, the {count} entries and the ELF header need {elf_start + ELF_HEADER.size:#x}")
    entries = [ENTRY.unpack_from(data, HEADER.size + index * ENTRY.size) for index in range(count)]
    fields = list(ELF_HEADER.unpack_from(data, elf_start))
    ident, machine, phoff, phentsize, phnum = fields[0], fields[2], fields[5], fields[9], fields[10]
    if ident[:4] != b"\x7fELF" or ident[4] != 2 or ident[5] != 1:
        raise UnselfError(f"the SELF does not hold a 64-bit little-endian ELF: e_ident {ident[:6].hex(' ')}")
    if machine != EM_X86_64:
        raise UnselfError(f"the ELF is for e_machine {machine}, expected {EM_X86_64} (x86-64)")
    if phentsize != PROGRAM_HEADER.size or phoff < ELF_HEADER.size:
        raise UnselfError(f"unexpected program header table: e_phoff {phoff:#x}, e_phentsize {phentsize:#x}")
    table_end = phoff + phnum * PROGRAM_HEADER.size
    if elf_start + table_end > len(data):
        raise UnselfError(f"truncated SELF: the program header table ends at {elf_start + table_end:#x}, the file at {len(data):#x}")
    headers = [PROGRAM_HEADER.unpack_from(data, elf_start + phoff + index * PROGRAM_HEADER.size) for index in range(phnum)]

    notes = []
    if fields[6] or fields[12]:
        notes.append(f"section headers are not stored in a SELF: cleared e_shoff {fields[6]:#x} and e_shnum {fields[12]}")
    fields[6], fields[12], fields[13] = 0, 0, 0
    size = max([table_end] + [header[2] + header[5] for header in headers])
    output = bytearray(size)
    output[:ELF_HEADER.size] = ELF_HEADER.pack(*fields)
    output[phoff:table_end] = data[elf_start + phoff:elf_start + table_end]

    stored = {}
    body_end = elf_start + table_end
    for index, (properties, offset, filesz, memsz) in enumerate(entries):
        if offset + filesz > len(data):
            raise UnselfError(f"truncated SELF: entry {index} ends at {offset + filesz:#x}, the file at {len(data):#x}")
        body_end = max(body_end, offset + filesz)
        if not properties & BLOCKED:
            continue
        segment = (properties >> 20) & 0xFFF
        if properties & ENCRYPTED:
            raise UnselfError(f"segment {segment} (entry {index}) is encrypted: only fake-signed SELFs can be unwrapped")
        if properties & COMPRESSED:
            raise UnselfError(f"segment {segment} (entry {index}) is compressed: only fake-signed SELFs with plain segments can be unwrapped")
        if segment >= phnum:
            raise UnselfError(f"entry {index} holds segment {segment}, the ELF has {phnum} program headers")
        if segment in stored:
            raise UnselfError(f"entries {stored[segment]} and {index} both hold segment {segment}")
        p_offset, p_filesz = headers[segment][2], headers[segment][5]
        if filesz != p_filesz:
            raise UnselfError(f"entry {index} holds {filesz:#x} bytes, program header {segment} needs {p_filesz:#x}")
        output[p_offset:p_offset + filesz] = data[offset:offset + filesz]
        stored[segment] = index

    ranges = [(headers[segment][2], headers[segment][2] + headers[segment][5]) for segment in stored]
    for segment, header in enumerate(headers):
        kind, p_offset, p_filesz = header[0], header[2], header[5]
        if kind != PT_SCE_VERSION or segment in stored or not p_filesz:
            continue
        trailing = len(data) - body_end
        if trailing >= p_filesz:
            output[p_offset:p_offset + p_filesz] = data[len(data) - p_filesz:]
            ranges.append((p_offset, p_offset + p_filesz))
        elif trailing == 0:
            notes.append(f"program header {segment} (PT_SCE_VERSION, {p_filesz:#x} bytes) is not appended to the SELF: zero-filled")
            ranges.append((p_offset, p_offset + p_filesz))
        else:
            raise UnselfError(f"the SELF ends {trailing:#x} bytes after its last entry, PT_SCE_VERSION needs {p_filesz:#x}")

    for segment, header in enumerate(headers):
        kind, p_offset, p_filesz = header[0], header[2], header[5]
        if not p_filesz or covered(ranges, p_offset, p_offset + p_filesz):
            continue
        if kind in (PT_NOTE, PT_SCE_COMMENT) and header[3] == 0:
            notes.append(f"program header {segment} ({type_name(kind)}, {p_filesz:#x} bytes at {p_offset:#x}) is not stored in the SELF: zero-filled")
            continue
        raise UnselfError(f"program header {segment} ({type_name(kind)}, {p_filesz:#x} bytes at {p_offset:#x}) has no data in the SELF")
    return bytes(output), notes


def write(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        temporary.write(data)
    os.chmod(temporary.name, 0o644)
    os.replace(temporary.name, path)


def convert_file(source, destination):
    data = Path(source).read_bytes()
    if not is_self(data):
        if data[:4] == b"\x7fELF":
            write(destination, data)
            return [f"{source} is already an ELF: copied unchanged"]
        raise UnselfError(f"{source}: not a SELF: magic {data[:4].hex(' ')}, expected 4f 15 3d 1d or 54 14 f5 ee")
    try:
        elf, notes = unself(data)
    except UnselfError as error:
        raise UnselfError(f"{source}: {error}") from None
    write(destination, elf)
    return [f"{source}: {MAGICS[data[:4]]} unwrapped, {len(data):#x} -> {len(elf):#x} bytes"] + [f"{source}: {note}" for note in notes]


def convert_tree(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise UnselfError(f"{destination} already exists")
    if destination.resolve().is_relative_to(source):
        raise UnselfError(f"{destination} is inside {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}."))
    messages = []
    try:
        for directory, subdirectories, files in os.walk(source):
            subdirectories.sort()
            relative = Path(directory).relative_to(source)
            for name in subdirectories + files:
                if (Path(directory) / name).is_symlink():
                    raise UnselfError(f"{relative / name} in {source} is a symbolic link")
            for name in subdirectories:
                (staging / relative / name).mkdir()
            for name in sorted(files):
                path = Path(directory) / name
                if not path.is_file():
                    raise UnselfError(f"{relative / name} in {source} is not a regular file")
                with open(path, "rb") as file:
                    magic = file.read(4)
                if is_self(magic):
                    messages.extend(convert_file(path, staging / relative / name))
                else:
                    shutil.copyfile(path, staging / relative / name)
                    os.chmod(staging / relative / name, 0o644)
        staging.chmod(0o755)
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return messages


def main(arguments):
    if len(arguments) != 2:
        print("usage: unself.py <SELF file> <output ELF> | <title directory> <new output directory>", file=sys.stderr)
        return 2
    source, destination = arguments
    try:
        if Path(source).is_dir():
            messages = convert_tree(source, destination)
        else:
            messages = convert_file(source, destination)
    except (UnselfError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
