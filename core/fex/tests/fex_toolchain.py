"""Builds PS5-style x86-64 executables from C++ with the tools a Mac has: Apple clang for x86-64
FreeBSD objects, llvm-nm, and an LLVM ELF linker (ld.lld, or rust-lld as one). Every external symbol
is renamed to its NID, and stub libraries stand in for the prx."""
import os
import pathlib
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../relinker/relinker/tests/macos"))
from nid import nid

TARGET = "--target=x86_64-unknown-freebsd13"


def _xcrun(tool):
    found = subprocess.run(["xcrun", "--find", tool], capture_output=True, text=True)
    return found.stdout.strip() if found.returncode == 0 else None


def clang():
    return shutil.which("clang++") or _xcrun("clang++")


def nm():
    return shutil.which("llvm-nm") or _xcrun("llvm-nm")


def linker():
    """The linker command and the environment to run it in."""
    if lld := shutil.which("ld.lld"):
        return [lld], None
    rustup = pathlib.Path.home() / ".rustup" / "toolchains"
    for candidate in sorted(rustup.glob("*/lib/rustlib/*/bin/rust-lld")):
        toolchain = candidate.parents[4]
        return [str(candidate), "-flavor", "gnu"], {**os.environ, "DYLD_LIBRARY_PATH": str(toolchain / "lib")}
    return None, None


def link(arguments):
    command, environment = linker()
    subprocess.run([*command, *arguments], check=True, env=environment)


def available():
    return clang() is not None and nm() is not None and linker()[0] is not None


def compile(source, obj, pic=False, extra=()):
    subprocess.run([clang(), TARGET, "-fPIC" if pic else "-fPIE", "-O1", "-fexceptions", "-fno-stack-protector",
                    "-ffreestanding", "-fno-builtin", "-nostdinc++", *extra, "-c", str(source), "-o", str(obj)], check=True)


def symbols(obj, *flags):
    return subprocess.run([nm(), *flags, "--format=just-symbols", str(obj)], check=True, capture_output=True,
                          text=True).stdout.split()


def nidify(obj, out, keep=("_start",)):
    """Renames the object's undefined and global symbols to their NIDs, as llvm-objcopy --redefine-syms
    would: the new names go in a string table appended to the file."""
    names = [n for n in symbols(obj, "-u") + symbols(obj, "-g", "--defined-only") if n not in keep]
    mapping = {name: nid(name) for name in names}
    data = bytearray(pathlib.Path(obj).read_bytes())
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3a)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    symtab = next(i for i, section in enumerate(sections) if section[1] == 2)
    strtab = sections[symtab][6]
    old = data[sections[strtab][4]:sections[strtab][4] + sections[strtab][5]]
    table = bytearray(old)
    offset, size, entsize = sections[symtab][4], sections[symtab][5], sections[symtab][9]
    for entry in range(offset, offset + size, entsize):
        name_offset, = struct.unpack_from("<I", data, entry)
        name = old[name_offset:old.index(0, name_offset)].decode()
        if name in mapping:
            struct.pack_into("<I", data, entry, len(table))
            table += mapping[name].encode() + b"\0"
    while len(data) % 8:
        data.append(0)
    new_offset = len(data)
    data += table
    header = list(sections[strtab])
    header[4], header[5] = new_offset, len(table)
    struct.pack_into("<IIQQQQIIQQ", data, shoff + strtab * shentsize, *header)
    pathlib.Path(out).write_bytes(data)
    return mapping


def stub_library(names, out, soname):
    lines = ["\t.text"]
    for name in names:
        if name.startswith("_ZTV") or name.startswith("_ZTI"):
            lines += ["\t.data", f'\t.globl "{nid(name)}"', f'\t.type "{nid(name)}",@object', f'\t.size "{nid(name)}",24',
                      f'"{nid(name)}":', "\t.quad 0, 0, 0", "\t.text"]
        else:
            lines += [f'\t.globl "{nid(name)}"', f'\t.type "{nid(name)}",@function', f'"{nid(name)}":', "\tret"]
    pathlib.Path(str(out) + ".s").write_text("\n".join(lines) + "\n")
    subprocess.run([clang(), TARGET, "-c", "-x", "assembler", str(out) + ".s", "-o", str(out) + ".o"], check=True)
    link(["-shared", "-soname", soname, str(out) + ".o", "-o", str(out)])


def link_module(objs, libs, out, soname):
    link(["-shared", "-z", "now", "--hash-style=sysv", "--eh-frame-hdr", "-soname", soname, *map(str, objs), *map(str, libs),
          "-o", str(out)])
    clear_static_tls_flag(out)


def clear_static_tls_flag(path):
    """The relinker's guest module reader accepts DT_FLAGS = DF_BIND_NOW only; the linker adds
    DF_STATIC_TLS for an initial-exec variable."""
    data = bytearray(pathlib.Path(path).read_bytes())
    phoff, = struct.unpack_from("<Q", data, 32)
    phnum, = struct.unpack_from("<H", data, 56)
    for index in range(phnum):
        kind, _, offset, _, _, size = struct.unpack_from("<IIQQQQ", data, phoff + index * 56)
        if kind != 2:
            continue
        for entry in range(offset, offset + size, 16):
            tag, value = struct.unpack_from("<qQ", data, entry)
            if tag == 30:
                struct.pack_into("<qQ", data, entry, 30, value & 8)
    pathlib.Path(path).write_bytes(data)


def link_executable(objs, libs, out):
    link(["-pie", "-z", "now", "--hash-style=sysv", "--eh-frame-hdr", "--no-dynamic-linker", "-e", "_start",
          *map(str, objs), *map(str, libs), "-o", str(out)])


def symbol(path, name):
    """The address and size of a symbol in an ELF file's symbol table, or of a C++ variable of that name
    with internal linkage."""
    data = pathlib.Path(path).read_bytes()
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3a)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    symtab = next(section for section in sections if section[1] == 2)
    strtab = sections[symtab[6]]
    for offset in range(symtab[4], symtab[4] + symtab[5], 24):
        string, _, _, _, value, size = struct.unpack_from("<IBBHQQ", data, offset)
        end = data.index(b"\0", strtab[4] + string)
        if data[strtab[4] + string:end].decode() in (name, f"_ZL{len(name)}{name}"):
            return value, size
    raise KeyError(name)


def add_process_parameters(path, name="processParameters"):
    """Adds the PT_SCE_PROCPARAM program header a PS5 executable has, over the named object, to a file
    whose program header table make_header_room moved to its end."""
    address, size = symbol(path, name)
    data = bytearray(pathlib.Path(path).read_bytes())
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    headers = [struct.unpack_from("<IIQQQQQQ", data, phoff + i * phentsize) for i in range(phnum)]
    load = next(h for h in headers if h[0] == 1 and h[3] <= address and address + size <= h[3] + h[5])
    table = data[phoff:phoff + phentsize * phnum]
    table += struct.pack("<IIQQQQQQ", 0x61000001, 4, load[2] + address - load[3], address, address, size, size, 8)
    del data[phoff:]
    struct.pack_into("<H", data, 0x38, phnum + 1)
    data += table
    pathlib.Path(path).write_bytes(data)


def make_header_room(path, extra=4):
    """Moves the program header table to the end of the file with extra PT_SCE_VERSION entries, which
    the relinker drops, as a PS5 executable has slots for the headers the relinker adds; PT_PHDR
    becomes one of them, since the relinker writes its own."""
    data = bytearray(pathlib.Path(path).read_bytes())
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    table = bytearray(data[phoff:phoff + phentsize * phnum])
    filler = struct.pack("<IIQQQQQQ", 0x6FFFFF01, 0, 0, 0, 0, 0, 0, 1)
    for index in range(phnum):
        if struct.unpack_from("<I", table, index * phentsize)[0] == 6:
            table[index * phentsize:(index + 1) * phentsize] = filler
    table += filler * extra
    while len(data) % 8:
        data.append(0)
    struct.pack_into("<Q", data, 0x20, len(data))
    struct.pack_into("<H", data, 0x38, phnum + extra)
    data += table
    pathlib.Path(path).write_bytes(data)
