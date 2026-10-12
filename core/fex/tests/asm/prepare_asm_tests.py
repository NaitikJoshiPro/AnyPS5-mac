"""Prepare FEX's x86-64 ASM unit tests for the aps5-fex-asm harness on macOS arm64.

    python3 prepare_asm_tests.py <FEX unittests/ASM> <list directory> <output directory>

Each test X.asm becomes X.bin, assembled by nasm from "BITS 64", the source and "ret" the way FEX's
unittests/ASM/CMakeLists.txt does, and X.cfg, the harness configuration, one entry per line:

    skip <reason>                      the test is not run (exit 125)
    fail <reason>                      the test cannot be prepared; the harness fails it with the reason
    known_failure <variant> <reason>   the test is expected to fail in this variant ("*" for every variant)
    code <address>                     where X.bin is loaded and run from, with RSP = 0
    map <address> <size>               memory to map read-write before the run
    memory <address> <hex bytes>       initial memory (MemoryData)
    option <FEX_NAME> <value>          FEX configuration from the test's Env
    feature <name>                     a HostFeatures entry
    reg <name> <value>...              an expected register (RegData), one value per 64-bit lane
    alt <name> <lane> <value>          another accepted value for one lane: the moved form of an address

macOS arm64 maps nothing below 4 GiB (__PAGEZERO), but the tests run their code at 0x10000, use
scratch memory at 0xe0000000 and 0xe8000000, and some declare MemoryRegions below 4 GiB or where the
macOS executable, dyld and the heap live. Everything the harness maps is moved up by
DELTA = 0x100_0000_0000, which keeps the low 32 bits of every address. (macOS 27 reserves a guard
region at 0x10_0000_0000-0x70_0000_0000 and the GPU carveout and malloc regions above it, so the
moved windows start at 1 TiB.) The source is preprocessed with
nasm -E, and each mov that loads an address of a mapped range into a register that is later used as
a pointer loads the moved address instead. Uses this cannot follow (a 32-bit address size, absolute
displacements, raw 0x67 prefixes, addresses kept in data, pointers truncated to 32 bits) make the
test unmovable: it must be listed in Disabled_Tests_macos with its reason, or the harness fails it.

The lists are FEX's own (Known_Failures, Known_Failures_jit, Disabled_Tests, and
Disabled_Tests_ARMv8.4, the class Scripts/ClassifyCPU.py gives Apple M-series cores) and ours in the
list directory, where every entry carries its reason:

    <test path relative to unittests/ASM>[ [<variant>]]: <reason>"""
import ast
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

DELTA = 0x100_0000_0000
CODE_BASE = 0x1_0000
PAGE = 0x4000
SCRATCH = 0xE000_0000
SCRATCH_SIZE = 10 * PAGE
SIB8 = 0xE800_0000
POINTER_WINDOW = (0xE000_0000, 0xE801_0000)
EXPECTATION_SLACK = 0x10_0000
VARIANTS = ("default", "full")
FEX_LISTS = ("Known_Failures", "Known_Failures_jit", "Disabled_Tests", "Disabled_Tests_ARMv8.4")
FEX_VARIANT_NAMES = {"jit_500_m": "default", "jit_1": "full"}
OUR_LISTS = ("Disabled_Tests_macos", "Known_Failures_macos")
FORMAT = "aps5-fex-asm 1"

GPR64 = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi"] + [f"r{i}" for i in range(8, 16)]
GPR32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"] + [f"r{i}d" for i in range(8, 16)]
GPR16 = ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di"] + [f"r{i}w" for i in range(8, 16)]
GPR8 = ["al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil"] + [f"r{i}b" for i in range(8, 16)]
REGISTERS = {}
for index, family in enumerate(GPR64):
    REGISTERS[family] = (family, 64)
    REGISTERS[GPR32[index]] = (family, 32)
    REGISTERS[GPR16[index]] = (family, 16)
    REGISTERS[GPR8[index]] = (family, 8)
for index in range(8, 16):
    REGISTERS[f"r{index}l"] = (f"r{index}", 8)
for high, family in (("ah", "rax"), ("ch", "rcx"), ("dh", "rdx"), ("bh", "rbx")):
    REGISTERS[high] = (family, 8)

PREFIXES = {"lock", "rep", "repe", "repz", "repne", "repnz", "a16", "a32", "a64", "o16", "o32", "o64",
            "xacquire", "xrelease", "bnd", "nobnd"}
DATA = {"db", "dw", "dd", "dq", "dt", "do", "dy", "dz"}
RESERVE = {"resb", "resw", "resd", "resq", "rest", "reso", "resy", "resz", "incbin"}
DIRECTIVES = DATA | RESERVE | {"equ", "times"}
SIZE_WORDS = re.compile(r"\b(byte|word|dword|qword|tword|oword|yword|zword|strict|ptr|near|far|short|to|rel|abs|nosplit)\b",
                        re.I)
SEGMENT = re.compile(r"\b[c-gs]s\s*:", re.I)
WORD = re.compile(r"[A-Za-z_.?@$][\w.?@$#~]*")
TOKEN = re.compile(r"([A-Za-z_.?@][\w.?@$#~]*)\s*")
LABEL = re.compile(r"([A-Za-z_.?@$][\w.?@$#~]*)\s*:(?!:)\s*")
NUMBER = re.compile(r"\$[0-9][0-9a-f_]*|0[xh][0-9a-f_]+|0[by][01_]+|0[oq][0-7_]+|0[dt][0-9_]+|[0-9][0-9a-f_]*[hqoybdt]?",
                    re.I)
STRING_OPERATION = re.compile(r"^(movs|cmps|stos|scas|lods|ins|outs)([bwdq]?)$")
STRING_REGISTERS = {"movs": ("rsi", "rdi"), "cmps": ("rsi", "rdi"), "stos": ("rdi",), "scas": ("rdi",),
                    "ins": ("rdi",), "lods": ("rsi",), "outs": ("rsi",)}
STACK = {"push", "pop", "call", "ret", "retn", "retf", "enter", "leave", "pushf", "pushfq", "pushfw", "popf", "popfq",
         "popfw", "iret", "iretq", "iretd", "iretw"}
IMPLICIT_POINTERS = {"xlat": ("rbx",), "xlatb": ("rbx",), "maskmovq": ("rdi",), "maskmovdqu": ("rdi",),
                     "vmaskmovdqu": ("rdi",), "clzero": ("rax",), "monitor": ("rax",), "monitorx": ("rax",)}
CLOBBERS = {"mul": ("rax", "rdx"), "div": ("rax", "rdx"), "idiv": ("rax", "rdx"), "cpuid": ("rax", "rbx", "rcx", "rdx"),
            "rdtsc": ("rax", "rdx"), "rdtscp": ("rax", "rdx", "rcx"), "rdpmc": ("rax", "rdx"), "xgetbv": ("rax", "rdx"),
            "rdpkru": ("rax", "rdx"), "syscall": ("rax", "rcx", "r11"), "cqo": ("rdx",), "cdq": ("rdx",), "cwd": ("rdx",),
            "cdqe": ("rax",), "cwde": ("rax",), "cmpxchg": ("rax",), "cmpxchg8b": ("rax", "rdx"),
            "cmpxchg16b": ("rax", "rdx"), "in": ("rax",), "pcmpestri": ("rcx",), "pcmpistri": ("rcx",),
            "vpcmpestri": ("rcx",), "vpcmpistri": ("rcx",)}
NO_DESTINATION = {"cmp", "test", "bt", "push", "jmp", "call", "out", "wrfsbase", "wrgsbase", "ptwrite", "umonitor",
                  "movdir64b", "enqcmd", "enqcmds", "bound"}
LINKING = {"cmp", "test", "sub", "sbb", "add", "adc", "xor", "and", "or"}
POINTER_ARITHMETIC = {"add", "sub", "adc", "sbb", "inc", "dec", "and", "or", "xor", "neg", "not", "shl", "shr", "sal",
                      "sar", "rol", "ror", "rcl", "rcr", "bts", "btr", "btc", "imul", "shld", "shrd", "xadd", "bswap",
                      "adcx", "adox"}
PREFIX_BYTES = {0x66, 0x67, 0xF2, 0xF3, 0xF0, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65}


class PrepareError(Exception):
    pass


def nasm_number(token):
    text = token.lower().replace("_", "")
    try:
        if text.startswith("$"):
            return int(text[1:], 16)
        if text.startswith("0x") or (text.startswith("0h") and len(text) > 2):
            return int(text[2:], 16)
        if text.startswith(("0b", "0y")) and len(text) > 2 and set(text[2:]) <= set("01"):
            return int(text[2:], 2)
        if text.startswith(("0o", "0q")) and len(text) > 2:
            return int(text[2:], 8)
        if text.startswith(("0d", "0t")) and len(text) > 2:
            return int(text[2:], 10)
        if text.endswith("h"):
            return int(text[:-1], 16)
        if text.endswith(("q", "o")):
            return int(text[:-1], 8)
        if text.endswith(("b", "y")) and set(text[:-1]) <= set("01"):
            return int(text[:-1], 2)
        if text.endswith(("d", "t")):
            return int(text[:-1], 10)
        return int(text, 10)
    except ValueError:
        return None


def numbers_in(text):
    for match in NUMBER.finditer(text):
        if match.start() > 0 and (text[match.start() - 1].isalnum() or text[match.start() - 1] in "_.@?"):
            continue
        yield match


def constant(expression):
    """The value of a nasm expression made of numbers and arithmetic only, or None."""
    text = SIZE_WORDS.sub(" ", expression).strip()
    if not text or any(mark in text for mark in "[]'\"`"):
        return None
    pieces, position = [], 0
    for match in numbers_in(text):
        value = nasm_number(match.group(0))
        if value is None:
            return None
        pieces.append(text[position:match.start()])
        pieces.append(str(value))
        position = match.end()
    pieces.append(text[position:])
    python = "".join(pieces)
    if re.search(r"[A-Za-z_.?@$]", python):
        return None
    try:
        return evaluate(ast.parse(python.replace("//", "/"), mode="eval").body) & ((1 << 64) - 1)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return None


def evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
        value = evaluate(node.operand)
        return {ast.USub: -value, ast.UAdd: value, ast.Invert: ~value}[type(node.op)]
    if isinstance(node, ast.BinOp):
        left, right = evaluate(node.left), evaluate(node.right)
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise ZeroDivisionError()
        operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right,
                      ast.Div: lambda: left // right, ast.Mod: lambda: left % right,
                      ast.LShift: lambda: left << right, ast.RShift: lambda: left >> right,
                      ast.BitOr: lambda: left | right, ast.BitAnd: lambda: left & right,
                      ast.BitXor: lambda: left ^ right}
        if type(node.op) in operations and not (isinstance(node.op, (ast.LShift, ast.RShift)) and not 0 <= right < 128):
            return operations[type(node.op)]()
    raise ValueError("not a constant expression")


def strip_comment(line):
    quote = None
    for index, character in enumerate(line):
        if quote:
            if character == quote:
                quote = None
        elif character in "'\"`":
            quote = character
        elif character == ";":
            return line[:index]
    return line


def split_operands(text):
    operands, depth, current = [], 0, ""
    for character in text:
        if character in "[(":
            depth += 1
        elif character in "])":
            depth -= 1
        if character == "," and depth == 0:
            operands.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        operands.append(current.strip())
    return operands


@dataclasses.dataclass
class Instruction:
    head: str
    prefixes: list
    mnemonic: str
    operands: list


def parse_line(line):
    """Splits a preprocessed source line into its head (labels and prefixes), mnemonic and operands."""
    text = strip_comment(line).rstrip()
    position = len(text) - len(text.lstrip())
    if position == len(text) or text[position] in "%[":
        return None
    while True:
        match = LABEL.match(text, position)
        if not match or match.group(1).lower() in PREFIXES:
            break
        position = match.end()
    prefixes = []
    while True:
        match = TOKEN.match(text, position)
        if not match:
            return None
        if match.group(1).lower() not in PREFIXES:
            break
        prefixes.append(match.group(1).lower())
        position = match.end()
    head, mnemonic, rest = text[:position], match.group(1).lower(), text[match.end():]
    following = TOKEN.match(rest)
    if mnemonic not in DIRECTIVES and following and following.group(1).lower() in DIRECTIVES:
        head = text[:match.end()]
        mnemonic, rest = following.group(1).lower(), rest[following.end():]
    if mnemonic == "times":
        tokens = rest.split()
        for count in range(1, len(tokens)):
            expression = " ".join(tokens[:count])
            if expression.count("(") == expression.count(")") and re.match(r"[A-Za-z_.]", tokens[count]) and \
                    expression[-1] not in "+-*/%|&^~<>(":
                nested = parse_line(" ".join(tokens[count:]))
                return Instruction(head, prefixes + nested.prefixes, nested.mnemonic, nested.operands) if nested else None
        return None
    return Instruction(head, prefixes, mnemonic, split_operands(rest))


def register(operand):
    return REGISTERS.get(SIZE_WORDS.sub(" ", operand).strip().lower())


def memory_inner(operand):
    return operand[operand.index("[") + 1:operand.rindex("]") if "]" in operand else len(operand)]


def memory_registers(operand):
    return [REGISTERS[word.lower()] for word in WORD.findall(SEGMENT.sub(" ", memory_inner(operand)))
            if word.lower() in REGISTERS]


def memory_literals(operand):
    inner = SEGMENT.sub(" ", SIZE_WORDS.sub(" ", memory_inner(operand)))
    return [value for value in (nasm_number(match.group(0)) for match in numbers_in(inner)) if value is not None]


def segment_relative(operand):
    return bool(re.search(r"\b[fg]s\s*:", operand, re.I))


@dataclasses.dataclass
class Origin:
    """A mov of a constant into a register; the constant is moved by DELTA when the register is
    used as a pointer into a mapped range."""
    number: int
    family: str
    value: int
    used: bool = False
    hazards: list = dataclasses.field(default_factory=list)
    links: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Held:
    """What a register holds: a value computed from an origin's constant, the value itself when it
    is known, and whether a 32-bit operation dropped its upper half."""
    origin: Origin
    value: object
    truncated: bool = False


@dataclasses.dataclass
class Analysis:
    origins: list
    problems: list


class Ranges:
    def __init__(self, ranges):
        self.ranges = list(ranges)

    def __contains__(self, value):
        return value is not None and any(begin <= value <= end for begin, end in self.ranges)


def effective_address(inner, state):
    """The address a memory operand reaches when every register in it holds a known value."""
    def substitute(match):
        entry = REGISTERS.get(match.group(0).lower())
        if entry is None:
            return match.group(0)
        held = state.get(entry[0])
        if held is None or held.value is None:
            raise LookupError(entry[0])
        return str(held.value & ((1 << entry[1]) - 1))
    try:
        return constant(WORD.sub(substitute, inner))
    except LookupError:
        return None


def unscaled_registers(inner):
    found = []
    for match in WORD.finditer(inner):
        entry = REGISTERS.get(match.group(0).lower())
        if entry and entry[1] == 64 and not re.search(r"\*\s*$", inner[:match.start()]) and \
                not re.match(r"\s*\*", inner[match.end():]):
            found.append(entry[0])
    return found


def analyse(lines, pointer_ranges):
    """Follows, in source order, which registers hold addresses of the mapped ranges, and returns
    the movs whose constant must be moved and the uses that cannot be moved."""
    state, origins, problems, stack = {}, [], [], []
    seen_halt = False

    def access(number, why, registers, address, address32, unscaled=()):
        pointing = [(family, width) for family, width in registers
                    if family in state and state[family].origin.value in pointer_ranges]
        for family, width in pointing:
            held = state[family]
            held.origin.used = True
            if width == 32 or address32 or held.truncated:
                held.origin.hazards.append(f"line {number + 1}: {why} uses the pointer from line "
                                           f"{held.origin.number + 1} truncated to 32 bits")
        if pointing or address is None:
            return
        if address32 or any(width == 32 for _, width in registers):
            if address & 0xFFFF_FFFF in pointer_ranges:
                problems.append(f"line {number + 1}: {why} uses the 32-bit address {address & 0xFFFF_FFFF:#x}")
            return
        if address in pointer_ranges:
            candidates = [family for family in unscaled if family in state and not state[family].truncated]
            if candidates:
                state[candidates[0]].origin.used = True
            else:
                problems.append(f"line {number + 1}: {why} reaches {address:#x} through values that cannot be moved")

    def implicit(number, why, family, address32):
        held = state.get(family)
        address = held.value if held else None
        access(number, why, [(family, 32 if address32 else 64)], address, address32, (family,))

    def clear(family, width):
        if width >= 32:
            state.pop(family, None)

    def update(family, width, value):
        held = state[family]
        if width == 64:
            state[family] = Held(held.origin, value & ((1 << 64) - 1) if value is not None else None, held.truncated)
        elif width == 32:
            state[family] = Held(held.origin, value & 0xFFFF_FFFF if value is not None else None, True)
        else:
            state[family] = Held(held.origin, None, held.truncated)

    for number, line in enumerate(lines):
        instruction = parse_line(line)
        if instruction is None:
            continue
        mnemonic, operands, prefixes = instruction.mnemonic, instruction.operands, instruction.prefixes
        if mnemonic == "hlt":
            seen_halt = True
        if mnemonic == "equ" or mnemonic in RESERVE:
            continue
        if mnemonic in DATA:
            values = [constant(operand) for operand in operands]
            if mnemonic in ("dd", "dq"):
                for value in values:
                    if value in pointer_ranges:
                        problems.append(f"line {number + 1}: {mnemonic} keeps the address {value:#x} in data")
            if mnemonic == "db" and not seen_halt:
                for value in values:
                    if value not in PREFIX_BYTES:
                        break
                    if value == 0x67:
                        problems.append(f"line {number + 1}: raw-encoded instruction with a 0x67 address-size prefix")
                        break
            continue
        address32 = "a32" in prefixes
        for operand in operands:
            if "[" not in operand or segment_relative(operand):
                continue
            inner = SIZE_WORDS.sub(" ", memory_inner(operand))
            registers = memory_registers(operand)
            if not registers:
                for value in memory_literals(operand):
                    if value in pointer_ranges:
                        problems.append(f"line {number + 1}: absolute address {value:#x} in a memory operand")
            if mnemonic in ("lea", "nop"):
                continue
            access(number, mnemonic, registers, effective_address(inner, state), address32, unscaled_registers(inner))
        string = STRING_OPERATION.match(mnemonic)
        if string and not operands:
            for family in STRING_REGISTERS[string.group(1)]:
                implicit(number, mnemonic, family, address32)
            if any(prefix.startswith("rep") for prefix in prefixes):
                state.pop("rcx", None)
            if string.group(1) == "lods":
                clear("rax", 64 if string.group(2) in ("d", "q") else 8)
            continue
        if mnemonic in STACK:
            implicit(number, mnemonic, "rsp", False)
            if mnemonic == "leave":
                implicit(number, mnemonic, "rbp", False)
                if "rbp" in state:
                    state["rsp"] = state["rbp"]
                else:
                    state.pop("rsp", None)
                state.pop("rbp", None)
            elif mnemonic == "enter":
                state.pop("rbp", None)
            elif mnemonic == "push":
                pushed = register(operands[0]) if operands and "[" not in operands[0] else None
                stack.append(state.get(pushed[0]) if pushed and pushed[1] == 64 else None)
            elif mnemonic in ("pushf", "pushfq", "pushfw"):
                stack.append(None)
            elif mnemonic in ("popf", "popfq", "popfw"):
                if stack:
                    stack.pop()
            elif mnemonic == "pop":
                popped = stack.pop() if stack else None
                target = register(operands[0]) if operands and "[" not in operands[0] else None
                if target and target[1] == 64:
                    if popped:
                        state[target[0]] = popped
                    else:
                        state.pop(target[0], None)
            continue
        for family in IMPLICIT_POINTERS.get(mnemonic, ()):
            implicit(number, mnemonic, family, address32)
        if mnemonic in ("wrfsbase", "wrgsbase", "movdir64b") and operands and register(operands[0]):
            family, width = register(operands[0])
            implicit(number, mnemonic, family, width == 32)
        if mnemonic == "imul" and len(operands) == 1:
            for family in ("rax", "rdx"):
                state.pop(family, None)
            continue
        for family in CLOBBERS.get(mnemonic, ()):
            state.pop(family, None)
        if mnemonic in LINKING and len(operands) == 2 and all("[" not in operand for operand in operands):
            pair = [state.get(entry[0]) for entry in map(register, operands) if entry]
            if len(pair) == 2 and all(held and held.origin.value in pointer_ranges for held in pair):
                pair[0].origin.links.append(pair[1].origin)
                pair[1].origin.links.append(pair[0].origin)
        if not operands or mnemonic in NO_DESTINATION or mnemonic.startswith(("j", "loop", "set", "prefetch")):
            continue
        destination = register(operands[0]) if "[" not in operands[0] else None
        if destination is None:
            continue
        family, width = destination
        source = operands[1] if len(operands) > 1 else ""
        source_register = register(source) if source and "[" not in source else None
        immediate = constant(source) if source and "[" not in source and not source_register else None
        if mnemonic == "mov":
            if source_register:
                held = state.get(source_register[0])
                if held and source_register[1] >= 32 and width >= 32:
                    narrow = width == 32 or source_register[1] == 32
                    value = held.value & 0xFFFF_FFFF if narrow and held.value is not None else held.value
                    state[family] = Held(held.origin, value, held.truncated or narrow)
                else:
                    clear(family, width)
                continue
            if immediate is not None and width >= 32:
                origin = Origin(number, family, immediate & ((1 << width) - 1))
                origins.append(origin)
                state[family] = Held(origin, origin.value)
                continue
            if immediate in pointer_ranges:
                problems.append(f"line {number + 1}: address {immediate:#x} loaded into a {width}-bit register")
            clear(family, width)
            continue
        if mnemonic == "lea":
            inner = SIZE_WORDS.sub(" ", memory_inner(source)) if "[" in source else ""
            if width == 32 and inner and not memory_registers(source) and constant(inner) is None:
                problems.append(f"line {number + 1}: lea of a code or data label into a 32-bit register")
            held = [entry for entry in (memory_registers(source) if inner else []) if entry[0] in state]
            if held and width >= 32 and not segment_relative(source):
                first = state[held[0][0]]
                narrow = width == 32 or held[0][1] == 32 or address32
                address = effective_address(inner, state)
                if narrow and address is not None:
                    address &= 0xFFFF_FFFF
                state[family] = Held(first.origin, address, first.truncated or narrow)
            else:
                clear(family, width)
            continue
        if mnemonic == "xchg" and source_register:
            mine, theirs = state.pop(family, None), state.pop(source_register[0], None)
            if theirs:
                state[family] = theirs
            if mine:
                state[source_register[0]] = mine
            continue
        if mnemonic.startswith("cmov") and source_register:
            if source_register[0] in state and width == 64:
                state[family] = state[source_register[0]]
            continue
        if mnemonic in ("xor", "sub") and source_register and source_register[0] == family:
            clear(family, width)
            continue
        if mnemonic in POINTER_ARITHMETIC:
            held = state.get(family)
            if held:
                value = None
                if held.value is not None and mnemonic in ("add", "sub", "inc", "dec") and width >= 32:
                    step = {"inc": 1, "dec": -1}.get(mnemonic, immediate)
                    if step is not None:
                        value = held.value + (-step if mnemonic == "sub" else step)
                update(family, width, value)
            elif source_register and source_register[0] in state and width == 64 and mnemonic in ("add", "adc", "or"):
                state[family] = Held(state[source_register[0]].origin, None, state[source_register[0]].truncated)
            continue
        if mnemonic == "mulx" and len(operands) > 1 and register(operands[1]):
            clear(*register(operands[1]))
        clear(family, width)
    pending = [origin for origin in origins if origin.used]
    while pending:
        for linked in pending.pop().links:
            if not linked.used:
                linked.used = True
                pending.append(linked)
    for origin in origins:
        if origin.used and origin.hazards:
            problems.extend(origin.hazards)
    return Analysis([origin for origin in origins if origin.used and not origin.hazards], problems)


def rewrite(lines, analysis):
    """Loads the moved constant in each mov the analysis found, into the 64-bit register."""
    lines = list(lines)
    for origin in analysis.origins:
        instruction = parse_line(lines[origin.number])
        lines[origin.number] = f"{instruction.head}mov {origin.family}, {origin.value + DELTA:#x}"
    return lines


def parse_config(text):
    parts = text.split("%ifdef CONFIG")
    if len(parts) < 2:
        return {}
    body = parts[1].split("%endif")
    if len(body) < 2:
        return {}
    try:
        config = json.loads(body[0].strip())
    except json.JSONDecodeError as error:
        raise PrepareError(f"the CONFIG block is not JSON: {error}") from error
    return {key.upper(): value for key, value in config.items()}


def page_round(begin, end):
    return begin & ~(PAGE - 1), (end + PAGE - 1) & ~(PAGE - 1)


def memory_bytes(text):
    """A MemoryData value as Scripts/json_config_parse.py reads it: space-separated hex numbers,
    each stored little-endian."""
    result = bytearray()
    for number in text.split(" "):
        if text.startswith("0x"):
            number = number[2:]
        while number:
            result.append(int(number[-2:], 16))
            number = number[:-2]
    return bytes(result)


def read_fex_list(path):
    """FEX's list format: "Test_<path>" or "<config>/Test_64Bit_<path>" entries; the comment lines
    above a group of entries give their reason."""
    entries, reason = {}, None
    if not path.exists():
        return entries
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            reason = None
            continue
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            reason = f"{reason}; {text}" if reason else text
            continue
        variant, name = "*", line
        if "/Test_64Bit_" in name:
            prefix, name = name.split("/Test_64Bit_", 1)
            variant = FEX_VARIANT_NAMES.get(prefix)
            if variant is None:
                continue
        elif name.startswith("Test_"):
            name = name[len("Test_"):]
        else:
            continue
        entries.setdefault(name, []).append((variant, f"FEX {path.name}: {reason or 'no reason given'}"))
    return entries


def read_our_list(path, nasm_major):
    """Our list format: "<test>.asm[ [<qualifiers>]]: <reason>". Qualifiers, comma-separated, are a
    variant (known failures only) and nasm2 or nasm3, for entries that hold with one major version of
    the assembler only."""
    entries = {}
    if not path.exists():
        return entries
    pattern = re.compile(r"^(\S+?\.asm)(?:\s+\[([\w, ]+)\])?:\s*(\S.*)$")
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            raise PrepareError(f"{path.name}:{number}: expected '<test>.asm[ [<qualifiers>]]: <reason>', got {line!r}")
        qualifiers = [word.strip() for word in (match.group(2) or "").split(",") if word.strip()]
        unknown = [word for word in qualifiers if word not in VARIANTS + ("nasm2", "nasm3")]
        if unknown:
            raise PrepareError(f"{path.name}:{number}: unknown qualifier {unknown[0]!r}; use {', '.join(VARIANTS)}, nasm2 or nasm3")
        variants = [word for word in qualifiers if word in VARIANTS]
        if variants and path.name.startswith("Disabled"):
            raise PrepareError(f"{path.name}:{number}: a disabled test is disabled in every variant")
        assemblers = [word for word in qualifiers if word.startswith("nasm")]
        if assemblers and f"nasm{3 if nasm_major >= 3 else 2}" not in assemblers:
            entries.setdefault(match.group(1), [])
            continue
        for variant in variants or ["*"]:
            entries.setdefault(match.group(1), []).append((variant, f"{path.name}: {match.group(3)}"))
    return entries


@dataclasses.dataclass
class Lists:
    disabled: dict
    known: dict
    ours: set


def read_lists(asm_root, list_directory, nasm_major):
    lists = Lists({}, {}, set())
    for name in FEX_LISTS + OUR_LISTS:
        target = lists.disabled if name.startswith("Disabled") else lists.known
        entries = read_fex_list(asm_root / name) if name in FEX_LISTS else read_our_list(list_directory / name, nasm_major)
        for test, found in entries.items():
            if found:
                target.setdefault(test, []).extend(found)
            if name in OUR_LISTS:
                lists.ours.add(test)
    return lists


def nasm_major(nasm):
    if nasm is None:
        return 0
    match = re.search(r"version (\d+)\.", subprocess.run([nasm, "-v"], capture_output=True, text=True).stdout)
    return int(match.group(1)) if match else 0


def run_nasm(nasm, arguments):
    result = subprocess.run([nasm] + arguments, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise PrepareError(f"nasm failed: {result.stderr.strip()[:400]}")
    return result


def listing_offsets(path):
    """Maps each source line of a nasm listing to the offset of the first byte it emits."""
    offsets = {}
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        match = re.match(r"^\s*(\d+) ([0-9A-F]{8}) [0-9A-F]", line)
        if match:
            offsets.setdefault(int(match.group(1)), int(match.group(2), 16))
    return offsets


@dataclasses.dataclass
class Assembled:
    code: bytes
    original_size: int
    relocations: int
    code_map: dict
    instructions: list


VECTOR_REGISTER = re.compile(r"\b(ymm|xmm|mm)\d+\b", re.I)


def instruction_mix(lines):
    """The distinct instructions of a test as tokens: optional "lock:" or "rep:" prefix, the
    mnemonic, and "@ymm", "@xmm" or "@mm" for the widest vector register among its operands."""
    tokens = []
    for line in lines:
        instruction = parse_line(line)
        if instruction is None or instruction.mnemonic in DIRECTIVES:
            continue
        kinds = {match.group(1).lower() for operand in instruction.operands for match in VECTOR_REGISTER.finditer(operand)}
        width = next((f"@{kind}" for kind in ("ymm", "xmm", "mm") if kind in kinds), "")
        prefix = "lock:" if "lock" in instruction.prefixes else "rep:" if any(
            prefix.startswith("rep") for prefix in instruction.prefixes) else ""
        tokens.append(f"{prefix}{instruction.mnemonic}{width}")
    return sorted(set(tokens))


def assemble(nasm, asm_root, text, pointer_ranges):
    """Assembles the test as FEX does and, when it uses mapped addresses, again with the moved
    addresses; returns the code, the size of FEX's binary and the map from FEX's code offsets
    to ours for the instructions whose offset changed."""
    with tempfile.TemporaryDirectory(prefix="aps5-asm-") as scratch:
        scratch = pathlib.Path(scratch)
        wrapped = scratch / "test.asm"
        wrapped.write_text("BITS 64\n" + (text if text.endswith("\n") else text + "\n") + "ret\n\n")
        includes = str(asm_root / "Includes") + "/"
        original = scratch / "original.bin"
        run_nasm(nasm, ["-i", includes, str(wrapped), "-o", str(original)])
        lines = [line for line in run_nasm(nasm, ["-E", "-i", includes, str(wrapped)]).stdout.splitlines()
                 if not line.startswith("%line")]
        analysis = analyse(lines, pointer_ranges)
        if analysis.problems:
            raise Unmovable("; ".join(dict.fromkeys(analysis.problems)))
        mix = instruction_mix(lines)
        if not analysis.origins:
            code = original.read_bytes()
            return Assembled(code, len(code), 0, {}, mix)
        old, new = scratch / "old", scratch / "new"
        old.with_suffix(".asm").write_text("\n".join(lines) + "\n")
        new.with_suffix(".asm").write_text("\n".join(rewrite(lines, analysis)) + "\n")
        for path in (old, new):
            run_nasm(nasm, [str(path.with_suffix(".asm")), "-o", str(path.with_suffix(".bin")), "-l",
                            str(path.with_suffix(".lst"))])
        if old.with_suffix(".bin").read_bytes() != original.read_bytes():
            raise PrepareError("the preprocessed source does not assemble to the bytes of the original")
        old_offsets, new_offsets = listing_offsets(old.with_suffix(".lst")), listing_offsets(new.with_suffix(".lst"))
        code_map = {}
        for line_number, offset in sorted(old_offsets.items()):
            if line_number in new_offsets:
                code_map.setdefault(offset, new_offsets[line_number])
        return Assembled(new.with_suffix(".bin").read_bytes(), original.stat().st_size, len(analysis.origins), code_map, mix)


class Unmovable(Exception):
    pass


def expectations(config, data_ranges, assembled):
    data_ranges = Ranges([(begin - EXPECTATION_SLACK, end + EXPECTATION_SLACK) for begin, end in data_ranges.ranges])
    lines = []
    registers = config.get("REGDATA") or {}
    if not isinstance(registers, dict):
        raise PrepareError("RegData is not an object")
    for name, values in registers.items():
        values = values if isinstance(values, list) else [values]
        numbers = [int(str(value), 0) & ((1 << 64) - 1) for value in values]
        lines.append(f"reg {name.upper()} " + " ".join(f"{value:#x}" for value in numbers))
        for lane, value in enumerate(numbers):
            moved = set()
            if value in data_ranges:
                moved.add(value + DELTA)
            if CODE_BASE <= value <= CODE_BASE + assembled.original_size:
                offset = value - CODE_BASE
                moved.add(DELTA + CODE_BASE + (assembled.code_map.get(offset, offset) if assembled.relocations else offset))
            for alternative in sorted(moved - {value}):
                lines.append(f"alt {name.upper()} {lane} {alternative:#x}")
    return lines


@dataclasses.dataclass
class Prepared:
    status: str
    detail: str = ""
    relocations: int = 0


def body(text, nasm, asm_root):
    """The part of a test's configuration that does not depend on the lists, and the code to load
    (None when the test is not run)."""
    if nasm is None:
        return ["skip nasm was not found (set NASM or put nasm on PATH)"], None
    try:
        config = parse_config(text)
        if str(config.get("MODE", "64BIT")).upper() != "64BIT":
            return ["skip 32-bit mode is not run"], None
        regions = []
        for key, value in (config.get("MEMORYREGIONS") or {}).items():
            begin = int(str(key), 0)
            regions.append((begin, begin + int(str(value), 0)))
        data_ranges = Ranges([POINTER_WINDOW] + regions)
        assembled = assemble(nasm, asm_root, text, data_ranges)
        lines = [f"relocations {assembled.relocations}", "instructions " + " ".join(assembled.instructions),
                 f"code {CODE_BASE + DELTA:#x}"]
        for begin, end in [(CODE_BASE, CODE_BASE + max(len(assembled.code), 1)), (SCRATCH, SCRATCH + SCRATCH_SIZE),
                           (SIB8 - PAGE, SIB8 + PAGE)] + regions:
            low, high = page_round(begin, end)
            lines.append(f"map {low + DELTA:#x} {high - low:#x}")
        for key, value in (config.get("MEMORYDATA") or {}).items():
            lines.append(f"memory {int(str(key), 0) + DELTA:#x} {memory_bytes(str(value)).hex()}")
        for key, value in (config.get("ENV") or {}).items():
            lines.append(f"option {key} {value}")
        for feature in config.get("HOSTFEATURES") or []:
            lines.append(f"feature {str(feature).upper()}")
        lines.extend(expectations(config, data_ranges, assembled))
        return lines, assembled.code
    except Unmovable as error:
        return [f"fail uses low addresses that cannot be moved, list it in Disabled_Tests_macos: {error}"], None
    except (PrepareError, subprocess.TimeoutExpired, ValueError, TypeError) as error:
        return [f"fail cannot prepare: {error}"], None


def replace(path, data):
    temporary = path.with_name(path.name + ".tmp")
    if isinstance(data, bytes):
        temporary.write_bytes(data)
    else:
        temporary.write_text(data)
    temporary.replace(path)


def prepare(asm_root, rel, lists, output, nasm, fingerprint):
    """Writes X.cfg from the lists and the cached X.body, which is assembled again only when the
    test, the includes, nasm or this script change."""
    text = (asm_root / rel).read_text(errors="replace")
    cfg_path, bin_path, body_path = (output / (rel + suffix) for suffix in (".cfg", ".bin", ".body"))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if rel in lists.disabled:
        lines = ["skip disabled by " + "; ".join(reason for _, reason in lists.disabled[rel])]
    else:
        digest = f"hash {hashlib.sha256(fingerprint + text.encode()).hexdigest()}"
        cached = body_path.read_text().splitlines() if body_path.exists() else []
        if cached[:1] == [digest] and (bin_path.exists() or not any(line.startswith("code ") for line in cached)):
            lines = cached[1:]
        else:
            lines, code = body(text, nasm, asm_root)
            if code is not None:
                replace(bin_path, code)
            elif bin_path.exists():
                bin_path.unlink()
            replace(body_path, "\n".join([digest] + lines) + "\n")
        lines = [f"known_failure {variant} {reason}" for variant, reason in lists.known.get(rel, [])] + lines
    content = "\n".join([FORMAT, f"test {rel}"] + lines) + "\n"
    if not cfg_path.exists() or cfg_path.read_text() != content:
        replace(cfg_path, content)
    fields = dict(line.split(" ", 1) for line in lines if " " in line)
    status = "skip" if "skip" in fields else "fail" if "fail" in fields else "run"
    return Prepared(status, fields.get(status, ""), int(fields.get("relocations", 0)))


def tests_in(asm_root):
    return sorted(str(path.relative_to(asm_root)) for path in asm_root.rglob("*.asm"))


def fingerprint(asm_root, nasm):
    digest = hashlib.sha256(pathlib.Path(__file__).read_bytes())
    for path in sorted((asm_root / "Includes").glob("*")):
        digest.update(path.name.encode() + path.read_bytes())
    if nasm:
        digest.update(subprocess.run([nasm, "-v"], capture_output=True, text=True).stdout.encode())
    return digest.digest()


def main(arguments):
    if len(arguments) != 3:
        sys.stderr.write("usage: prepare_asm_tests.py <FEX unittests/ASM> <list directory> <output directory>\n")
        return 2
    asm_root, list_directory, output = (pathlib.Path(argument).resolve() for argument in arguments)
    nasm = os.environ.get("NASM") or shutil.which("nasm")
    try:
        lists = read_lists(asm_root, list_directory, nasm_major(nasm))
    except PrepareError as error:
        sys.stderr.write(f"error: {error}\n")
        return 1
    tests = tests_in(asm_root)
    missing = sorted(lists.ours - set(tests))
    for name in missing:
        sys.stderr.write(f"error: {name} is listed in {list_directory} but is not a test in {asm_root}\n")
    marks = fingerprint(asm_root, nasm)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        futures = {pool.submit(prepare, asm_root, rel, lists, output, nasm, marks): rel for rel in tests}
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    with open(output / "index.tsv", "w") as index:
        index.write("test\tstatus\trelocations\tdetail\n")
        for rel in tests:
            index.write(f"{rel}\t{results[rel].status}\t{results[rel].relocations}\t{results[rel].detail}\n")
    counts = {}
    for result in results.values():
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"prepared {len(tests)} tests in {output}: " + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())))
    for rel in tests:
        if results[rel].status == "fail":
            print(f"cannot run {rel}: {results[rel].detail}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
