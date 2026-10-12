"""Small PS5-style x86-64 ELF programs, built byte by byte, for the arm64 guest runner."""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../relinker/relinker/tests/macos"))
from nid import nid

PT_LOAD = 1
PT_DYNAMIC = 2
PT_SCE_VERSION = 0x6FFFFF01
CODE = 0x4000
DATA = 0x5000

SUB_RSP_8 = b"\x48\x83\xec\x08"
ADD_RSP_16 = b"\x48\x83\xc4\x10"
PUSH_5 = b"\x6a\x05"
PUSH_6 = b"\x6a\x06"
RET = b"\xc3"
UD2 = b"\x0f\x0b"
XOR_EAX_EAX = b"\x31\xc0"
XOR_ESI_ESI = b"\x31\xf6"
XOR_EDI_EDI = b"\x31\xff"
TEST_EAX_EAX = b"\x85\xc0"
TEST_RAX_RAX = b"\x48\x85\xc0"
MOV_EDI_EAX = b"\x89\xc7"
SHR_EDI_31 = b"\xc1\xef\x1f"
IMUL_EDI_EDI_10 = b"\x6b\xff\x0a"
ADDSD_XMM0_XMM0 = b"\xf2\x0f\x58\xc0"
CVTTSD2SI_EDI_XMM0 = b"\xf2\x0f\x2c\xf8"
MOV_EAX_FROM_RDI = b"\x8b\x07"
SUB_EAX_FROM_RSI = b"\x2b\x06"
MOV_RAX_FROM_FS_0 = b"\x64\x48\x8b\x04\x25\x00\x00\x00\x00"
CMP_RAX_WITH_RAX_POINTEE = b"\x48\x3b\x00"
LEA_RAX_RDI_PLUS_22 = b"\x48\x8d\x47\x16"
CMP_RAX_RCX = b"\x48\x39\xc8"

MOV_EAX = b"\xb8"
MOV_ECX = b"\xb9"
MOV_EDX = b"\xba"
MOV_ESI = b"\xbe"
MOV_EDI = b"\xbf"
MOV_R9D = b"\x41\xb9"
MOV_RCX_64 = b"\x48\xb9"
CMP_EAX = b"\x3d"

CALL_RIP = b"\xff\x15"
LEA_RCX_RIP = b"\x48\x8d\x0d"
LEA_RDX_RIP = b"\x48\x8d\x15"
LEA_RSI_RIP = b"\x48\x8d\x35"
LEA_RDI_RIP = b"\x48\x8d\x3d"
LEA_R8_RIP = b"\x4c\x8d\x05"
MOV_EDI_RIP = b"\x8b\x3d"
ADD_EDI_RIP = b"\x03\x3d"
MOV_RAX_RIP = b"\x48\x8b\x05"
MOV_RDI_RIP = b"\x48\x8b\x3d"
MOVZX_EAX_WORD_RIP = b"\x0f\xb7\x05"
MOVSD_XMM0_RIP = b"\xf2\x0f\x10\x05"
FSTP_TWORD_RIP = b"\xdb\x3d"

JZ = b"\x74"
JNZ = b"\x75"


class Code:
    """x86-64 code at CODE. RIP-relative operands are given as addresses, and jumps and the addresses
    of code as labels, resolved by bytes()."""

    def __init__(self):
        self.out = bytearray()
        self.labels = {}
        self.jumps = []
        self.addresses = []

    def emit(self, *parts):
        for part in parts:
            self.out += part

    def immediate(self, opcode, value):
        self.out += opcode + struct.pack("<I", value)

    def immediate64(self, opcode, value):
        self.out += opcode + struct.pack("<Q", value)

    def rip(self, opcode, address):
        end = CODE + len(self.out) + len(opcode) + 4
        self.out += opcode + struct.pack("<i", address - end)

    def rip_label(self, opcode, label):
        self.out += opcode
        self.addresses.append((len(self.out), label))
        self.out += bytes(4)

    def jump(self, opcode, label):
        self.out += opcode
        self.jumps.append((len(self.out), label))
        self.out += bytes(1)

    def label(self, name):
        self.labels[name] = len(self.out)

    def bytes(self):
        for position, label in self.jumps:
            self.out[position] = self.labels[label] - (position + 1)
        for position, label in self.addresses:
            struct.pack_into("<i", self.out, position, self.labels[label] - (position + 4))
        return bytes(self.out)


def slot(index):
    return DATA + 8 * index


def data(offset):
    return DATA + 0x100 + offset


def program(code, imports, contents=b"", slots=8):
    """An executable with the given code at CODE, which calls imports through the GOT slots at
    slot(index), and contents at data(0)."""
    image = bytearray(0x6000)
    image[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<HHIQQQIHHHHHH", image, 16, 3, 62, 1, CODE, 64, 0, 0, 64, 56, slots, 64, 0, 0)
    struct.pack_into("<IIQQQQQQ", image, 64, PT_LOAD, 5, CODE, CODE, CODE, 0x1000, 0x1000, 0x1000)
    struct.pack_into("<IIQQQQQQ", image, 120, PT_LOAD, 6, DATA, DATA, DATA, 0x1000, 0x1000, 0x1000)
    body = code.bytes()
    image[CODE:CODE + len(body)] = body
    image[data(0):data(len(contents))] = contents
    strings = b"\0libc.prx\0"
    offsets = []
    for name in imports:
        offsets.append(len(strings))
        strings += (nid(name) + "#A#B").encode() + b"\0"
    image[0x5A00:0x5A00 + len(strings)] = strings
    for index, offset in enumerate(offsets):
        struct.pack_into("<IBBHQQ", image, 0x5B00 + 24 * (index + 1), offset, 0x12, 0, 0, 0, 0)
        struct.pack_into("<QQq", image, 0x5C00 + 24 * index, slot(index), ((index + 1) << 32) | 6, 0)
    tags = [(1, 1), (5, 0x5A00), (10, len(strings)), (6, 0x5B00), (11, 24), (0x6100003f, 24 * (len(imports) + 1)),
            (7, 0x5C00), (8, 24 * len(imports)), (9, 24), (0, 0)]
    for index, tag in enumerate(tags):
        struct.pack_into("<qQ", image, 0x5800 + index * 16, *tag)
    struct.pack_into("<IIQQQQQQ", image, 176, PT_DYNAMIC, 6, 0x5800, 0x5800, 0x5800, len(tags) * 16, len(tags) * 16, 8)
    for index in range(3, slots):
        struct.pack_into("<IIQQQQQQ", image, 64 + index * 56, PT_SCE_VERSION, 0, 0, 0, 0, 0, 0, 1)
    return image


def hello(message=b"hello from x86-64 guest code on arm64\0", status=42):
    """puts(message); exit(status)"""
    code = Code()
    code.emit(SUB_RSP_8)
    code.rip(LEA_RDI_RIP, data(0))
    code.rip(CALL_RIP, slot(0))
    code.immediate(MOV_EDI, status)
    code.rip(CALL_RIP, slot(1))
    code.emit(UD2)
    return program(code, ["puts", "exit"], message)


def floating():
    """exit(2 * atof("20.5")): a double comes back in xmm0."""
    code = Code()
    code.emit(SUB_RSP_8)
    code.rip(LEA_RDI_RIP, data(0))
    code.rip(CALL_RIP, slot(0))
    code.emit(ADDSD_XMM0_XMM0, CVTTSD2SI_EDI_XMM0)
    code.rip(CALL_RIP, slot(1))
    code.emit(UD2)
    return program(code, ["atof", "exit"], b"20.5\0")


def sorting():
    """qsort({5, 3, 9, 1, 7}) with a guest comparator, then exit(10 * a[0] + a[4]): the library calls
    back into guest code."""
    code = Code()
    code.emit(SUB_RSP_8)
    code.rip(LEA_RDI_RIP, data(0))
    code.immediate(MOV_ESI, 5)
    code.immediate(MOV_EDX, 4)
    code.rip_label(LEA_RCX_RIP, "compare")
    code.rip(CALL_RIP, slot(0))
    code.rip(MOV_EDI_RIP, data(0))
    code.emit(IMUL_EDI_EDI_10)
    code.rip(ADD_EDI_RIP, data(16))
    code.rip(CALL_RIP, slot(1))
    code.emit(UD2)
    code.label("compare")
    code.emit(MOV_EAX_FROM_RDI, SUB_EAX_FROM_RSI, RET)
    return program(code, ["qsort", "exit"], struct.pack("<5i", 5, 3, 9, 1, 7))


def variadic():
    """snprintf with integer, string, double and stack arguments, then puts, then printf."""
    contents = bytearray(0xc0)
    contents[0x00:0x15] = b"%d %s %.2f %d %d %d\0"
    contents[0x40:0x42] = b"x\0"
    contents[0x48:0x50] = struct.pack("<d", 2.5)
    contents[0x50:0x57] = b"%s %d\n\0"
    contents[0x60:0x68] = b"printed\0"
    buffer = data(0x80)

    code = Code()
    code.emit(SUB_RSP_8, PUSH_6, PUSH_5)
    code.rip(LEA_RDI_RIP, buffer)
    code.immediate(MOV_ESI, 64)
    code.rip(LEA_RDX_RIP, data(0x00))
    code.immediate(MOV_ECX, 42)
    code.rip(LEA_R8_RIP, data(0x40))
    code.immediate(MOV_R9D, 4)
    code.rip(MOVSD_XMM0_RIP, data(0x48))
    code.immediate(MOV_EAX, 1)
    code.rip(CALL_RIP, slot(0))
    code.emit(ADD_RSP_16)
    code.rip(LEA_RDI_RIP, buffer)
    code.rip(CALL_RIP, slot(1))
    code.rip(LEA_RDI_RIP, data(0x50))
    code.rip(LEA_RSI_RIP, data(0x60))
    code.immediate(MOV_EDX, 7)
    code.emit(XOR_EAX_EAX)
    code.rip(CALL_RIP, slot(2))
    code.emit(XOR_EDI_EDI)
    code.rip(CALL_RIP, slot(3))
    code.emit(UD2)
    return program(code, ["snprintf", "puts", "printf", "exit"], bytes(contents))


def opening():
    """_open("created.txt", O_WRONLY | O_CREAT | O_TRUNC, 0640), then exit(fd < 0): the mode is an
    integer variadic argument."""
    code = Code()
    code.emit(SUB_RSP_8)
    code.rip(LEA_RDI_RIP, data(0))
    code.immediate(MOV_ESI, 0x601)
    code.immediate(MOV_EDX, 0o640)
    code.emit(XOR_EAX_EAX)
    code.rip(CALL_RIP, slot(0))
    code.emit(MOV_EDI_EAX, SHR_EDI_31)
    code.rip(CALL_RIP, slot(1))
    code.emit(UD2)
    return program(code, ["_open", "exit"], b"created.txt\0")


def threading():
    """pthread_create with a guest entry that checks its thread pointer and returns its argument plus
    22, then pthread_join and exit with what the thread returned."""
    thread, result = data(0), data(8)
    code = Code()
    code.emit(SUB_RSP_8)
    code.rip(LEA_RDI_RIP, thread)
    code.emit(XOR_ESI_ESI)
    code.rip_label(LEA_RDX_RIP, "entry")
    code.immediate(MOV_ECX, 20)
    code.rip(CALL_RIP, slot(0))
    code.emit(TEST_EAX_EAX)
    code.jump(JNZ, "fail")
    code.rip(MOV_RDI_RIP, thread)
    code.rip(LEA_RSI_RIP, result)
    code.rip(CALL_RIP, slot(1))
    code.emit(TEST_EAX_EAX)
    code.jump(JNZ, "fail")
    code.rip(MOV_EDI_RIP, result)
    code.rip(CALL_RIP, slot(2))
    code.label("fail")
    code.immediate(MOV_EDI, 1)
    code.rip(CALL_RIP, slot(2))
    code.emit(UD2)
    code.label("entry")
    code.emit(MOV_RAX_FROM_FS_0, TEST_RAX_RAX)
    code.jump(JZ, "bad")
    code.emit(CMP_RAX_WITH_RAX_POINTEE)
    code.jump(JNZ, "bad")
    code.emit(LEA_RAX_RDI_PLUS_22, RET)
    code.label("bad")
    code.emit(XOR_EAX_EAX, RET)
    return program(code, ["pthread_create", "pthread_join", "exit"], bytes(16))


def extended():
    """strtold("1.0000000000000000001"), stored from st(0), then exit(42) if it holds the x87 value
    1 + 2^-63: the long double result reaches the guest's x87 stack."""
    contents = bytearray(0x30)
    contents[0:22] = b"1.0000000000000000001\0"
    value = data(0x20)

    code = Code()
    code.emit(SUB_RSP_8)
    code.rip(LEA_RDI_RIP, data(0))
    code.emit(XOR_ESI_ESI)
    code.rip(CALL_RIP, slot(0))
    code.rip(FSTP_TWORD_RIP, value)
    code.rip(MOV_RAX_RIP, value)
    code.immediate64(MOV_RCX_64, 0x8000000000000001)
    code.emit(CMP_RAX_RCX)
    code.jump(JNZ, "fail")
    code.rip(MOVZX_EAX_WORD_RIP, value + 8)
    code.immediate(CMP_EAX, 0x3fff)
    code.jump(JNZ, "fail")
    code.immediate(MOV_EDI, 42)
    code.rip(CALL_RIP, slot(1))
    code.label("fail")
    code.immediate(MOV_EDI, 1)
    code.rip(CALL_RIP, slot(1))
    code.emit(UD2)
    return program(code, ["strtold", "exit"], bytes(contents))


FIXTURES = {"hello": hello, "floating": floating, "sorting": sorting, "variadic": variadic, "opening": opening,
            "threading": threading, "extended": extended}

if __name__ == "__main__":
    open(sys.argv[2], "wb").write(FIXTURES[sys.argv[1]]())
