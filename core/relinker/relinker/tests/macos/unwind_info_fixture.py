"""A PS5-style ELF with PT_GNU_EH_FRAME and one CIE that asks for its own unwind information and exits with eh_frame_size.

fixture() asks sceKernelGetModuleInfoForUnwind; module_info_fixture() asks sceKernelGetModuleInfoFromAddr with
flags 2 and a stale st_size, as a title's own unwinder does; that size counts the terminator."""
import struct, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from nid import nid

DATA = 0x8000
HEADER = 0x4800
FRAMES = 0x4820
INFO = DATA + 0x400
SIZE_FIELD = INFO + 8 + 256 + 16
MODULE_INFO_SIZE_FIELD = INFO + 0x15C
EXPECTED = 16
MODULE_INFO_EXPECTED = EXPECTED + 4

def fixture(function="sceKernelGetModuleInfoForUnwind", flags=0, size_field=SIZE_FIELD, stale_size=0):
    image = bytearray(DATA + 0x1000)
    image[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<HHIQQQIHHHHHH", image, 16, 3, 62, 1, 0x4000, 64, 0, 0, 64, 56, 4, 64, 0, 0)
    struct.pack_into("<IIQQQQQQ", image, 64, 1, 5, 0x4000, 0x4000, 0x4000, 0x1000, 0x1000, 0x1000)
    struct.pack_into("<IIQQQQQQ", image, 120, 1, 6, DATA, DATA, DATA, 0x1000, 0x1000, 0x1000)
    struct.pack_into("<IIQQQQQQ", image, 232, 0x6474E550, 4, HEADER, HEADER, HEADER, 12, 12, 4)
    code = bytearray(b"\x48\x83\xec\x08")
    code += b"\x48\x8d\x3d" + struct.pack("<i", 0x4000 - (0x4000 + len(code) + 7))
    code += b"\xbe" + struct.pack("<I", flags)
    code += b"\x48\x8d\x15" + struct.pack("<i", INFO - (0x4000 + len(code) + 7))
    code += b"\xff\x15" + struct.pack("<i", DATA - (0x4000 + len(code) + 6))
    code += b"\x8b\x3d" + struct.pack("<i", size_field - (0x4000 + len(code) + 6))
    code += b"\xff\x15" + struct.pack("<i", DATA + 8 - (0x4000 + len(code) + 6))
    code += b"\x0f\x0b"
    image[0x4000:0x4000 + len(code)] = code
    image[HEADER:HEADER + 12] = bytes([1, 0x1B, 0x03, 0x3B]) + struct.pack("<i", FRAMES - (HEADER + 4)) + struct.pack("<I", 0)
    cie = struct.pack("<I", 12) + struct.pack("<I", 0) + bytes([1, 0, 1, 0x78, 16, 0x0C, 7, 8])
    assert len(cie) == EXPECTED
    image[FRAMES:FRAMES + len(cie)] = cie
    struct.pack_into("<Q", image, INFO, stale_size)
    strings = b"\0libc.prx\0" + (nid(function) + "#A#B").encode() + b"\0" + (nid("exit") + "#A#B").encode() + b"\0"
    image[DATA + 0xA00:DATA + 0xA00 + len(strings)] = strings
    first = strings.index(b"\0", 1) + 1
    second = strings.index(b"\0", first) + 1
    struct.pack_into("<IBBHQQ", image, DATA + 0xB00 + 24, first, 0x12, 0, 0, 0, 0)
    struct.pack_into("<IBBHQQ", image, DATA + 0xB00 + 48, second, 0x12, 0, 0, 0, 0)
    struct.pack_into("<QQq", image, DATA + 0xC00, DATA, (1 << 32) | 6, 0)
    struct.pack_into("<QQq", image, DATA + 0xC18, DATA + 8, (2 << 32) | 6, 0)
    tags = [(1, 1), (5, DATA + 0xA00), (10, len(strings)), (6, DATA + 0xB00), (11, 24), (0x6100003f, 72), (7, DATA + 0xC00), (8, 48), (9, 24), (0, 0)]
    for index, tag in enumerate(tags):
        struct.pack_into("<qQ", image, DATA + 0x800 + index * 16, *tag)
    struct.pack_into("<IIQQQQQQ", image, 176, 2, 6, DATA + 0x800, DATA + 0x800, DATA + 0x800, len(tags) * 16, len(tags) * 16, 8)
    return image

def module_info_fixture():
    return fixture("sceKernelGetModuleInfoFromAddr", 2, MODULE_INFO_SIZE_FIELD, 8)

if __name__ == "__main__":
    open(sys.argv[1], "wb").write(fixture())
