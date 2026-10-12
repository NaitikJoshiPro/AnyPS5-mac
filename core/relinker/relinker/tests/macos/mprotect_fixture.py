"""A PS5-style ELF whose two segments are 12 KiB apart and that protects the page at the start of the second.

The guest rounds the range to its 16 KiB pages. When the image is loaded at a multiple of 4 KiB only,
the rounded range starts in the gap, which no host page backs."""
import struct, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from nid import nid

DATA = 0x8000

def fixture():
    image = bytearray(0x9000)
    image[:16] = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<HHIQQQIHHHHHH", image, 16, 3, 62, 1, 0x4000, 64, 0, 0, 64, 56, 3, 64, 0, 0)
    struct.pack_into("<IIQQQQQQ", image, 64, 1, 5, 0x4000, 0x4000, 0x4000, 0x1000, 0x1000, 0x1000)
    struct.pack_into("<IIQQQQQQ", image, 120, 1, 6, DATA, DATA, DATA, 0x1000, 0x1000, 0x1000)
    code = bytearray(b"\x48\x83\xec\x08")
    code += b"\x48\x8d\x3d" + struct.pack("<i", DATA - (0x4004 + 7))
    code += b"\xbe" + struct.pack("<I", 0x1000)
    code += b"\xba" + struct.pack("<I", 3)
    code += b"\xff\x15" + struct.pack("<i", DATA - (0x4000 + len(code) + 6))
    code += b"\x89\xc7"
    code += b"\xff\x15" + struct.pack("<i", DATA + 8 - (0x4000 + len(code) + 6))
    code += b"\x0f\x0b"
    image[0x4000:0x4000 + len(code)] = code
    strings = b"\0libc.prx\0" + (nid("sceKernelMprotect") + "#A#B").encode() + b"\0" + (nid("exit") + "#A#B").encode() + b"\0"
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

if __name__ == "__main__":
    open(sys.argv[1], "wb").write(fixture())
