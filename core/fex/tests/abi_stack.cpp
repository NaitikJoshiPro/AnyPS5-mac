// Stack arguments: Apple's arm64 convention packs them at their natural size where SysV uses 8-byte
// slots. The test library echoes 13 narrow arguments; the libSceAgc exports with more than eight
// arguments write packets whose fields come from the stack.
#include "abi.h"

struct CommandBuffer {
    u32* bottom;
    u32* top;
    u32* cursorUp;
    u32* cursorDown;
    void* callback;
    void* userData;
    u32 reservedDwords;
};

extern "C" {
u32* sceAgcCbReleaseMem(CommandBuffer* buffer, u8 action, u16 gcrControl, u8 destination, u8 cachePolicy, const volatile u64* address,
                        u8 dataSelect, u64 data, u16 gdsOffset, u16 gdsSize, u8 interrupt, u32 interruptContextId);
u32* sceAgcDcbDmaData(CommandBuffer* buffer, u8 a1, u8 a2, u8 a3, u64 a4, u8 a5, u8 a6, u64 a7, u32 a8, u8 a9, u8 a10, u8 a11);
}

static void Echo() {
    u64 out[13] = {};
    int marker = 0;
    const u32 count = Aps5AbiStack(out, 0xa1, 0xb2c3, 0xd4, 0xe5, &marker, 0xf6, 0x0102030405060708ULL, 0x1718, 0x292a, 0x3b, 0x4c5d6e7f,
                                   -5, -1234);
    Check(count == 13, 0);
    Check(out[0] == 0xa1 && out[1] == 0xb2c3 && out[2] == 0xd4 && out[3] == 0xe5, 1);
    Check(out[4] == reinterpret_cast<u64>(&marker) && out[5] == 0xf6 && out[6] == 0x0102030405060708ULL, 2);
    Check(out[7] == 0x1718 && out[8] == 0x292a && out[9] == 0x3b && out[10] == 0x4c5d6e7f, 3);
    Check(out[11] == static_cast<u64>(-5LL) && out[12] == static_cast<u64>(-1234LL), 4);
}

static void ReleaseMem() {
    static u32 storage[64];
    static volatile u64 label __attribute__((aligned(8)));
    CommandBuffer buffer {storage, storage + 64, storage, nullptr, nullptr, nullptr, 0};
    // GDS release: data select 5 takes the value from gdsOffset (9th) and gdsSize (10th), then the
    // interrupt selector (11th) and context id (12th): all four on the stack.
    u32* packet = sceAgcCbReleaseMem(&buffer, 0x14, 0x123, 1, 2, &label, 5, 0, 0x1234, 0x5678, 3, 0x0abcdef);
    Check(packet == storage && buffer.cursorUp == storage + 8, 5);
    Check(packet[0] == (0xc0000000u | (6u << 16) | (0x49u << 8)), 6);
    Check(packet[1] == (0x14u | (5u << 8) | (0x323u << 12) | (2u << 25)), 7);
    Check(packet[2] == ((1u << 16) | (3u << 24) | (5u << 29)), 8);
    Check(packet[3] == static_cast<u32>(reinterpret_cast<u64>(&label)) && packet[4] == static_cast<u32>(reinterpret_cast<u64>(&label) >> 32), 9);
    Check(packet[5] == (0x1234u | (0x5678u << 16)) && packet[6] == 0, 10);
    Check(packet[7] == 0x0abcdef, 11);
}

ABI_MAIN(Echo(); ReleaseMem())
