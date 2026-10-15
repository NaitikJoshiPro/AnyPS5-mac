extern "C" {
unsigned char guestFlexibleMemoryModuleData[512u * 1024u * 1024u] = {};
}

#ifdef __APPLE__
#include <cstdint>

// On macOS dl_iterate_phdr (prx/libc/src/specifics/macos/GuestImages.cpp) lists only the dyld images that
// carry the relinker's __APS5DATA,__meta program headers, so a plain host dylib is no guest module there. The
// fixture writes the metadata a relinked module has (core/relinker/elfpatcher/src/macos/MacOsImage.cpp:
// version 2, load bias at 0x08, header count at 0x58, the ELF program headers from 0x60), with one
// read/write PT_LOAD over its data, so the budget counts it as the relinked module would be counted.
namespace {

struct ProgramHeader {
    std::uint32_t p_type;
    std::uint32_t p_flags;
    std::uint64_t p_offset;
    std::uint64_t p_vaddr;
    std::uint64_t p_paddr;
    std::uint64_t p_filesz;
    std::uint64_t p_memsz;
    std::uint64_t p_align;
};
static_assert(sizeof(ProgramHeader) == 56);

struct Meta {
    std::uint64_t version;
    std::uint64_t loadBias;
    unsigned char unused[0x58 - 0x10];
    std::uint64_t headerCount;
    ProgramHeader headers[1];
};
static_assert(__builtin_offsetof(Meta, headerCount) == 0x58 && __builtin_offsetof(Meta, headers) == 0x60);

__attribute__((used, section("__APS5DATA,__meta"))) Meta meta{};

__attribute__((constructor)) void DescribeData() {
    constexpr std::uint32_t loadSegment = 1;
    constexpr std::uint32_t readWrite = 4 | 2;
    meta.loadBias = 0;
    meta.headerCount = 1;
    meta.headers[0] = {loadSegment, readWrite, 0x1000, reinterpret_cast<std::uintptr_t>(guestFlexibleMemoryModuleData), 0, 0,
                       sizeof(guestFlexibleMemoryModuleData), 0x4000};
    meta.version = 2;
}

}
#endif
