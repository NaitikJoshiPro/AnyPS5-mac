#ifndef CORE_FEX_SRC_AOTSEEDS_HPP
#define CORE_FEX_SRC_AOTSEEDS_HPP

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace Aps5Fex {

// Guest addresses where code starts, found statically in a loaded (and relocated) x86-64 ELF image, for compiling
// ahead of time: the entry point, exports, DT_INIT/DT_FINI and the init/fini arrays, targets of relocations that
// point into executable segments, and the starts of the .eh_frame FDEs (through PT_GNU_EH_FRAME; .eh_frame is walked
// directly when the .eh_frame_hdr search table is empty or missing).
struct AotSeedImage {
    std::uint64_t base = 0;                 // load bias: address = base + p_vaddr
    const void* programHeaders = nullptr;   // Elf64_Phdr[programHeaderCount], as in the file
    std::size_t programHeaderCount = 0;
    std::uint64_t entry = 0;                // absolute; 0 when none
    std::vector<std::uint64_t> exports;     // absolute addresses of exported symbols
};

struct AotSeeds {
    std::vector<std::uint64_t> addresses;                          // sorted, unique, all inside executable ranges
    std::vector<std::pair<std::uint64_t, std::uint64_t>> executable; // [begin, end) of the PF_X segments
    std::vector<std::pair<std::uint64_t, std::uint64_t>> fdeRanges;  // [begin, end) of each FDE's code, clipped
    std::size_t entryPoints = 0, exports = 0, initFini = 0, relocations = 0, fdes = 0;
    bool fdeTableUsed = false; // .eh_frame_hdr's binary-search table was present and used
};

AotSeeds FindAotSeeds(const AotSeedImage& image);

}

#endif
