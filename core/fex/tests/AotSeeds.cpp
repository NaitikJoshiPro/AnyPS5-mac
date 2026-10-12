// FindAotSeeds on a synthetic loaded image: entry, exports, DT_INIT/DT_FINI and their arrays, relocation targets,
// and .eh_frame FDEs, both through the .eh_frame_hdr table and by walking .eh_frame when the table is empty. Seeds
// outside the executable segment are dropped and FDE ranges are clipped to it.
#include "AotSeeds.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <source_location>
#include <vector>

namespace {

using namespace Aps5Fex;

void Require(bool value, std::source_location where = std::source_location::current()) {
    if (!value) {
        std::fprintf(stderr, "AOT seed check failed at line %u\n", static_cast<unsigned>(where.line()));
        std::abort();
    }
}

struct ProgramHeader {
    std::uint32_t type, flags;
    std::uint64_t offset, address, physicalAddress, fileSize, memorySize, alignment;
};

constexpr std::uint32_t PT_LOAD = 1, PT_DYNAMIC = 2, PT_GNU_EH_FRAME = 0x6474e550;
constexpr std::uint64_t Text = 0x1000, TextEnd = 0x2000, Data = 0x2000, DataEnd = 0x3000;
constexpr std::uint64_t EhFrameHdr = 0x2100, EhFrame = 0x2200, InitArray = 0x2400, FiniArray = 0x2410, Rela = 0x2500,
                        Slots = 0x2600, Dynamic = 0x2800, DataObject = 0x2700;

class Image {
public:
    Image() : memory(DataEnd, 0) {}

    std::uint64_t Base() const { return reinterpret_cast<std::uint64_t>(memory.data()); }

    template <typename T>
    void Put(std::uint64_t offset, T value) {
        std::memcpy(memory.data() + offset, &value, sizeof(value));
    }

    // CIE ("zR", FDE pointers pcrel|sdata4) then two FDEs and a terminator; returns the FDEs' offsets.
    std::pair<std::uint64_t, std::uint64_t> WriteEhFrame() {
        std::uint64_t at = EhFrame;
        const std::uint8_t cie[] = {0, 0, 0, 0, 1, 'z', 'R', 0, 1, 0x78, 16, 1, 0x1b, 0, 0, 0};
        Put<std::uint32_t>(at, sizeof(cie));
        std::memcpy(memory.data() + at + 4, cie, sizeof(cie));
        const std::uint64_t cieStart = at;
        at += 4 + sizeof(cie);
        const auto fde = [&](std::uint64_t begin, std::int32_t range) {
            const std::uint64_t start = at;
            Put<std::uint32_t>(at, 16);                                                   // length
            Put<std::uint32_t>(at + 4, static_cast<std::uint32_t>(at + 4 - cieStart));    // CIE pointer
            Put<std::int32_t>(at + 8, static_cast<std::int32_t>(begin - (at + 8)));       // pc_begin, pcrel
            Put<std::int32_t>(at + 12, range);                                            // pc_range
            Put<std::uint8_t>(at + 16, 0);                                                // augmentation length
            at += 20;
            return start;
        };
        const std::uint64_t first = fde(0x1100, 0x40);
        const std::uint64_t second = fde(0x1200, 0x1000); // runs past the text: clipped
        Put<std::uint32_t>(at, 0);
        return {first, second};
    }

    void WriteEhFrameHdr(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& table) {
        const std::uint8_t header[] = {1, 0x1b, 0x03, static_cast<std::uint8_t>(table.empty() ? 0xff : 0x3b)};
        std::memcpy(memory.data() + EhFrameHdr, header, sizeof(header));
        Put<std::int32_t>(EhFrameHdr + 4, static_cast<std::int32_t>(EhFrame - (EhFrameHdr + 4)));
        Put<std::uint32_t>(EhFrameHdr + 8, static_cast<std::uint32_t>(table.size()));
        for (std::size_t i = 0; i < table.size(); ++i) {
            Put<std::int32_t>(EhFrameHdr + 12 + i * 8, static_cast<std::int32_t>(table[i].first - EhFrameHdr));
            Put<std::int32_t>(EhFrameHdr + 16 + i * 8, static_cast<std::int32_t>(table[i].second - EhFrameHdr));
        }
    }

    void WriteDynamic() {
        const std::int64_t tags[][2] = {{12, 0x1010},       {13, 0x1020},        {25, InitArray}, {27, 16}, {26, FiniArray}, {28, 8},
                                        {7, Rela},          {8, 2 * 24},         {0, 0}};
        for (std::size_t i = 0; i < std::size(tags); ++i) {
            Put<std::int64_t>(Dynamic + i * 16, tags[i][0]);
            Put<std::int64_t>(Dynamic + i * 16 + 8, tags[i][1]);
        }
        // The arrays and relocated slots hold absolute (relocated) addresses.
        Put<std::uint64_t>(InitArray, Base() + 0x1030);
        Put<std::uint64_t>(InitArray + 8, Base() + 0x1070);
        Put<std::uint64_t>(FiniArray, Base() + 0x1040);
        Put<std::uint64_t>(Rela, Slots);            // R_X86_64_RELATIVE at Slots
        Put<std::uint64_t>(Rela + 8, 8);
        Put<std::int64_t>(Rela + 16, 0x1050);
        Put<std::uint64_t>(Rela + 24, Slots + 8);   // R_X86_64_64 to a data object: not code
        Put<std::uint64_t>(Rela + 32, 1);
        Put<std::int64_t>(Rela + 40, 0);
        Put<std::uint64_t>(Slots, Base() + 0x1050);
        Put<std::uint64_t>(Slots + 8, Base() + DataObject);
    }

    AotSeedImage SeedImage() const {
        return {Base(), headers, std::size(headers), Base() + Text, {Base() + 0x1060, Base() + DataObject}};
    }

private:
    std::vector<std::uint8_t> memory;
    ProgramHeader headers[4] = {
        {PT_LOAD, 5, Text, Text, Text, TextEnd - Text, TextEnd - Text, 0x1000},
        {PT_LOAD, 6, Data, Data, Data, DataEnd - Data, DataEnd - Data, 0x1000},
        {PT_DYNAMIC, 6, Dynamic, Dynamic, Dynamic, 0x100, 0x100, 8},
        {PT_GNU_EH_FRAME, 4, EhFrameHdr, EhFrameHdr, EhFrameHdr, 0x20, 0x20, 4},
    };
};

void Check(const AotSeeds& seeds, std::uint64_t base, bool table) {
    std::vector<std::uint64_t> expected = {0x1000, 0x1010, 0x1020, 0x1030, 0x1040, 0x1050, 0x1060, 0x1070, 0x1100, 0x1200};
    for (auto& address : expected) address += base;
    Require(seeds.addresses == expected);
    Require(seeds.entryPoints == 1 && seeds.exports == 1 && seeds.initFini == 5 && seeds.relocations == 1 && seeds.fdes == 2);
    Require(seeds.fdeTableUsed == table);
    Require(seeds.executable.size() == 1 && seeds.executable[0] == std::pair {base + Text, base + TextEnd});
    Require(seeds.fdeRanges.size() == 2);
    Require(seeds.fdeRanges[0] == std::pair {base + 0x1100, base + 0x1140});
    Require(seeds.fdeRanges[1] == std::pair {base + 0x1200, base + TextEnd});
}

}

int main() {
    {
        Image image;
        image.WriteEhFrame();
        image.WriteEhFrameHdr({}); // an empty search table: .eh_frame is walked
        image.WriteDynamic();
        Check(FindAotSeeds(image.SeedImage()), image.Base(), false);
    }
    {
        Image image;
        const auto [first, second] = image.WriteEhFrame();
        image.WriteEhFrameHdr({{0x1100, first}, {0x1200, second}});
        image.WriteDynamic();
        Check(FindAotSeeds(image.SeedImage()), image.Base(), true);
    }
    {
        // An all-zero .eh_frame (the relinked DOOM's libc): no FDEs, the other seeds still found.
        Image image;
        image.WriteEhFrameHdr({});
        image.WriteDynamic();
        const AotSeeds seeds = FindAotSeeds(image.SeedImage());
        Require(seeds.fdes == 0 && seeds.fdeRanges.empty() && seeds.addresses.size() == 8);
    }
    std::puts("AOT seeds: all checks passed");
    return 0;
}
