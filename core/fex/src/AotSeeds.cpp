#include "AotSeeds.hpp"

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_map>

namespace Aps5Fex {

namespace {

struct ProgramHeader {
    std::uint32_t type;
    std::uint32_t flags;
    std::uint64_t offset;
    std::uint64_t address;
    std::uint64_t physicalAddress;
    std::uint64_t fileSize;
    std::uint64_t memorySize;
    std::uint64_t alignment;
};

struct DynamicEntry {
    std::int64_t tag;
    std::uint64_t value;
};

struct Relocation {
    std::uint64_t offset;
    std::uint64_t info;
    std::int64_t addend;
};

constexpr std::uint32_t PT_LOAD = 1;
constexpr std::uint32_t PT_DYNAMIC = 2;
constexpr std::uint32_t PT_GNU_EH_FRAME = 0x6474e550;
constexpr std::uint32_t PF_X = 1;

constexpr std::int64_t DT_NULL = 0, DT_PLTRELSZ = 2, DT_RELA = 7, DT_RELASZ = 8, DT_INIT = 12, DT_FINI = 13, DT_JMPREL = 23,
                       DT_INIT_ARRAY = 25, DT_FINI_ARRAY = 26, DT_INIT_ARRAYSZ = 27, DT_FINI_ARRAYSZ = 28, DT_PREINIT_ARRAY = 32,
                       DT_PREINIT_ARRAYSZ = 33;

constexpr std::uint8_t DW_EH_PE_omit = 0xff;

class Memory {
public:
    explicit Memory(std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges) : ranges(std::move(ranges)) {}

    bool Readable(std::uint64_t address, std::uint64_t length) const {
        for (const auto& [begin, end] : ranges)
            if (address >= begin && address <= end && length <= end - address) return true;
        return false;
    }

    template <typename T>
    std::optional<T> Read(std::uint64_t address) const {
        if (!Readable(address, sizeof(T))) return std::nullopt;
        T value;
        std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(T));
        return value;
    }

private:
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
};

// Reads DWARF EH data sequentially, giving up (ok == false) at the first byte outside the image.
class Cursor {
public:
    Cursor(const Memory& memory, std::uint64_t address) : memory(memory), address(address) {}

    std::uint64_t Address() const { return address; }
    bool Ok() const { return ok; }

    template <typename T>
    T Fixed() {
        const auto value = memory.Read<T>(address);
        if (!value) {
            ok = false;
            return T {};
        }
        address += sizeof(T);
        return *value;
    }

    std::uint64_t Uleb() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; ok && shift < 64; shift += 7) {
            const auto byte = Fixed<std::uint8_t>();
            value |= std::uint64_t(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) return value;
        }
        ok = false;
        return 0;
    }

    std::int64_t Sleb() {
        std::int64_t value = 0;
        unsigned shift = 0;
        std::uint8_t byte = 0;
        do {
            byte = Fixed<std::uint8_t>();
            value |= std::int64_t(byte & 0x7f) << shift;
            shift += 7;
        } while (ok && (byte & 0x80) != 0 && shift < 64);
        if (ok && shift < 64 && (byte & 0x40) != 0) value |= -(std::int64_t {1} << shift);
        return value;
    }

    // A DW_EH_PE-encoded pointer. dataBase is the base for DW_EH_PE_datarel (.eh_frame_hdr's start).
    std::optional<std::uint64_t> Pointer(std::uint8_t encoding, std::uint64_t dataBase = 0) {
        if (encoding == DW_EH_PE_omit) return std::nullopt;
        const std::uint64_t field = address;
        std::uint64_t value = 0;
        switch (encoding & 0x0f) {
        case 0x00: value = Fixed<std::uint64_t>(); break;
        case 0x01: value = Uleb(); break;
        case 0x02: value = Fixed<std::uint16_t>(); break;
        case 0x03: value = Fixed<std::uint32_t>(); break;
        case 0x04: value = Fixed<std::uint64_t>(); break;
        case 0x09: value = static_cast<std::uint64_t>(Sleb()); break;
        case 0x0a: value = static_cast<std::uint64_t>(static_cast<std::int64_t>(Fixed<std::int16_t>())); break;
        case 0x0b: value = static_cast<std::uint64_t>(static_cast<std::int64_t>(Fixed<std::int32_t>())); break;
        case 0x0c: value = Fixed<std::uint64_t>(); break;
        default: ok = false; return std::nullopt;
        }
        switch (encoding & 0x70) {
        case 0x00: break;
        case 0x10: value += field; break;
        case 0x30: value += dataBase; break;
        default: ok = false; return std::nullopt;
        }
        if ((encoding & 0x80) != 0) {
            const auto indirect = memory.Read<std::uint64_t>(value);
            if (!indirect) {
                ok = false;
                return std::nullopt;
            }
            value = *indirect;
        }
        if (!ok) return std::nullopt;
        return value;
    }

    // The size part of an FDE's address range: the format of the encoding, never its application.
    std::optional<std::uint64_t> Size(std::uint8_t encoding) { return Pointer(encoding & 0x0f); }

    void Skip(std::uint64_t bytes) {
        if (!memory.Readable(address, bytes)) ok = false;
        else address += bytes;
    }

private:
    const Memory& memory;
    std::uint64_t address;
    bool ok = true;
};

class EhFrame {
public:
    explicit EhFrame(const Memory& memory) : memory(memory) {}

    // Parses the record at address: an FDE yields [begin, end) of its code. next is the following record (0 at
    // the terminator or on malformed data).
    std::optional<std::pair<std::uint64_t, std::uint64_t>> Record(std::uint64_t address, std::uint64_t& next) {
        next = 0;
        Cursor cursor(memory, address);
        std::uint64_t length = cursor.Fixed<std::uint32_t>();
        if (!cursor.Ok() || length == 0) return std::nullopt;
        if (length == 0xffffffff) length = cursor.Fixed<std::uint64_t>();
        const std::uint64_t body = cursor.Address();
        if (!cursor.Ok() || !memory.Readable(body, length)) return std::nullopt;
        next = body + length;
        const std::uint64_t idField = cursor.Address();
        const std::uint32_t id = cursor.Fixed<std::uint32_t>();
        if (!cursor.Ok() || id == 0) return std::nullopt; // a CIE
        const auto encoding = CieEncoding(idField - id);
        if (!encoding) return std::nullopt;
        const auto begin = cursor.Pointer(*encoding);
        const auto size = cursor.Size(*encoding);
        if (!begin || !size || *begin == 0) return std::nullopt;
        return std::pair {*begin, *begin + *size};
    }

private:
    // The FDE pointer encoding from the CIE's augmentation ('R'); DW_EH_PE_absptr without one.
    std::optional<std::uint8_t> CieEncoding(std::uint64_t cie) {
        if (const auto found = cies.find(cie); found != cies.end()) return found->second;
        Cursor cursor(memory, cie);
        std::uint64_t length = cursor.Fixed<std::uint32_t>();
        if (length == 0xffffffff) length = cursor.Fixed<std::uint64_t>();
        if (cursor.Fixed<std::uint32_t>() != 0 || !cursor.Ok()) return std::nullopt;
        const auto version = cursor.Fixed<std::uint8_t>();
        std::string augmentation;
        for (char c = static_cast<char>(cursor.Fixed<std::uint8_t>()); cursor.Ok() && c != 0; c = static_cast<char>(cursor.Fixed<std::uint8_t>()))
            augmentation.push_back(c);
        cursor.Uleb();                                   // code alignment
        cursor.Sleb();                                   // data alignment
        if (version == 1) cursor.Fixed<std::uint8_t>();  // return address register
        else cursor.Uleb();
        std::uint8_t encoding = 0;
        if (!augmentation.empty() && augmentation[0] == 'z') {
            cursor.Uleb(); // augmentation data length
            for (std::size_t i = 1; i < augmentation.size() && cursor.Ok(); ++i) {
                switch (augmentation[i]) {
                case 'R': encoding = cursor.Fixed<std::uint8_t>(); break;
                case 'L': cursor.Fixed<std::uint8_t>(); break;
                case 'P': cursor.Pointer(cursor.Fixed<std::uint8_t>()); break;
                case 'S':
                case 'B': break;
                default: return std::nullopt;
                }
            }
        }
        if (!cursor.Ok()) return std::nullopt;
        cies.emplace(cie, encoding);
        return encoding;
    }

    const Memory& memory;
    std::unordered_map<std::uint64_t, std::uint8_t> cies;
};

}

AotSeeds FindAotSeeds(const AotSeedImage& image) {
    AotSeeds seeds;
    std::vector<ProgramHeader> headers(image.programHeaderCount);
    if (!headers.empty()) std::memcpy(headers.data(), image.programHeaders, headers.size() * sizeof(ProgramHeader));

    std::vector<std::pair<std::uint64_t, std::uint64_t>> loaded;
    std::uint64_t dynamic = 0, ehFrameHdr = 0;
    for (const auto& header : headers) {
        if (header.type == PT_LOAD && header.memorySize != 0) {
            const std::uint64_t begin = image.base + header.address;
            loaded.emplace_back(begin, begin + header.memorySize);
            if ((header.flags & PF_X) != 0) seeds.executable.emplace_back(begin, begin + header.memorySize);
        } else if (header.type == PT_DYNAMIC) {
            dynamic = image.base + header.address;
        } else if (header.type == PT_GNU_EH_FRAME) {
            ehFrameHdr = image.base + header.address;
        }
    }
    std::sort(seeds.executable.begin(), seeds.executable.end());
    const Memory memory(loaded);
    const auto executable = [&](std::uint64_t address) {
        return std::any_of(seeds.executable.begin(), seeds.executable.end(),
                           [address](const auto& range) { return address >= range.first && address < range.second; });
    };
    const auto add = [&](std::uint64_t address, std::size_t& counter) {
        if (!executable(address)) return;
        seeds.addresses.push_back(address);
        ++counter;
    };

    if (image.entry != 0) add(image.entry, seeds.entryPoints);
    for (const std::uint64_t address : image.exports) add(address, seeds.exports);

    if (dynamic != 0) {
        std::uint64_t init = 0, fini = 0, rela = 0, relaSize = 0, plt = 0, pltSize = 0;
        std::uint64_t arrays[3][2] = {};
        for (std::uint64_t at = dynamic;; at += sizeof(DynamicEntry)) {
            const auto entry = memory.Read<DynamicEntry>(at);
            if (!entry || entry->tag == DT_NULL) break;
            switch (entry->tag) {
            case DT_INIT: init = image.base + entry->value; break;
            case DT_FINI: fini = image.base + entry->value; break;
            case DT_RELA: rela = image.base + entry->value; break;
            case DT_RELASZ: relaSize = entry->value; break;
            case DT_JMPREL: plt = image.base + entry->value; break;
            case DT_PLTRELSZ: pltSize = entry->value; break;
            case DT_INIT_ARRAY: arrays[0][0] = image.base + entry->value; break;
            case DT_INIT_ARRAYSZ: arrays[0][1] = entry->value; break;
            case DT_FINI_ARRAY: arrays[1][0] = image.base + entry->value; break;
            case DT_FINI_ARRAYSZ: arrays[1][1] = entry->value; break;
            case DT_PREINIT_ARRAY: arrays[2][0] = image.base + entry->value; break;
            case DT_PREINIT_ARRAYSZ: arrays[2][1] = entry->value; break;
            default: break;
            }
        }
        if (init != 0) add(init, seeds.initFini);
        if (fini != 0) add(fini, seeds.initFini);
        for (const auto& [array, size] : arrays)
            for (std::uint64_t i = 0; array != 0 && i < size / sizeof(std::uint64_t); ++i)
                if (const auto function = memory.Read<std::uint64_t>(array + i * sizeof(std::uint64_t))) add(*function, seeds.initFini);
        // Pointers the relocations stored that land in code: function pointers, vtables' entries, jump tables.
        for (const auto& [table, size] : {std::pair {rela, relaSize}, std::pair {plt, pltSize}})
            for (std::uint64_t i = 0; table != 0 && i < size / sizeof(Relocation); ++i)
                if (const auto relocation = memory.Read<Relocation>(table + i * sizeof(Relocation)))
                    if (const auto value = memory.Read<std::uint64_t>(image.base + relocation->offset)) add(*value, seeds.relocations);
    }

    if (ehFrameHdr != 0) {
        EhFrame frames(memory);
        Cursor header(memory, ehFrameHdr);
        const auto version = header.Fixed<std::uint8_t>();
        const auto frameEncoding = header.Fixed<std::uint8_t>();
        const auto countEncoding = header.Fixed<std::uint8_t>();
        const auto tableEncoding = header.Fixed<std::uint8_t>();
        const auto ehFrame = header.Pointer(frameEncoding, ehFrameHdr);
        const auto count = header.Pointer(countEncoding, ehFrameHdr);
        const auto addFde = [&](std::uint64_t record, std::uint64_t& next) {
            if (const auto range = frames.Record(record, next)) {
                if (!executable(range->first)) return;
                add(range->first, seeds.fdes);
                seeds.fdeRanges.emplace_back(range->first, range->second);
            }
        };
        if (header.Ok() && version == 1 && count && *count != 0 && tableEncoding != DW_EH_PE_omit) {
            seeds.fdeTableUsed = true;
            for (std::uint64_t i = 0; i < *count && header.Ok(); ++i) {
                header.Pointer(tableEncoding, ehFrameHdr); // initial location: the FDE says it again
                const auto fde = header.Pointer(tableEncoding, ehFrameHdr);
                std::uint64_t next = 0;
                if (fde) addFde(*fde, next);
            }
        } else if (header.Ok() && version == 1 && ehFrame) {
            std::uint64_t next = *ehFrame;
            for (std::size_t records = 0; next != 0 && records < (std::size_t {1} << 24); ++records) {
                const std::uint64_t record = next;
                addFde(record, next);
            }
        }
    }

    std::sort(seeds.addresses.begin(), seeds.addresses.end());
    seeds.addresses.erase(std::unique(seeds.addresses.begin(), seeds.addresses.end()), seeds.addresses.end());
    for (auto& [begin, end] : seeds.fdeRanges) {
        for (const auto& [low, high] : seeds.executable)
            if (begin >= low && begin < high) end = std::min(end, high);
    }
    std::sort(seeds.fdeRanges.begin(), seeds.fdeRanges.end());
    return seeds;
}

}
