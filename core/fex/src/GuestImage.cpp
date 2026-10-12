#include "GuestImage.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <sys/mman.h>

namespace Aps5Fex {

namespace {

struct ElfHeader {
    unsigned char ident[16];
    std::uint16_t type;
    std::uint16_t machine;
    std::uint32_t version;
    std::uint64_t entry;
    std::uint64_t programHeaderOffset;
    std::uint64_t sectionHeaderOffset;
    std::uint32_t flags;
    std::uint16_t headerSize;
    std::uint16_t programHeaderSize;
    std::uint16_t programHeaderCount;
    std::uint16_t sectionHeaderSize;
    std::uint16_t sectionHeaderCount;
    std::uint16_t sectionNameIndex;
};

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

struct Symbol {
    std::uint32_t name;
    unsigned char info;
    unsigned char other;
    std::uint16_t section;
    std::uint64_t value;
    std::uint64_t size;
};

struct Relocation {
    std::uint64_t offset;
    std::uint64_t info;
    std::int64_t addend;
};

constexpr std::uint32_t PT_LOAD = 1;
constexpr std::uint32_t PT_DYNAMIC = 2;
constexpr std::uint32_t PT_TLS = 7;
constexpr std::int64_t DT_NULL = 0;
constexpr std::int64_t DT_NEEDED = 1;
constexpr std::int64_t DT_PLTRELSZ = 2;
constexpr std::int64_t DT_HASH = 4;
constexpr std::int64_t DT_STRTAB = 5;
constexpr std::int64_t DT_SYMTAB = 6;
constexpr std::int64_t DT_RELA = 7;
constexpr std::int64_t DT_RELASZ = 8;
constexpr std::int64_t DT_INIT = 12;
constexpr std::int64_t DT_JMPREL = 23;
constexpr std::int64_t DT_INIT_ARRAY = 25;
constexpr std::int64_t DT_INIT_ARRAYSZ = 27;
constexpr std::uint32_t R_X86_64_NONE = 0;
constexpr std::uint32_t R_X86_64_64 = 1;
constexpr std::uint32_t R_X86_64_GLOB_DAT = 6;
constexpr std::uint32_t R_X86_64_JUMP_SLOT = 7;
constexpr std::uint32_t R_X86_64_RELATIVE = 8;
constexpr std::uint32_t R_X86_64_DTPMOD64 = 16;
constexpr std::uint32_t R_X86_64_DTPOFF64 = 17;
constexpr std::uint32_t R_X86_64_TPOFF64 = 18;
constexpr unsigned char STB_LOCAL = 0;
constexpr unsigned char STB_WEAK = 2;
constexpr unsigned char STT_TLS = 6;
constexpr unsigned char STV_INTERNAL = 1;
constexpr unsigned char STV_HIDDEN = 2;
constexpr std::uint64_t PageSize = 0x4000;

std::uint64_t PageDown(std::uint64_t value) { return value & ~(PageSize - 1); }
std::uint64_t PageUp(std::uint64_t value) { return (value + PageSize - 1) & ~(PageSize - 1); }

}

GuestImage::GuestImage(const std::filesystem::path& path) : path(path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path.string());
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    ElfHeader header;
    if (bytes.size() < sizeof(header)) throw std::runtime_error(path.string() + " is not an ELF file");
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (std::memcmp(header.ident, "\x7f" "ELF\x02\x01", 6) != 0 || header.machine != 62)
        throw std::runtime_error(path.string() + " is not an x86-64 ELF file");
    if (header.programHeaderSize != sizeof(ProgramHeader) ||
        header.programHeaderOffset + std::uint64_t(header.programHeaderCount) * sizeof(ProgramHeader) > bytes.size())
        throw std::runtime_error(path.string() + " has malformed program headers");
    std::vector<ProgramHeader> headers(header.programHeaderCount);
    std::memcpy(headers.data(), bytes.data() + header.programHeaderOffset, headers.size() * sizeof(ProgramHeader));
    static_assert(sizeof(ProgramHeader) == ProgramHeaderSize);
    programHeaders.resize(headers.size() * sizeof(ProgramHeader));
    std::memcpy(programHeaders.data(), headers.data(), programHeaders.size());

    std::uint64_t low = UINT64_MAX;
    std::uint64_t high = 0;
    for (const auto& segment : headers) {
        if (segment.type != PT_LOAD || segment.memorySize == 0) continue;
        if (segment.offset + segment.fileSize > bytes.size() || segment.fileSize > segment.memorySize)
            throw std::runtime_error(path.string() + " has a segment outside the file");
        low = std::min(low, PageDown(segment.address));
        high = std::max(high, PageUp(segment.address + segment.memorySize));
    }
    if (low >= high) throw std::runtime_error(path.string() + " has no loadable segments");

    void* reservation = mmap(nullptr, high - low, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (reservation == MAP_FAILED) throw std::runtime_error("cannot reserve address space for " + path.string());
    start = reinterpret_cast<std::uint64_t>(reservation);
    size = high - low;
    base = start - low;
    for (const auto& segment : headers) {
        if (segment.type == PT_LOAD && segment.memorySize != 0) {
            const std::uint64_t first = PageDown(segment.address);
            const std::uint64_t last = PageUp(segment.address + segment.memorySize);
            if (mprotect(reinterpret_cast<void*>(base + first), last - first, PROT_READ | PROT_WRITE) != 0)
                throw std::runtime_error("cannot map a segment of " + path.string());
            std::memcpy(reinterpret_cast<void*>(base + segment.address), bytes.data() + segment.offset, segment.fileSize);
        } else if (segment.type == PT_DYNAMIC) {
            dynamic = segment.address;
        } else if (segment.type == PT_TLS) {
            tls = {segment.address, static_cast<std::size_t>(segment.fileSize), static_cast<std::size_t>(segment.memorySize),
                   static_cast<std::size_t>(std::max<std::uint64_t>(segment.alignment, 1))};
        }
    }
    entry = base + header.entry;
    if (dynamic == 0) throw std::runtime_error(path.string() + " has no dynamic segment");

    std::uint64_t strings = 0, symbols = 0, hash = 0;
    for (auto* tag = reinterpret_cast<const DynamicEntry*>(base + dynamic); tag->tag != DT_NULL; ++tag) {
        if (tag->tag == DT_STRTAB) strings = base + tag->value;
        if (tag->tag == DT_SYMTAB) symbols = base + tag->value;
        if (tag->tag == DT_HASH) hash = base + tag->value;
    }
    for (auto* tag = reinterpret_cast<const DynamicEntry*>(base + dynamic); tag->tag != DT_NULL; ++tag)
        if (tag->tag == DT_NEEDED) needed.emplace_back(reinterpret_cast<const char*>(strings + tag->value));
    if (hash != 0 && symbols != 0) {
        const std::uint32_t count = reinterpret_cast<const std::uint32_t*>(hash)[1];
        for (std::uint32_t index = 1; index < count; ++index) {
            const auto& symbol = reinterpret_cast<const Symbol*>(symbols)[index];
            const unsigned visibility = symbol.other & 3;
            if (symbol.section == 0 || (symbol.info >> 4) == STB_LOCAL || visibility == STV_INTERNAL || visibility == STV_HIDDEN) continue;
            const bool tlsSymbol = (symbol.info & 15) == STT_TLS;
            exports.emplace(reinterpret_cast<const char*>(strings + symbol.name), Exported {tlsSymbol ? symbol.value : base + symbol.value, tlsSymbol});
        }
    }
}

std::optional<std::uint64_t> GuestImage::StartFunction() const {
    constexpr unsigned char MoveRspToRdiAlignRspClearRsiCall[] = {0x48, 0x89, 0xe7, 0x48, 0x83, 0xe4, 0xf0, 0x48, 0x31, 0xf6, 0xe8};
    constexpr std::size_t CallTarget = sizeof(MoveRspToRdiAlignRspClearRsiCall);
    const auto* code = reinterpret_cast<const unsigned char*>(entry);
    if (entry < start || entry + CallTarget + 6 > start + size || std::memcmp(code, MoveRspToRdiAlignRspClearRsiCall, CallTarget) != 0) return std::nullopt;
    std::int32_t displacement;
    std::memcpy(&displacement, code + CallTarget, sizeof(displacement));
    return entry + CallTarget + sizeof(displacement) + displacement;
}

std::optional<GuestSymbol> GuestImage::Export(const std::string& name) const {
    const auto found = exports.find(name);
    if (found == exports.end()) return std::nullopt;
    if (!found->second.tls) return GuestSymbol {found->second.value};
    return GuestSymbol {0, true, tlsModule, found->second.value, tlsOffset};
}

std::vector<std::uint64_t> GuestImage::Initializers() const {
    std::uint64_t init = 0, array = 0, arraySize = 0;
    for (auto* tag = reinterpret_cast<const DynamicEntry*>(base + dynamic); tag->tag != DT_NULL; ++tag) {
        if (tag->tag == DT_INIT) init = base + tag->value;
        if (tag->tag == DT_INIT_ARRAY) array = base + tag->value;
        if (tag->tag == DT_INIT_ARRAYSZ) arraySize = tag->value;
    }
    std::vector<std::uint64_t> functions;
    if (init != 0) functions.push_back(init);
    for (std::size_t i = 0; i < arraySize / sizeof(std::uint64_t); ++i) {
        const std::uint64_t function = reinterpret_cast<const std::uint64_t*>(array)[i];
        if (function != 0 && function != ~std::uint64_t {0}) functions.push_back(function);
    }
    return functions;
}

GuestImage::~GuestImage() {
    munmap(reinterpret_cast<void*>(start), size);
}

void GuestImage::Relocate(const Resolver& resolve) {
    std::uint64_t strings = 0, symbols = 0, rela = 0, relaSize = 0, plt = 0, pltSize = 0;
    for (auto* tag = reinterpret_cast<const DynamicEntry*>(base + dynamic); tag->tag != DT_NULL; ++tag) {
        switch (tag->tag) {
        case DT_STRTAB: strings = base + tag->value; break;
        case DT_SYMTAB: symbols = base + tag->value; break;
        case DT_RELA: rela = base + tag->value; break;
        case DT_RELASZ: relaSize = tag->value; break;
        case DT_JMPREL: plt = base + tag->value; break;
        case DT_PLTRELSZ: pltSize = tag->value; break;
        default: break;
        }
    }
    const auto symbolOf = [&](std::uint32_t index) -> GuestSymbol {
        if (index == 0) return {0, false, tlsModule, 0, tlsOffset};
        const auto& symbol = reinterpret_cast<const Symbol*>(symbols)[index];
        if (symbol.section != 0) {
            if ((symbol.info & 15) == STT_TLS) return {0, true, tlsModule, symbol.value, tlsOffset};
            return {base + symbol.value};
        }
        const std::string name(reinterpret_cast<const char*>(strings + symbol.name));
        const bool weak = (symbol.info >> 4) == STB_WEAK;
        if (const auto resolved = resolve(name, weak)) return *resolved;
        if (weak) return {};
        throw std::runtime_error("unresolved import " + name + " in " + path.string());
    };
    const auto tlsOf = [&](std::uint32_t index) {
        const GuestSymbol symbol = symbolOf(index);
        if (index != 0 && !symbol.tls) throw std::runtime_error("a TLS relocation names a symbol that is not TLS in " + path.string());
        return symbol;
    };
    const auto apply = [&](std::uint64_t table, std::uint64_t tableSize) {
        const auto* relocations = reinterpret_cast<const Relocation*>(table);
        for (std::size_t i = 0; i < tableSize / sizeof(Relocation); ++i) {
            const auto& relocation = relocations[i];
            auto* target = reinterpret_cast<std::uint64_t*>(base + relocation.offset);
            const auto type = static_cast<std::uint32_t>(relocation.info);
            const auto symbol = static_cast<std::uint32_t>(relocation.info >> 32);
            switch (type) {
            case R_X86_64_NONE: break;
            case R_X86_64_64: *target = symbolOf(symbol).address + relocation.addend; break;
            case R_X86_64_GLOB_DAT:
            case R_X86_64_JUMP_SLOT: *target = symbolOf(symbol).address; break;
            case R_X86_64_RELATIVE: *target = base + relocation.addend; break;
            case R_X86_64_DTPMOD64: *target = tlsOf(symbol).tlsModule; break;
            case R_X86_64_DTPOFF64: *target = tlsOf(symbol).tlsValue + relocation.addend; break;
            case R_X86_64_TPOFF64: {
                const GuestSymbol tlsSymbol = tlsOf(symbol);
                *target = tlsSymbol.tlsValue + relocation.addend - tlsSymbol.tlsOffset;
                break;
            }
            default: throw std::runtime_error("unsupported relocation type " + std::to_string(type));
            }
        }
    };
    apply(rela, relaSize);
    apply(plt, pltSize);
}

}
