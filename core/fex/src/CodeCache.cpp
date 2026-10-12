#include "CodeCache.hpp"

#include "GuestProgram.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <xxhash.h>

namespace Aps5Fex {

std::optional<std::filesystem::path> CodeCacheDirectory(const std::filesystem::path& executable) {
    if (const char* enabled = std::getenv("APS5_FEX_CACHE"); enabled != nullptr && std::string(enabled) == "0") return std::nullopt;
    if (const char* directory = std::getenv("APS5_FEX_CACHE_DIR"); directory != nullptr && *directory != 0)
        return std::filesystem::absolute(directory);
    static const std::regex TitleId("^[A-Z]{4}[0-9]{5}");
    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == 0) return std::nullopt;
    for (auto directory = std::filesystem::absolute(executable).parent_path(); directory.has_relative_path(); directory = directory.parent_path()) {
        std::smatch match;
        const std::string name = directory.filename().string();
        if (std::regex_search(name, match, TitleId))
            return std::filesystem::path(home) / "Library" / "Caches" / "AnyPS5" / match.str(0) / "fex";
    }
    return std::nullopt;
}

GuestModules::GuestModules(const GuestProgram& program) {
    for (const auto& image : program.Images()) {
        Module module;
        module.path = image->Path().string();
        module.start = image->Start();
        module.size = image->Size();
        module.executable = image->ExecutableRanges();
        module.info = std::make_unique<FEXCore::ExecutableFileInfo>();
        module.info->FileId = image->FileId();
        module.info->Filename = module.path.c_str();
        // Lets the cache keep blocks whose constants point anywhere in this module (data included).
        module.info->MappedSize = image->Size();
        module.seedImage = {image->Base(), image->ProgramHeaders(), image->ProgramHeaderCount(), image.get() == &program.Executable() ? image->Entry() : 0,
                            image->ExportAddresses()};
        modules.push_back(std::move(module));
    }
}

std::optional<FEXCore::ExecutableFileSectionInfo> GuestModules::Section(std::uint64_t address) const {
    for (const auto& module : modules)
        for (const auto& [begin, end] : module.executable)
            if (address >= begin && address < end) return FEXCore::ExecutableFileSectionInfo {*module.info, module.start, begin, end};
    return std::nullopt;
}

std::optional<FEXCore::HLE::ExecutableRangeInfo> GuestModules::ExecutableRange(std::uint64_t address) const {
    for (const auto& module : modules)
        for (const auto& [begin, end] : module.executable)
            if (address >= begin && address < end) return FEXCore::HLE::ExecutableRangeInfo {begin, end - begin, false};
    return std::nullopt;
}

namespace {

struct Block {
    std::uint64_t entry;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges; // [begin, end)
    FEXCore::Context::CompiledBlockOrigin origin;
};

std::mutex coverageMutex;
bool coverageEnabled = false;
std::vector<Block> blocks;

using Ranges = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

Ranges Merge(Ranges ranges) {
    std::sort(ranges.begin(), ranges.end());
    Ranges merged;
    for (const auto& range : ranges) {
        if (range.first >= range.second) continue;
        if (!merged.empty() && range.first <= merged.back().second) merged.back().second = std::max(merged.back().second, range.second);
        else merged.push_back(range);
    }
    return merged;
}

std::uint64_t Bytes(const Ranges& merged) {
    std::uint64_t total = 0;
    for (const auto& [begin, end] : merged) total += end - begin;
    return total;
}

// Clips merged ranges to [low, high) and makes them relative to base.
Ranges Relative(const Ranges& merged, std::uint64_t low, std::uint64_t high, std::uint64_t base) {
    Ranges clipped;
    for (const auto& [begin, end] : merged) {
        const std::uint64_t b = std::max(begin, low), e = std::min(end, high);
        if (b < e) clipped.emplace_back(b - base, e - base);
    }
    return clipped;
}

void WriteRanges(std::FILE* file, const Ranges& ranges) {
    std::fputc('[', file);
    for (std::size_t i = 0; i < ranges.size(); ++i)
        std::fprintf(file, "%s[%llu, %llu]", i == 0 ? "" : ", ", static_cast<unsigned long long>(ranges[i].first),
                     static_cast<unsigned long long>(ranges[i].second - ranges[i].first));
    std::fputc(']', file);
}

std::string JsonString(const std::string& text) {
    std::string quoted = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') quoted.push_back('\\');
        if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            quoted += escaped;
        } else {
            quoted.push_back(c);
        }
    }
    return quoted + "\"";
}

bool InModule(const GuestModules::Module& module, std::uint64_t address) {
    return address >= module.start && address < module.start + module.size;
}

}

bool Coverage::Enabled() {
    return coverageEnabled;
}

void Coverage::Enable() {
    coverageEnabled = true;
}

void Coverage::Observe(std::uint64_t entry, const std::uint64_t* ranges, std::size_t count, FEXCore::Context::CompiledBlockOrigin origin) {
    Block block {entry, {}, origin};
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t begin = ranges[2 * i], end = begin + ranges[2 * i + 1];
        // A cached block's extents leave out the data bytes FEX masks (immediates, displacements): close gaps
        // shorter than an instruction so the report counts whole instructions.
        constexpr std::uint64_t MaxInstruction = 15;
        if (!block.ranges.empty() && begin >= block.ranges.back().second && begin - block.ranges.back().second < MaxInstruction)
            block.ranges.back().second = std::max(block.ranges.back().second, end);
        else
            block.ranges.emplace_back(begin, end);
    }
    std::lock_guard lock(coverageMutex);
    blocks.push_back(std::move(block));
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> Coverage::CompiledRanges() {
    std::lock_guard lock(coverageMutex);
    Ranges ranges;
    for (const Block& block : blocks)
        if (block.origin == FEXCore::Context::CompiledBlockOrigin::JIT) ranges.insert(ranges.end(), block.ranges.begin(), block.ranges.end());
    return Merge(std::move(ranges));
}

std::size_t Coverage::JitBlocksInModules(const GuestModules& modules) {
    std::lock_guard lock(coverageMutex);
    return static_cast<std::size_t>(std::count_if(blocks.begin(), blocks.end(), [&](const Block& block) {
        return block.origin == FEXCore::Context::CompiledBlockOrigin::JIT &&
               std::any_of(modules.Modules().begin(), modules.Modules().end(), [&](const auto& module) { return InModule(module, block.entry); });
    }));
}

void Coverage::Write(const std::filesystem::path& path, const char* mode, const GuestModules& modules, const std::vector<AotSeeds>* seeds) {
    std::lock_guard lock(coverageMutex);
    std::FILE* file = std::fopen(path.c_str(), "w");
    if (file == nullptr) throw std::runtime_error("cannot write the coverage report " + path.string());
    std::size_t jitOutside = 0, hitOutside = 0;
    for (const Block& block : blocks) {
        const bool inside = std::any_of(modules.Modules().begin(), modules.Modules().end(), [&](const auto& module) { return InModule(module, block.entry); });
        if (!inside) (block.origin == FEXCore::Context::CompiledBlockOrigin::JIT ? jitOutside : hitOutside) += 1;
    }
    std::fprintf(file, "{\n  \"version\": 1,\n  \"mode\": \"%s\",\n  \"blocks_outside_modules\": {\"jit\": %zu, \"cache_hit\": %zu},\n  \"modules\": [", mode,
                 jitOutside, hitOutside);
    const auto& list = modules.Modules();
    for (std::size_t m = 0; m < list.size(); ++m) {
        const auto& module = list[m];
        Ranges jit, hit;
        std::size_t jitBlocks = 0, hitBlocks = 0;
        for (const Block& block : blocks) {
            if (!InModule(module, block.entry)) continue;
            const bool compiled = block.origin == FEXCore::Context::CompiledBlockOrigin::JIT;
            ++(compiled ? jitBlocks : hitBlocks);
            auto& target = compiled ? jit : hit;
            target.insert(target.end(), block.ranges.begin(), block.ranges.end());
        }
        jit = Merge(std::move(jit));
        hit = Merge(std::move(hit));
        Ranges text;
        for (const auto& range : module.executable) text.push_back(range);
        text = Merge(std::move(text));
        std::fprintf(file, "%s\n    {\n      \"path\": %s,\n      \"file_id\": \"%016llx\",\n      \"size\": %llu,\n      \"text\": ", m == 0 ? "" : ",",
                     JsonString(module.path).c_str(), static_cast<unsigned long long>(module.info->FileId), static_cast<unsigned long long>(module.size));
        WriteRanges(file, Relative(text, module.start, module.start + module.size, module.start));
        std::fprintf(file, ",\n      \"text_bytes\": %llu,\n      \"jit_blocks\": %zu,\n      \"cache_hit_blocks\": %zu,\n      \"jit\": ",
                     static_cast<unsigned long long>(Bytes(text)), jitBlocks, hitBlocks);
        WriteRanges(file, Relative(jit, module.start, module.start + module.size, module.start));
        std::fprintf(file, ",\n      \"cache_hit\": ");
        WriteRanges(file, Relative(hit, module.start, module.start + module.size, module.start));
        if (seeds != nullptr && m < seeds->size()) {
            const AotSeeds& found = (*seeds)[m];
            std::fprintf(file, ",\n      \"seeds\": {\"total\": %zu, \"entry\": %zu, \"exports\": %zu, \"init_fini\": %zu, \"relocations\": %zu, "
                               "\"fdes\": %zu, \"fde_table\": %s},\n      \"fde\": ",
                         found.addresses.size(), found.entryPoints, found.exports, found.initFini, found.relocations, found.fdes,
                         found.fdeTableUsed ? "true" : "false");
            WriteRanges(file, Relative(Merge(found.fdeRanges), module.start, module.start + module.size, module.start));
        }
        std::fprintf(file, "\n    }");
    }
    std::fprintf(file, "\n  ]\n}\n");
    std::fclose(file);
}

}
