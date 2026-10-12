#ifndef CORE_FEX_SRC_CODECACHE_HPP
#define CORE_FEX_SRC_CODECACHE_HPP

#include "AotSeeds.hpp"

#include <FEXCore/Core/CodeCache.h>
#include <FEXCore/HLE/SourcecodeResolver.h>
#include <FEXCore/Core/Context.h>
#include <FEXCore/HLE/SyscallHandler.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Aps5Fex {

class GuestProgram;

// Where FEXCore's disk cache for a title lives: APS5_FEX_CACHE_DIR when set; otherwise, when a directory above the
// executable is named after a title ID (PPSA01234...), ~/Library/Caches/AnyPS5/<title ID>/fex; otherwise none (no
// cache). APS5_FEX_CACHE=0 turns it off. Entries are checked against the live guest bytes on every hit, so a
// changed binary misses instead of running stale code.
std::optional<std::filesystem::path> CodeCacheDirectory(const std::filesystem::path& executable);

// The guest modules as FEXCore's code cache sees them: a stable FileId per file (a hash of its bytes), the module's
// start as FileStartVA (cache keys are offsets from it, so they survive a different load address), the executable
// segments as code sections.
class GuestModules {
public:
    explicit GuestModules(const GuestProgram& program);

    std::optional<FEXCore::ExecutableFileSectionInfo> Section(std::uint64_t address) const;
    std::optional<FEXCore::HLE::ExecutableRangeInfo> ExecutableRange(std::uint64_t address) const;

    struct Module {
        std::string path;
        std::uint64_t start = 0;
        std::uint64_t size = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> executable;
        std::unique_ptr<FEXCore::ExecutableFileInfo> info;
        AotSeedImage seedImage;
    };
    const std::vector<Module>& Modules() const { return modules; }

private:
    std::vector<Module> modules;
};

// Records the guest bytes behind every block FEXCore adds (JIT or disk cache) for the coverage report. Enabled by
// APS5_FEX_COVERAGE=<file> (or --coverage for --precompile); the JSON is written by WriteCoverage.
class Coverage {
public:
    static bool Enabled();
    static void Enable();
    static void Observe(std::uint64_t entry, const std::uint64_t* ranges, std::size_t count, FEXCore::Context::CompiledBlockOrigin origin);
    // mode: "precompile" or "run". seeds: per module (precompile only), same order as modules.Modules().
    static void Write(const std::filesystem::path& file, const char* mode, const GuestModules& modules, const std::vector<AotSeeds>* seeds);
    static std::size_t JitBlocksInModules(const GuestModules& modules);
    // Merged [begin, end) of every JIT-compiled block so far, absolute.
    static std::vector<std::pair<std::uint64_t, std::uint64_t>> CompiledRanges();
};

}

#endif
