
#include "Bridge.hpp"
#include "CodeCache.hpp"
#include "GuestCpu.hpp"
#include "GuestProgram.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/mman.h>
#include <vector>

namespace {

using namespace Aps5Fex;

constexpr std::size_t GuestStackSize = std::size_t {8} << 20;

std::uint64_t CreateProcessStack(const std::vector<std::string>& arguments) {
    void* memory = mmap(nullptr, GuestStackSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory == MAP_FAILED) throw std::runtime_error("cannot allocate the guest stack");
    auto top = reinterpret_cast<std::uint64_t>(memory) + GuestStackSize;
    std::vector<std::uint64_t> pointers;
    for (const auto& argument : arguments) {
        top -= argument.size() + 1;
        std::memcpy(reinterpret_cast<void*>(top), argument.c_str(), argument.size() + 1);
        pointers.push_back(top);
    }
    std::vector<std::uint64_t> words {arguments.size()};
    words.insert(words.end(), pointers.begin(), pointers.end());
    words.insert(words.end(), {0, 0, 0, 0});
    top = (top - words.size() * sizeof(std::uint64_t)) & ~std::uint64_t {15};
    std::memcpy(reinterpret_cast<void*>(top), words.data(), words.size() * sizeof(std::uint64_t));
    return top;
}

// aps5-fex --precompile <relinked executable> [--coverage <json>] [--threads <n>]: compiles the code found statically
// in the executable and its guest modules into the disk cache (APS5_FEX_CACHE_DIR or the title's cache directory).
int Precompile(int argc, char** argv) {
    std::filesystem::path coverage;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::filesystem::path elf;
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--coverage" && i + 1 < argc) coverage = argv[++i];
        else if (argument == "--threads" && i + 1 < argc) threads = static_cast<unsigned>(std::stoul(argv[++i]));
        else if (elf.empty()) elf = argument;
        else throw std::runtime_error("--precompile: unexpected argument " + argument);
    }
    if (elf.empty()) throw std::runtime_error("--precompile needs an executable");
    const auto executable = std::filesystem::absolute(elf);
    if (!CodeCacheDirectory(executable)) throw std::runtime_error("--precompile: no cache directory (set APS5_FEX_CACHE_DIR)");
    Coverage::Enable(); // precompile follows the compiled blocks to find return sites
    Bridge bridge(executable.parent_path() / "libs");
    GuestProgram program(executable, bridge);
    GuestCpu cpu(bridge, program);
    const std::size_t seeds = cpu.Precompile(threads, coverage);
    std::fprintf(stderr, "[aps5-fex] precompiled %zu seeds into %s\n", seeds, CodeCacheDirectory(executable)->c_str());
    return 0;
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <relinked executable> [arguments...]\n       %s --precompile <relinked executable> [--coverage <json>] [--threads <n>]\n", argv[0], argv[0]);
        return 2;
    }
    if (std::string(argv[1]) == "--precompile") {
        try {
            return Precompile(argc, argv);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[aps5-fex] %s\n", error.what());
            return 1;
        }
    }
    try {
        const auto executable = std::filesystem::absolute(argv[1]);
        Bridge bridge(executable.parent_path() / "libs");
        GuestProgram program(executable, bridge);
        if (!bridge.Missing().empty()) {
            for (const auto& name : bridge.Missing()) std::fprintf(stderr, "[aps5-fex] no library exports %s\n", name.c_str());
        }
        const std::uint64_t rsp = CreateProcessStack(std::vector<std::string>(argv + 1, argv + argc));
        GuestCpu cpu(bridge, program);
        if (const auto start = program.Executable().StartFunction()) cpu.Start(bridge, *start, rsp, program.Initializers());
        const std::uint64_t rax = cpu.Run(program.Executable().Entry(), rsp, program.Initializers());
        std::fprintf(stderr, "[aps5-fex] the guest halted with RAX %#llx\n", static_cast<unsigned long long>(rax));
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[aps5-fex] %s\n", error.what());
        return 1;
    }
}
