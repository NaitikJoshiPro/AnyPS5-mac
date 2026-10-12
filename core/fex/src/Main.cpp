
#include "Bridge.hpp"
#include "GuestCpu.hpp"
#include "GuestProgram.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
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

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <relinked executable> [arguments...]\n", argv[0]);
        return 2;
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
