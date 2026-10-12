#ifndef CORE_FEX_SRC_GUESTCPU_HPP
#define CORE_FEX_SRC_GUESTCPU_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace FEXCore::Core {
struct CPUState;
}

namespace Aps5Fex {

class Bridge;
class GuestProgram;

std::uint64_t CallGuestFunction(std::uint64_t target, const std::uint64_t* arguments);
bool IsGuestCodeAddress(std::uint64_t address);

using GuestStateHook = void (*)(FEXCore::Core::CPUState& state, void* context);
void CallGuestWith(std::uint64_t target, GuestStateHook setup, GuestStateHook finish, void* context);

class GuestCpu {
public:
    GuestCpu(Bridge& bridge, const GuestProgram& program);
    GuestCpu(const GuestCpu&) = delete;
    GuestCpu& operator=(const GuestCpu&) = delete;
    ~GuestCpu();

    std::uint64_t Run(std::uint64_t rip, std::uint64_t rsp, const std::vector<std::uint64_t>& initializers);

    [[noreturn]] void Start(Bridge& bridge, std::uint64_t start, std::uint64_t block, std::vector<std::uint64_t> initializers);

    // Compiles the statically found code of every module (AotSeeds) on worker threads into the disk cache, without
    // running the guest. Writes the coverage report to coverage when it is not empty. Returns the seed count.
    std::size_t Precompile(unsigned threads, const std::filesystem::path& coverage);

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
