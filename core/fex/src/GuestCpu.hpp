#ifndef CORE_FEX_SRC_GUESTCPU_HPP
#define CORE_FEX_SRC_GUESTCPU_HPP

#include <cstdint>
#include <memory>
#include <vector>

namespace Aps5Fex {

class Bridge;
class GuestProgram;

std::uint64_t CallGuestFunction(std::uint64_t target, const std::uint64_t* arguments);

class GuestCpu {
public:
    GuestCpu(Bridge& bridge, const GuestProgram& program);
    GuestCpu(const GuestCpu&) = delete;
    GuestCpu& operator=(const GuestCpu&) = delete;
    ~GuestCpu();

    std::uint64_t Run(std::uint64_t rip, std::uint64_t rsp, const std::vector<std::uint64_t>& initializers);

    [[noreturn]] void Start(Bridge& bridge, std::uint64_t start, std::uint64_t block, std::vector<std::uint64_t> initializers);

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
