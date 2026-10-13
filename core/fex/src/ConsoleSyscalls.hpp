#ifndef CORE_FEX_SRC_CONSOLESYSCALLS_HPP
#define CORE_FEX_SRC_CONSOLESYSCALLS_HPP

#include <cstdint>

namespace Aps5Fex {

class Bridge;

constexpr std::uint64_t ConsoleLibkernelBase = 0x800000000;
constexpr std::uint64_t ConsoleSyscallGadget = ConsoleLibkernelBase + 0x4ea;

std::uint64_t CreateConsoleSyscallGadget(Bridge& bridge);

}

#endif
