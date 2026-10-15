#include "prx/libSceAgc/DcbState/include/Init.hpp"

#include "prx/libSceAgc/Command/include/Packet.hpp"
#include "prx/libSceAgc/Command/include/RegisterDefaults.hpp"
#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

// NID kW3GLb7QfPg is called both as (version) by DOOM PPSA99005 and as (state, 0xd) by the oops-sdk/oops-mesa
// (DECISIONS D34). Upstream's console probe (#4808) shows every out-of-range first argument, including a state
// pointer, succeeds without writes. The full 64-bit register is compared so a state pointer whose low 32 bits are
// small still takes that path.
int APS5_VABI sceAgcInit(std::uint64_t version) {
    if (version >= 14) return 0;
    Agc::Command::GetRegisterDefaults(static_cast<std::uint32_t>(version), false, __func__);
    Agc::Command::GetRegisterDefaults(static_cast<std::uint32_t>(version), true, __func__);
    return 0;
}

int APS5_VABI sceAgcInit_0090(std::uint32_t* state, std::uint32_t version) {
    Agc::Command::CheckAddress(reinterpret_cast<std::uintptr_t>(state), alignof(std::uint32_t), __func__);
    return sceAgcInit(version);
}

}
