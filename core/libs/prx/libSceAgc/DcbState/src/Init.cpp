#include "prx/libSceAgc/DcbState/include/Init.hpp"

#include "prx/libSceAgc/Command/include/Packet.hpp"
#include "prx/libSceAgc/Command/include/RegisterDefaults.hpp"
#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

static constexpr int SCE_AGC_ERROR_INVALID_ARGUMENT = static_cast<int>(0x80890001);

extern "C" {

int APS5_VABI sceAgcInit_0090(std::uint32_t* state, std::uint32_t version) {
    Agc::Command::CheckAddress(reinterpret_cast<std::uintptr_t>(state), alignof(std::uint32_t), __func__);
    Agc::Command::Require(version < 14, __func__, "unsupported AGC initialization version");
    return 0;
}

// NID kW3GLb7QfPg is called in two forms that both return 0 on the console: DOOM PPSA99005 passes one
// version (`movl $8, %edi`), and the oops-sdk/oops-mesa (SuperTuxKart) pass (state, 0xd), which oops-mesa's
// winsys requires to return 0 before its first submission (oops-mesa src/winsys/submit.c ensure_queue; the
// GL context renders on hardware, oops-mesa docs/WORKLOG.md 2026-09-20). A first argument below 14 is a
// version; a first argument at or above the guest's lowest mappable address is a state pointer and the
// version comes from the second argument. The state block is left as the caller zeroed it: what the console
// writes there is unknown and nothing in AnyPS5 reads it (DECISIONS D34).
static constexpr std::uint64_t LOWEST_STATE_ADDRESS = 0x10000;

int APS5_VABI sceAgcInit(std::uint64_t versionOrState, std::uint32_t stateVersion) {
    std::uint64_t version = versionOrState;
    if (versionOrState >= LOWEST_STATE_ADDRESS) {
        Agc::Command::CheckAddress(static_cast<std::uintptr_t>(versionOrState), alignof(std::uint32_t), __func__);
        version = stateVersion;
    }
    if (version >= 14) return SCE_AGC_ERROR_INVALID_ARGUMENT;
    Agc::Command::GetRegisterDefaults(version, false, __func__);
    Agc::Command::GetRegisterDefaults(version, true, __func__);
    return 0;
}

}
