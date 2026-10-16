#include "GuestGuards.hpp"

#include "Bridge.hpp"
#include "GuestProgram.hpp"

#include <FEXCore/Core/Context.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace Aps5Fex {

namespace {

constexpr std::array<const char*, 3> GuardFunctions {"__cxa_guard_acquire", "__cxa_guard_release", "__cxa_guard_abort"};
// The same functions' NIDs: a PS5 executable exports by NID, homebrew (oops-sdk) by name.
constexpr std::array<const char*, 3> GuardNids {"3GPpjQdAMTw", "9rAeANT2tyE", "2emaaluWzUw"};

}

std::size_t RedirectGuestGuards(Bridge& bridge, const GuestProgram& program, FEXCore::Context::Context& context) {
    if (const char* off = std::getenv("APS5_GUEST_GUARDS"); off != nullptr && std::string(off) == "0") return 0;
    std::size_t redirected = 0;
    std::array<std::uint64_t, GuardFunctions.size()> stubs {};
    for (const auto& image : program.Images()) {
        std::array<std::uint64_t, GuardFunctions.size()> guest {};
        bool complete = true;
        for (std::size_t i = 0; i < GuardFunctions.size(); ++i) {
            auto symbol = image->Export(GuardFunctions[i]);
            if (!symbol) symbol = image->Export(GuardNids[i]);
            if (!symbol || symbol->tls || symbol->address == 0) complete = false;
            else guest[i] = symbol->address;
        }
        if (!complete) continue;
        if (stubs[0] == 0) {
            for (std::size_t i = 0; i < GuardFunctions.size(); ++i) {
                // libc exports its guards (exception/GuardSupport.cpp) under their NIDs.
                void* function = bridge.HostSymbol(GuardNids[i]);
                const std::string name = std::string(GuardFunctions[i]) + " (" + GuardNids[i] + ")";
                if (function == nullptr) throw std::runtime_error("libc has no " + name + " for the guest's own static-initialisation guards");
                stubs[i] = bridge.StubFor(function, GuardNids[i]);
                if (stubs[i] == 0) throw std::runtime_error("libc's " + name + " has no guest stub");
            }
        }
        for (std::size_t i = 0; i < GuardFunctions.size(); ++i) context.AddThunkTrampolineIRHandler(guest[i], stubs[i]);
        std::fprintf(stderr, "[guards] %s: its own __cxa_guard_* run libc's thread-safe guards\n", image->Path().filename().c_str());
        ++redirected;
    }
    return redirected;
}

}
