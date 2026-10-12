#ifndef CORE_FEX_SRC_BAREGUEST_HPP
#define CORE_FEX_SRC_BAREGUEST_HPP

#include <FEXCore/Config/Config.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/HostFeatures.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Aps5Fex {

struct BareGuest {
    std::uint64_t rip = 0;
    std::uint64_t rsp = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> executable;
    std::vector<std::pair<FEXCore::Config::ConfigOption, std::string>> config;
};

struct BareGuestResult {
    FEXCore::Core::CPUState state {};
    std::uint32_t syscalls = 0;
    std::uint64_t firstSyscall = 0;
};

FEXCore::HostFeatures RunnerHostFeatures();

void RunBareGuest(const BareGuest& guest, BareGuestResult& result);

std::uint64_t BareGuestRip(std::uint64_t hostPc);

}

#endif
