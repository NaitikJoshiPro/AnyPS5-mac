#pragma once
#include <cstdint>

namespace GuestSockets {

struct SocketOptionState {
    int reportedError = 0;
    bool nonblocking = false;
    bool noSignal = false;
    std::int64_t sendTimeout = 0;
    std::int64_t receiveTimeout = 0;
    // A successful listen; read for SO_ACCEPTCONN where the host's getsockopt lacks it (macOS).
    bool listening = false;
};

int SetSocketLevelOption(std::uintptr_t native, SocketOptionState& state, int option, const void* value, std::uint32_t length);
int GetSocketLevelOption(std::uintptr_t native, SocketOptionState& state, int type, int option, void* value, std::uint32_t* length);
int SetTcpLevelOption(std::uintptr_t native, bool tcp, int option, const void* value, std::uint32_t length);
int GetTcpLevelOption(std::uintptr_t native, bool tcp, int option, void* value, std::uint32_t* length);

}
