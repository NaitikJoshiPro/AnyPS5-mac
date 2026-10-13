// Hardware identity and fan queries. They are informational; each returns what the host can honestly say:
// - sceKernelGetHwModelName: the host's model identifier (sysctl hw.model, e.g. "Mac15,12"); the console
//   writes its model ("CFI-...") into a caller buffer of at least 1000 bytes (oops-sdk passes 1024,
//   ps5-payload-sdk's hwinfo sample 1000). No size is passed, so at most 32 bytes are written.
// - sceKernelGetHwSerialNumber: a fixed placeholder, so that the host's serial number never reaches guest
//   code (which may send it over the network).
// - sceKernelGetCurrentFanDuty: duty 0. Callers disagree on the arguments: the oops-sdk declares
//   (int* unknown, int* duty 0..255), console tools (fan_target, ps5-exporter) (uint16_t* duty 0..1024,
//   uint64_t* chassis). Writing a 16-bit 0 to the first and a 32-bit 0 to the second fits both. The
//   MacBook Air this port targets has no fan; Macs with fans report 0 as well, since fan speed is only
//   readable through the undocumented SMC interface.
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace {

constexpr int SCE_KERNEL_ERROR_EFAULT = static_cast<int>(0x8002000Eu);
constexpr std::size_t MAX_WRITTEN = 32;
constexpr const char* SERIAL_PLACEHOLDER = "APS5-HOST-000000";

std::string HostModel() {
#ifdef __APPLE__
    char model[MAX_WRITTEN] = {};
    std::size_t size = sizeof(model) - 1;
    if (::sysctlbyname("hw.model", model, &size, nullptr, 0) != 0)
        throw std::runtime_error("sceKernelGetHwModelName: sysctl hw.model failed");
    return model;
#else
    return "AnyPS5 host";
#endif
}

int CopyOut(char* buffer, const std::string& text) {
    if (buffer == nullptr) return SCE_KERNEL_ERROR_EFAULT;
    const std::size_t length = std::min(text.size(), MAX_WRITTEN - 1);
    std::memcpy(buffer, text.data(), length);
    buffer[length] = '\0';
    return 0;
}

}

extern "C" {

int APS5_VABI sceKernelGetHwModelName(char* buffer) {
    return CopyOut(buffer, HostModel());
}

int APS5_VABI sceKernelGetHwSerialNumber(char* buffer) {
    return CopyOut(buffer, SERIAL_PLACEHOLDER);
}

int APS5_VABI sceKernelGetCurrentFanDuty(void* first, void* second) {
    if (first == nullptr && second == nullptr) return SCE_KERNEL_ERROR_EFAULT;
    if (first != nullptr) {
        const std::uint16_t duty = 0;
        std::memcpy(first, &duty, sizeof(duty));
    }
    if (second != nullptr) {
        const std::uint32_t value = 0;
        std::memcpy(second, &value, sizeof(value));
    }
    return 0;
}

}
