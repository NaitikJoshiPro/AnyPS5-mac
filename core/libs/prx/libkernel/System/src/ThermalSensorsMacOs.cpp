// macOS has no public API for SoC or CPU temperatures (the sensors sit behind the undocumented
// IOHIDEventSystem / SMC interfaces). It publishes the thermal pressure level instead
// (libkern/OSThermalNotification.h, the "com.apple.system.thermalpressurelevel" notification state),
// which is what drives its own throttling. sceKernelGetCpuTemperature and SoC sensors 0 and 1 report a
// representative temperature per level, so a title that throttles on temperature sees the host's
// pressure: nominal 50 C, moderate 70 C, heavy 85 C, trapping 95 C, sleeping 105 C. Sensor 2 (storage)
// has no equivalent and stays unavailable (EINVAL). The values are not measurements; TechnicalDebt
// records this.
#ifdef __APPLE__
#include <cstdint>
#include <libkern/OSThermalNotification.h>
#include <notify.h>

extern "C" int ThermalPressureCelsius_nid_no_patch(std::uint64_t level) {
    switch (level) {
    case kOSThermalPressureLevelNominal: return 50;
    case kOSThermalPressureLevelModerate: return 70;
    case kOSThermalPressureLevelHeavy: return 85;
    case kOSThermalPressureLevelTrapping: return 95;
    case kOSThermalPressureLevelSleeping: return 105;
    default: return -1;
    }
}

void ReadMacOsThermalPressure(int celsius[3]) {
    celsius[0] = celsius[1] = celsius[2] = -1;
    int token = 0;
    if (notify_register_check(kOSThermalNotificationPressureLevelName, &token) != NOTIFY_STATUS_OK) return;
    std::uint64_t level = 0;
    const bool read = notify_get_state(token, &level) == NOTIFY_STATUS_OK;
    notify_cancel(token);
    if (!read) return;
    celsius[0] = celsius[1] = ThermalPressureCelsius_nid_no_patch(level);
}
#endif
