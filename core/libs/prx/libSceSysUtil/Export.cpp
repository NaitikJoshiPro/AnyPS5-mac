#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <cstdint>
#include <string>

extern "C" {

int APS5_VABI sceSysUtilSendSystemNotificationWithText(int32_t type, const char* text) {
    if (text == nullptr) APS5_INVALID_ARG_EX;
    APS5_LOG_ERR("system notification (type %d): %s", type, text);
    return 0;
}

}
