// The running title's identity and state as libSceSystemService reports them. On AnyPS5 the title is the
// only application and the foreground "big app":
// - sceSystemServiceGetMainAppTitleId writes the title ID from /app0/sce_sys/param.json (NUL-terminated,
//   9 characters; callers pass at least 16 bytes).
// - sceSystemServiceGetAppIdOfBigApp returns the running title's app id, the one
//   sceSystemServiceGetAppIdOfRunningBigApp and sceSystemServiceKillApp use.
// - sceSystemServiceIsAppSuspended returns 0: a title that is running code is not suspended.
// - sceSystemServiceNavigateToGoHome: on the console the home screen comes up and the title keeps
//   running in the background until it is killed. The host has no home screen; the title keeps running
//   and its window stays where it is. The call is logged and returns 0 (TechnicalDebt).
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libkernel/AppMetadata/include/AppMetadata.hpp"
#include "prx/libSceSystemService/SystemService.hpp"
#include <cstdio>
#include <cstring>

extern "C" {

int APS5_VABI sceSystemServiceGetMainAppTitleId(char* titleId) {
    if (titleId == nullptr) return SYSTEM_SERVICE_ERROR_PARAMETER;
    const AppTitleId id = GetAppTitleId_nid_postfix();
    std::memcpy(titleId, id.value, std::strlen(id.value) + 1);
    return SYSTEM_SERVICE_OK;
}

int APS5_VABI sceSystemServiceGetAppIdOfBigApp(void) {
    return SYSTEM_SERVICE_RUNNING_APP_ID;
}

int APS5_VABI sceSystemServiceIsAppSuspended(void) {
    return 0;
}

int APS5_VABI sceSystemServiceNavigateToGoHome(void) {
    std::fprintf(stderr, "[libSceSystemService] sceSystemServiceNavigateToGoHome: the host has no home screen; the title keeps running\n");
    return SYSTEM_SERVICE_OK;
}

}
