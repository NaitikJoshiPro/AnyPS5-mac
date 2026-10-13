// sceSystemServiceGetMainAppTitleId, GetAppIdOfBigApp, IsAppSuspended and NavigateToGoHome. Runs in a
// directory whose app0/sce_sys/param.json names PPSA00000.
#include "SceTypes.hpp"
#include "prx/libSceSystemService/SystemService.hpp"
#include <cstdlib>
#include <cstring>

extern "C" {
int APS5_VABI sceSystemServiceGetMainAppTitleId(char* titleId);
int APS5_VABI sceSystemServiceGetAppIdOfBigApp(void);
int APS5_VABI sceSystemServiceGetAppIdOfRunningBigApp(void);
int APS5_VABI sceSystemServiceIsAppSuspended(void);
int APS5_VABI sceSystemServiceNavigateToGoHome(void);
}

static void Require(bool value) { if (!value) std::abort(); }

int main() {
    char titleId[64];
    std::memset(titleId, 'x', sizeof(titleId));
    Require(sceSystemServiceGetMainAppTitleId(titleId) == SYSTEM_SERVICE_OK);
    Require(std::strcmp(titleId, "PPSA00000") == 0 && titleId[10] == 'x');
    Require(sceSystemServiceGetMainAppTitleId(nullptr) == SYSTEM_SERVICE_ERROR_PARAMETER);
    Require(sceSystemServiceGetAppIdOfBigApp() == sceSystemServiceGetAppIdOfRunningBigApp());
    Require(sceSystemServiceGetAppIdOfBigApp() > 0);
    Require(sceSystemServiceIsAppSuspended() == 0);
    Require(sceSystemServiceNavigateToGoHome() == SYSTEM_SERVICE_OK);
    Require(sceSystemServiceIsAppSuspended() == 0);
}
