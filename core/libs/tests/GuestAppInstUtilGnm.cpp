#include "prx/libc/include/general/VabiMacros.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

extern "C" {
int APS5_VABI sceAppInstUtilInitialize();
int APS5_VABI sceAppInstUtilTerminate();
int APS5_VABI sceAppInstUtilAppExists();
int APS5_VABI sceAppInstUtilAppInstallPkg();
int APS5_VABI sceAppInstUtilAppUnInstall();
int APS5_VABI sceAppInstUtilGetInstallProgress();
int APS5_VABI sceAppInstUtilGetInstallProgressInfo();
int APS5_VABI sceGnmSubmitCommandBuffers();
int APS5_VABI sceGnmSubmitDone();
int APS5_VABI sceGnmDrawIndex();
int APS5_VABI sceGnmDispatchDirect();
int APS5_VABI sceGnmDispatchInitDefaultHardwareState();
}

namespace {

bool ReportsNotImplemented(const char* name, int (APS5_VABI *function)()) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) == std::string(name) + " not implemented") return true;
        std::fprintf(stderr, "%s threw: %s\n", name, error.what());
        return false;
    }
    std::fprintf(stderr, "%s returned instead of reporting that it is not implemented\n", name);
    return false;
}

bool ThrowsLogic(int (APS5_VABI *function)()) {
    try {
        function();
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

}

int main() {
    bool correct = true;
    correct &= ThrowsLogic(sceAppInstUtilTerminate);
    correct &= sceAppInstUtilInitialize() == 0;
    correct &= ThrowsLogic(sceAppInstUtilInitialize);
    correct &= sceAppInstUtilTerminate() == 0;
    correct &= ReportsNotImplemented("sceAppInstUtilAppExists", sceAppInstUtilAppExists);
    correct &= ReportsNotImplemented("sceAppInstUtilAppInstallPkg", sceAppInstUtilAppInstallPkg);
    correct &= ReportsNotImplemented("sceAppInstUtilAppUnInstall", sceAppInstUtilAppUnInstall);
    correct &= ReportsNotImplemented("sceAppInstUtilGetInstallProgress", sceAppInstUtilGetInstallProgress);
    correct &= ReportsNotImplemented("sceAppInstUtilGetInstallProgressInfo", sceAppInstUtilGetInstallProgressInfo);
    correct &= ReportsNotImplemented("sceGnmSubmitCommandBuffers", sceGnmSubmitCommandBuffers);
    correct &= ReportsNotImplemented("sceGnmSubmitDone", sceGnmSubmitDone);
    correct &= ReportsNotImplemented("sceGnmDrawIndex", sceGnmDrawIndex);
    correct &= ReportsNotImplemented("sceGnmDispatchDirect", sceGnmDispatchDirect);
    correct &= ReportsNotImplemented("sceGnmDispatchInitDefaultHardwareState", sceGnmDispatchInitDefaultHardwareState);
    return correct ? 0 : 1;
}
