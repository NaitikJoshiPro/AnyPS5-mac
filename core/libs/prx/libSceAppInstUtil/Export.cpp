#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <atomic>
#include <stdexcept>
#include <string>

namespace {
std::atomic<bool> g_initialized{false};
}

extern "C" {

int APS5_VABI sceAppInstUtilInitialize() {
    bool expected = false;
    if (!g_initialized.compare_exchange_strong(expected, true)) throw std::logic_error(std::string(__func__) + ": already initialized");
    return 0;
}

int APS5_VABI sceAppInstUtilTerminate() {
    bool expected = true;
    if (!g_initialized.compare_exchange_strong(expected, false)) throw std::logic_error(std::string(__func__) + ": not initialized");
    return 0;
}

int APS5_VABI sceAppInstUtilAppExists() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAppInstUtilAppInstallPkg() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAppInstUtilAppUnInstall() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAppInstUtilGetInstallProgress() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceAppInstUtilGetInstallProgressInfo() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

}
