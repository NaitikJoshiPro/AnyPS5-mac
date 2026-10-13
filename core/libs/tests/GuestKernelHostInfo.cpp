// link, scePthreadGetcpuclockid, sceKernelGetHwModelName/HwSerialNumber, sceKernelGetCurrentFanDuty and
// the macOS thermal-pressure mapping.
#include "SceTypes.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

extern "C" {
int APS5_VABI link_nid_postfix(const char* existing, const char* created);
int* APS5_VABI __error_nid_postfix();
Pthread APS5_VABI scePthreadSelf();
int APS5_VABI scePthreadCreate(Pthread* thread, const PthreadAttr* attr, PthreadEntry entry, void* arg, const char* name);
int APS5_VABI scePthreadJoin(Pthread thread, void** retval);
int APS5_VABI scePthreadGetcpuclockid(Pthread thread, int* clock);
int APS5_VABI sceKernelGetHwModelName(char* buffer);
int APS5_VABI sceKernelGetHwSerialNumber(char* buffer);
int APS5_VABI sceKernelGetCurrentFanDuty(void* first, void* second);
int APS5_VABI sceKernelGetCpuTemperature(int* celsius);
#ifdef __APPLE__
int ThermalPressureCelsius_nid_no_patch(std::uint64_t level);
#endif
}

static constexpr int SCE_KERNEL_ERROR_ESRCH = static_cast<int>(0x80020003u);
static constexpr int SCE_KERNEL_ERROR_EFAULT = static_cast<int>(0x8002000Eu);
static constexpr int SCE_KERNEL_ERROR_EINVAL = static_cast<int>(0x80020016u);

static void Require(bool value, const char* what) {
    if (!value) {
        std::fprintf(stderr, "kernel host info check failed: %s\n", what);
        std::abort();
    }
}

static int Error() { return *__error_nid_postfix(); }

static void* APS5_VABI Idle(void*) { return nullptr; }

int main() {
    const auto root = std::filesystem::path("anyps5-kernel-host-info");
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const std::string original = (root / "original.txt").string();
    const std::string linked = (root / "linked.txt").string();
    std::ofstream(original) << "payload";
    Require(link_nid_postfix(original.c_str(), linked.c_str()) == 0, "link");
    struct stat status{};
    Require(::stat(linked.c_str(), &status) == 0 && status.st_nlink == 2, "hard link count");
    std::string text;
    std::ifstream(linked) >> text;
    Require(text == "payload", "linked content");
    Require(link_nid_postfix(original.c_str(), linked.c_str()) == -1 && Error() == 17, "link onto an existing name");
    Require(link_nid_postfix((root / "absent").string().c_str(), (root / "x").string().c_str()) == -1 && Error() == 2, "link from a missing file");
    Require(link_nid_postfix(nullptr, linked.c_str()) == -1 && Error() == 14, "link null");
    Require(link_nid_postfix("", linked.c_str()) == -1 && Error() == 2, "link empty");
    std::filesystem::remove_all(root);

    int clock = -1;
    Require(scePthreadGetcpuclockid(scePthreadSelf(), &clock) == 0 && clock == 14, "own cpu clock");
    Require(scePthreadGetcpuclockid(nullptr, &clock) == SCE_KERNEL_ERROR_ESRCH, "null thread");
    Require(scePthreadGetcpuclockid(scePthreadSelf(), nullptr) == SCE_KERNEL_ERROR_EINVAL, "null clock");
    Pthread other = nullptr;
    Require(scePthreadCreate(&other, nullptr, Idle, nullptr, "other") == 0, "create");
    bool refused = false;
    try { scePthreadGetcpuclockid(other, &clock); } catch (const std::runtime_error&) { refused = true; }
    Require(refused, "another thread's clock throws");
    Require(scePthreadJoin(other, nullptr) == 0, "join");

    char model[1024];
    std::memset(model, 0x7f, sizeof(model));
    Require(sceKernelGetHwModelName(model) == 0 && std::strlen(model) > 0 && std::strlen(model) < 32, "model name");
    Require(model[std::strlen(model) + 1] == 0x7f || std::strlen(model) == 31, "model name writes at most its string");
    char serial[1024] = {};
    Require(sceKernelGetHwSerialNumber(serial) == 0 && std::strcmp(serial, "APS5-HOST-000000") == 0, "serial placeholder");
    Require(sceKernelGetHwModelName(nullptr) == SCE_KERNEL_ERROR_EFAULT && sceKernelGetHwSerialNumber(nullptr) == SCE_KERNEL_ERROR_EFAULT, "null buffers");

    struct { int first = -1; int second = -1; std::uint64_t guard = 0x1122334455667788ull; } fan;
    Require(sceKernelGetCurrentFanDuty(&fan.first, &fan.second) == 0, "fan duty");
    Require((fan.first & 0xffff) == 0 && fan.second == 0 && fan.guard == 0x1122334455667788ull, "fan duty writes 2 + 4 bytes");
    Require(sceKernelGetCurrentFanDuty(nullptr, nullptr) == SCE_KERNEL_ERROR_EFAULT, "fan duty null");

#ifdef __APPLE__
    Require(ThermalPressureCelsius_nid_no_patch(0) == 50 && ThermalPressureCelsius_nid_no_patch(1) == 70 &&
            ThermalPressureCelsius_nid_no_patch(2) == 85 && ThermalPressureCelsius_nid_no_patch(3) == 95 &&
            ThermalPressureCelsius_nid_no_patch(4) == 105 && ThermalPressureCelsius_nid_no_patch(9) == -1, "pressure mapping");
    int celsius = -1;
    Require(sceKernelGetCpuTemperature(&celsius) == 0 && celsius >= 50 && celsius <= 105, "cpu temperature from thermal pressure");
#endif
    return 0;
}
