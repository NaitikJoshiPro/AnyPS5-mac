// Fiber objects and save-data memory buffers in displaced guest memory (D31, D33): a fiber object is read and
// written in place on every call (the state atomic included) and save-data memory copies whole guest buffers, so
// both must go through host views, not the fault path. Run at the GPU carveout (0x4000000000) and the shared region
// tail (0x400000000); each case requires the right result and no redirected fault. Fiber switches need the FEXCore
// bridge (fex fixtures); here the native exports are covered: initialize (with the context size check), get info,
// rename, opt-param initialize, finalize. AvPlayer's displaced buffers are covered by guest_avplayer.
#include "SceTypes.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
using GuestFiberEntry = void (APS5_VABI*)(std::uint64_t, std::uint64_t);
int32_t APS5_VABI _sceFiberInitializeImpl_nid_postfix(FiberObject*, const char*, GuestFiberEntry, uint64_t, void*, uint64_t, const void*, uint32_t);
int32_t APS5_VABI sceFiberFinalize(FiberObject*);
int32_t APS5_VABI sceFiberGetInfo(FiberObject*, FiberInfo*);
int32_t APS5_VABI sceFiberRename(FiberObject*, const char*);
int32_t APS5_VABI sceFiberOptParamInitialize(FiberOptParam*);
int32_t APS5_VABI sceFiberStartContextSizeCheck(uint32_t);
int32_t APS5_VABI sceFiberStopContextSizeCheck(void);
int APS5_VABI sceSaveDataInitialize3(const void*);
int APS5_VABI sceSaveDataTerminate();
int APS5_VABI sceSaveDataSetupSaveDataMemory2(const SaveDataMemorySetup2*, SaveDataMemorySetupResult*);
int APS5_VABI sceSaveDataSetSaveDataMemory2(const SaveDataMemorySet2*);
int APS5_VABI sceSaveDataGetSaveDataMemory2(SaveDataMemoryGet2*);
}

namespace {

constexpr std::uintptr_t Base = 0x4000000000ull;
constexpr std::uintptr_t TailBase = 0x400000000ull;
constexpr std::size_t Bytes = 0x200000;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t Faults() {
    std::uint64_t faults = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, nullptr);
    return faults;
}

template <typename T>
T* Guest(std::uintptr_t address) {
    return reinterpret_cast<T*>(address);
}

template <typename T>
T* Host(std::uintptr_t address) {
    return reinterpret_cast<T*>(GuestArena::HostAddress(address));
}

void APS5_VABI Entry(std::uint64_t, std::uint64_t) {}

void Fibers(std::uintptr_t base, const char* where) {
    auto* object = Guest<FiberObject>(base);
    const auto name = base + 0x200, info = base + 0x300, opt = base + 0x400, rename = base + 0x500, context = base + 0x10000;
    std::strcpy(Host<char>(name), "displaced fiber");
    std::strcpy(Host<char>(rename), "renamed fiber");
    const auto before = Faults();
    Require(_sceFiberInitializeImpl_nid_postfix(object, Guest<char>(name), Entry, 7, Guest<void>(context), 0x4000, nullptr, 0) == 0, std::string(where) + ": initialize");
    Host<FiberInfo>(info)->size = sizeof(FiberInfo);
    Require(sceFiberGetInfo(object, Guest<FiberInfo>(info)) == 0, std::string(where) + ": get info");
    const auto* got = Host<FiberInfo>(info);
    Require(got->arg_on_initialize == 7 && got->addr_context == Guest<void>(context) && got->size_context == 0x4000 && std::strcmp(got->name, "displaced fiber") == 0,
        std::string(where) + ": info fields");
    Require(got->size_context_margin != static_cast<std::uint64_t>(-1), std::string(where) + ": context size check not reported");
    Require(sceFiberRename(object, Guest<char>(rename)) == 0 && sceFiberGetInfo(object, Guest<FiberInfo>(info)) == 0 && std::strcmp(got->name, "renamed fiber") == 0,
        std::string(where) + ": rename");
    Host<std::uint32_t>(opt)[0] = 0xffffffffu;
    Require(sceFiberOptParamInitialize(Guest<FiberOptParam>(opt)) == 0 && Host<std::uint32_t>(opt)[0] == 0, std::string(where) + ": opt param");
    Require(sceFiberFinalize(object) == 0, std::string(where) + ": finalize");
    Require(sceFiberGetInfo(object, Guest<FiberInfo>(info)) != 0, std::string(where) + ": a finalized fiber still answers");
    const auto faults = Faults() - before;
    std::printf("%s fibers: %llu faults\n", where, static_cast<unsigned long long>(faults));
    Require(faults == 0, std::string(where) + ": the fiber exports faulted on displaced guest memory");
}

void SaveDataMemory(std::uintptr_t base, int user, const char* where) {
    const auto initParam = base + 0x1000, setParam = base + 0x2000, getParam = base + 0x3000, setBuffer = base + 0x4000, getBuffer = base + 0x8000;
    std::memset(Host<void>(initParam), 0, sizeof(SaveDataParam));
    Host<SaveDataParam>(initParam)->user_param = 11;
    std::memset(Host<void>(setParam), 0, sizeof(SaveDataParam));
    Host<SaveDataParam>(setParam)->user_param = 22;
    for (std::size_t i = 0; i < 0x1000; ++i) Host<std::uint8_t>(setBuffer)[i] = static_cast<std::uint8_t>(i * 7 + user);
    std::memset(Host<void>(getBuffer), 0, 0x1000);
    const auto before = Faults();
    SaveDataMemorySetup2 setup{};
    setup.user_id = user;
    setup.memory_size = 0x2000;
    setup.option = 1;
    setup.init_param = Guest<SaveDataParam>(initParam);
    SaveDataMemorySetupResult result{};
    Require(sceSaveDataSetupSaveDataMemory2(&setup, &result) == 0, std::string(where) + ": setup");
    SaveDataMemoryData data{Guest<void>(setBuffer), 0x1000, 0x100};
    SaveDataMemorySet2 set{};
    set.user_id = user;
    set.data = &data;
    set.data_num = 1;
    set.param = Guest<SaveDataParam>(setParam);
    Require(sceSaveDataSetSaveDataMemory2(&set) == 0, std::string(where) + ": set");
    SaveDataMemoryData read{Guest<void>(getBuffer), 0x1000, 0x100};
    SaveDataMemoryGet2 get{};
    get.user_id = user;
    get.data = &read;
    get.param = Guest<SaveDataParam>(getParam);
    Require(sceSaveDataGetSaveDataMemory2(&get) == 0, std::string(where) + ": get");
    const auto faults = Faults() - before;
    std::printf("%s save-data memory: %llu faults\n", where, static_cast<unsigned long long>(faults));
    Require(std::memcmp(Host<void>(getBuffer), Host<void>(setBuffer), 0x1000) == 0, std::string(where) + ": the memory read back differs");
    Require(Host<SaveDataParam>(getParam)->user_param == 22, std::string(where) + ": the param read back differs");
    Require(faults == 0, std::string(where) + ": save-data memory faulted on displaced guest memory");
}

void Map(std::uintptr_t base) {
    std::int64_t physical = 0;
    Require(sceKernelAllocateDirectMemory(0, 0x100000000ll, Bytes, 0x200000, 3, &physical) == 0, "allocate direct memory");
    void* fixed = reinterpret_cast<void*>(base);
    Require(sceKernelMapDirectMemory(&fixed, Bytes, 0x33, 0x10, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(base), "map displaced memory");
}

}

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("guest displaced fiber/media: no displaced window on this host, skipped");
    return 0;
#else
    try {
        Require(GuestArena::GuestDisplacedActive_nid_no_patch(), "the displaced window is not active on macOS arm64");
        Map(Base);
        Map(TailBase);
        Require(sceFiberStartContextSizeCheck(0) == 0, "start context size check");
        Fibers(Base, "carveout");
        Fibers(TailBase, "shared region tail");
        Require(sceFiberStopContextSizeCheck() == 0, "stop context size check");
        const auto previous = std::filesystem::current_path();
        const auto root = std::filesystem::temp_directory_path() /
            ("anyps5-displaced-savedata-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        std::filesystem::current_path(root);
        Require(sceSaveDataInitialize3(nullptr) == 0, "save data initialize");
        SaveDataMemory(Base, 8801, "carveout");
        SaveDataMemory(TailBase, 8802, "shared region tail");
        sceSaveDataTerminate();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(root);
        std::puts("guest displaced fiber/media: all checks passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "guest displaced fiber/media: %s\n", error.what());
        return 1;
    }
#endif
}
