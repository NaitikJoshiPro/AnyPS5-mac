// Small, hot status and out-parameter exports with their structures in displaced guest memory (D31, D33): event
// queue waits (event array, *out, *timeout) and the event accessors, clock_gettime/gettimeofday, event flag
// poll/wait, VideoOut event accessors, Pad information, AudioOut port queries and outputs, AudioOut2/UserService
// out-parameters. Titles that keep their heap displaced (ASTRO BOT) call these every frame; through the guest
// address each load/store of native code takes two signals. Each export must use a host view: the checks require
// the right result and no redirected fault. Run at the shared region tail (0x400000000) and the GPU carveout
// (0x4000000000). Not covered: VideoOut flip/vblank/output/resolution status (need an open video port), scePadRead
// and scePadReadState (need the pad service running), Audio3d (needs an initialised port).
#include "SceTypes.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
int APS5_VABI sceKernelCreateEqueue(KernelEqueue*, const char*);
int APS5_VABI sceKernelDeleteEqueue(KernelEqueue);
int APS5_VABI sceKernelAddUserEvent(KernelEqueue, int);
int APS5_VABI sceKernelTriggerUserEvent(KernelEqueue, int, void*);
int APS5_VABI sceKernelWaitEqueue(KernelEqueue, KernelEvent*, int, int*, const KernelUseconds*);
intptr_t APS5_VABI sceKernelGetEventData(const KernelEvent*);
uintptr_t APS5_VABI sceKernelGetEventId(const KernelEvent*);
int APS5_VABI sceKernelGetEventFilter(const KernelEvent*);
void* APS5_VABI sceKernelGetEventUserData(const KernelEvent*);
int APS5_VABI sceKernelClockGettime(KernelClockid, KernelTimespec*);
int APS5_VABI sceKernelGettimeofday(KernelTimeval*);
int APS5_VABI sceKernelCreateEventFlag(KernelEventFlag*, const char*, std::uint32_t, std::uint64_t, const void*);
int APS5_VABI sceKernelDeleteEventFlag(KernelEventFlag);
int APS5_VABI sceKernelPollEventFlag(KernelEventFlag, std::uint64_t, std::uint32_t, std::uint64_t*);
int APS5_VABI sceKernelWaitEventFlag(KernelEventFlag, std::uint64_t, std::uint32_t, std::uint64_t*, KernelUseconds*);
int APS5_VABI sceVideoOutGetEventId(const KernelEvent*);
int APS5_VABI sceVideoOutGetEventData(const KernelEvent*, std::int64_t*);
int APS5_VABI sceVideoOutGetEventCount(const KernelEvent*);
int APS5_VABI scePadGetControllerInformation(int, void*);
int APS5_VABI scePadIsRemoteController(int, bool*);
int APS5_VABI sceAudioOutOpen(int, int, int, std::uint32_t, std::uint32_t, std::uint32_t);
int APS5_VABI sceAudioOutClose(int);
int APS5_VABI sceAudioOutOutputs(AudioOutOutputParam*, std::uint32_t);
int APS5_VABI sceAudioOutGetPortState(int, AudioOutPortState*);
int APS5_VABI sceAudioOutGetLastOutputTime(int, std::uint64_t*);
int APS5_VABI sceUserServiceGetInitialUser(int*);
int APS5_VABI sceUserServiceGetUserName(int, char*, std::size_t);
int APS5_VABI sceUserServiceGetLoginUserIdList(void*);
}

namespace {

constexpr std::uintptr_t TailBase = 0x400000000ull;
constexpr std::uintptr_t CarveoutBase = 0x4000000000ull;
constexpr std::size_t Bytes = 0x200000;
constexpr int CpuGpuReadWrite = 0x33;
constexpr int MapFixed = 0x10;
constexpr int InitialUser = 0x10000000;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t Faults() {
    std::uint64_t faults = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, nullptr);
    return faults;
}

// Runs one export on displaced structures and requires that it took no redirected fault.
template <typename F>
void NoFaults(std::uintptr_t base, const char* what, F&& call) {
    const auto before = Faults();
    call();
    const auto faults = Faults() - before;
    char at[64];
    std::snprintf(at, sizeof(at), " at 0x%llx", static_cast<unsigned long long>(base));
    Require(faults == 0, std::string(what) + at + ": " + std::to_string(faults) + " faults through the guest address");
}

// Guest pointers into the displaced window; values are checked through the alias.
template <typename T>
T* At(std::uintptr_t base, std::size_t offset) {
    return reinterpret_cast<T*>(base + offset);
}

template <typename T>
T& Host(T* guest) {
    return *GuestArena::HostPointer(guest);
}

void Kernel(std::uintptr_t base) {
    KernelEqueue queue = 0;
    Require(sceKernelCreateEqueue(&queue, "displaced") == 0 && sceKernelAddUserEvent(queue, 7) == 0, "equeue setup");
    auto* events = At<KernelEvent>(base, 0x100);
    auto* out = At<int>(base, 0x300);
    auto* timeout = At<KernelUseconds>(base, 0x308);
    int tag = 0;
    Require(sceKernelTriggerUserEvent(queue, 7, &tag) == 0, "trigger user event");
    Host(timeout) = 1000;
    NoFaults(base, "sceKernelWaitEqueue", [&] { Require(sceKernelWaitEqueue(queue, events, 4, out, timeout) == 0, "sceKernelWaitEqueue"); });
    Require(Host(out) == 1 && Host(events).ident == 7 && Host(events).udata == &tag, "sceKernelWaitEqueue: wrong event");
    NoFaults(base, "sceKernelGetEvent*", [&] {
        Require(sceKernelGetEventId(events) == 7 && sceKernelGetEventUserData(events) == &tag && sceKernelGetEventFilter(events) == Host(events).filter,
            "event accessors");
        (void)sceKernelGetEventData(events);
    });
    Require(sceKernelDeleteEqueue(queue) == 0, "delete equeue");

    auto* time = At<KernelTimespec>(base, 0x400);
    Host(time) = {};
    NoFaults(base, "sceKernelClockGettime", [&] { Require(sceKernelClockGettime(4, time) == 0, "sceKernelClockGettime"); });
    Require(Host(time).tv_sec != 0 || Host(time).tv_nsec != 0, "sceKernelClockGettime wrote nothing");
    auto* day = At<KernelTimeval>(base, 0x420);
    Host(day) = {};
    NoFaults(base, "sceKernelGettimeofday", [&] { Require(sceKernelGettimeofday(day) == 0, "sceKernelGettimeofday"); });
    Require(Host(day).tv_sec > 1600000000, "sceKernelGettimeofday wrote no time");

    KernelEventFlag flag = nullptr;
    Require(sceKernelCreateEventFlag(&flag, "displaced", 0, 0x5, nullptr) == 0, "create event flag");
    auto* result = At<std::uint64_t>(base, 0x440);
    auto* flagTimeout = At<KernelUseconds>(base, 0x448);
    NoFaults(base, "sceKernelPollEventFlag", [&] { Require(sceKernelPollEventFlag(flag, 0x4, 0x1, result) == 0, "sceKernelPollEventFlag"); });
    Require(Host(result) == 0x5, "sceKernelPollEventFlag: wrong pattern");
    Host(flagTimeout) = 100;
    NoFaults(base, "sceKernelWaitEventFlag", [&] { Require(sceKernelWaitEventFlag(flag, 0x8, 0x1, result, flagTimeout) != 0, "sceKernelWaitEventFlag did not time out"); });
    Require(Host(result) == 0x5 && Host(flagTimeout) == 0, "sceKernelWaitEventFlag: wrong pattern or timeout");
    Require(sceKernelDeleteEventFlag(flag) == 0, "delete event flag");
}

void VideoOutEvents(std::uintptr_t base) {
    auto* event = At<KernelEvent>(base, 0x500);
    Host(event) = {};
    Host(event).ident = 0; // flip
    Host(event).filter = -13; // EVFILT_VIDEO_OUT
    Host(event).data = static_cast<std::intptr_t>((0x1234ull << 16) | (3ull << 12));
    auto* data = At<std::int64_t>(base, 0x540);
    NoFaults(base, "sceVideoOutGetEvent*", [&] {
        Require(sceVideoOutGetEventId(event) == 0 && sceVideoOutGetEventCount(event) == 3, "VideoOut event id/count");
        Require(sceVideoOutGetEventData(event, data) == 0, "sceVideoOutGetEventData");
    });
    Require(Host(data) == 0x1234, "sceVideoOutGetEventData: wrong data");
}

void Pad(std::uintptr_t base) {
    auto* info = At<std::uint8_t>(base, 0x600);
    std::memset(GuestArena::HostPointer(info), 0xa5, 0x100);
    NoFaults(base, "scePadGetControllerInformation", [&] { Require(scePadGetControllerInformation(1, info) == 0, "scePadGetControllerInformation"); });
    Require(GuestArena::HostPointer(info)[0] != 0xa5, "scePadGetControllerInformation wrote nothing");
    auto* remote = At<bool>(base, 0x700);
    Host(remote) = true;
    NoFaults(base, "scePadIsRemoteController", [&] { Require(scePadIsRemoteController(1, remote) == 0, "scePadIsRemoteController"); });
    Require(!Host(remote), "scePadIsRemoteController wrote nothing");
}

void Audio(std::uintptr_t base) {
    const int port = sceAudioOutOpen(InitialUser, 0, 0, 256, 48000, 0);
    Require(port > 0, "sceAudioOutOpen");
    auto* time = At<std::uint64_t>(base, 0x800);
    Host(time) = ~0ull;
    NoFaults(base, "sceAudioOutGetLastOutputTime", [&] { Require(sceAudioOutGetLastOutputTime(port, time) == 0, "sceAudioOutGetLastOutputTime"); });
    Require(Host(time) != ~0ull, "sceAudioOutGetLastOutputTime wrote nothing");
    auto* state = At<AudioOutPortState>(base, 0x840);
    std::memset(GuestArena::HostPointer(state), 0xa5, sizeof(AudioOutPortState));
    NoFaults(base, "sceAudioOutGetPortState", [&] { Require(sceAudioOutGetPortState(port, state) == 0, "sceAudioOutGetPortState"); });
    Require(Host(state).volume == 127, "sceAudioOutGetPortState: wrong volume");
    auto* param = At<AudioOutOutputParam>(base, 0x900);
    Host(param) = {};
    Host(param).handle = port;
    Host(param).ptr = nullptr; // no PCM: the parameter array is what is read
    NoFaults(base, "sceAudioOutOutputs", [&] { Require(sceAudioOutOutputs(param, 1) == 256, "sceAudioOutOutputs"); });
    Require(sceAudioOutClose(port) == 0, "sceAudioOutClose");
}

void UserService(std::uintptr_t base) {
    auto* user = At<int>(base, 0xa00);
    auto* name = At<char>(base, 0xa40);
    auto* list = At<std::int32_t>(base, 0xb00);
    NoFaults(base, "sceUserServiceGet*", [&] {
        Require(sceUserServiceGetInitialUser(user) == 0, "sceUserServiceGetInitialUser");
        Require(sceUserServiceGetUserName(InitialUser, name, 32) == 0, "sceUserServiceGetUserName");
        Require(sceUserServiceGetLoginUserIdList(list) == 0, "sceUserServiceGetLoginUserIdList");
    });
    Require(Host(user) == InitialUser && GuestArena::HostPointer(name)[0] != 0 && Host(list) == InitialUser, "UserService: wrong values");
}

}

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("guest displaced status: no displaced window on this host, skipped");
    return 0;
#else
    try {
        Require(GuestArena::GuestDisplacedActive_nid_no_patch(), "the displaced window is not active on macOS arm64");
        std::int64_t physical = 0;
        Require(sceKernelAllocateDirectMemory(0, 0x100000000ll, Bytes, 0x200000, 3, &physical) == 0, "allocate direct memory");
        for (const std::uintptr_t base : {TailBase, CarveoutBase}) {
            void* fixed = reinterpret_cast<void*>(base);
            Require(sceKernelMapDirectMemory(&fixed, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(base),
                "the fixed mapping failed");
            Kernel(base);
            VideoOutEvents(base);
            Pad(base);
            Audio(base);
            UserService(base);
            Require(sceKernelMunmap(fixed, Bytes) == 0, "munmap");
            std::printf("guest displaced status at 0x%llx passed\n", static_cast<unsigned long long>(base));
        }
        Require(sceKernelReleaseDirectMemory(physical, Bytes) == 0, "release direct memory");
        std::puts("guest displaced status tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "guest displaced status: %s\n", error.what());
        return 1;
    }
#endif
}
