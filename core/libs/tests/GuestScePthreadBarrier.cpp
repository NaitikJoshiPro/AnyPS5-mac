#include "SceTypes.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>

struct GuestBarrier;
struct GuestBarrierattr;

extern "C" {
int APS5_VABI scePthreadCreate(Pthread* thread, const PthreadAttr* attr, PthreadEntry entry, void* arg, const char* name);
int APS5_VABI scePthreadJoin(Pthread thread, void** retval);
int APS5_VABI scePthreadBarrierattrInit(GuestBarrierattr** attr);
int APS5_VABI scePthreadBarrierattrDestroy(GuestBarrierattr** attr);
int APS5_VABI scePthreadBarrierattrGetpshared(GuestBarrierattr* const* attr, int* pshared);
int APS5_VABI scePthreadBarrierattrSetpshared(GuestBarrierattr** attr, int pshared);
int APS5_VABI scePthreadBarrierInit(GuestBarrier** barrier, GuestBarrierattr* const* attr, unsigned count, const char* name);
int APS5_VABI scePthreadBarrierWait(GuestBarrier** barrier);
int APS5_VABI scePthreadBarrierDestroy(GuestBarrier** barrier);
}

static constexpr int SCE_KERNEL_ERROR_EINVAL = static_cast<int>(0x80020016u);
static constexpr int BARRIER_SERIAL_THREAD = -1;
static constexpr unsigned WAITERS = 4;
static constexpr int ROUNDS = 25;

static void Require(bool value, int line = __builtin_LINE()) {
    if (value) return;
    std::fprintf(stderr, "scePthreadBarrier check failed at line %d\n", line);
    std::abort();
}

struct Context {
    GuestBarrier* barrier = nullptr;
    std::atomic<int> serial{0};
    std::atomic<int> released{0};
};

// Every waiter passes the barrier ROUNDS times; each round has exactly one serial thread.
static void* APS5_VABI Waiter(void* arg) {
    auto& context = *static_cast<Context*>(arg);
    for (int round = 0; round < ROUNDS; ++round) {
        const int result = scePthreadBarrierWait(&context.barrier);
        if (result == BARRIER_SERIAL_THREAD) ++context.serial;
        else if (result == 0) ++context.released;
        else std::abort();
    }
    return nullptr;
}

int main() {
    GuestBarrierattr* missing = nullptr;
    Require(scePthreadBarrierattrInit(nullptr) == SCE_KERNEL_ERROR_EINVAL);
    Require(scePthreadBarrierattrDestroy(&missing) == SCE_KERNEL_ERROR_EINVAL);
    Require(scePthreadBarrierattrSetpshared(&missing, 0) == SCE_KERNEL_ERROR_EINVAL);

    GuestBarrierattr* attr = nullptr;
    int pshared = -1;
    Require(scePthreadBarrierattrInit(&attr) == 0 && attr != nullptr);
    Require(scePthreadBarrierattrGetpshared(&attr, &pshared) == 0 && pshared == 0);
    Require(scePthreadBarrierattrSetpshared(&attr, 2) == SCE_KERNEL_ERROR_EINVAL);
    // The PS5 accepts only PTHREAD_PROCESS_PRIVATE (upstream db504f6b9c, firmware 12.70).
    Require(scePthreadBarrierattrSetpshared(&attr, 1) == SCE_KERNEL_ERROR_EINVAL);
    Require(scePthreadBarrierattrSetpshared(&attr, 0) == 0);
    pshared = -1;
    Require(scePthreadBarrierattrGetpshared(&attr, &pshared) == 0 && pshared == 0);

    GuestBarrier* invalid = nullptr;
    Require(scePthreadBarrierInit(nullptr, &attr, 1, "null") == SCE_KERNEL_ERROR_EINVAL);
    Require(scePthreadBarrierInit(&invalid, &attr, 0, "zero") == SCE_KERNEL_ERROR_EINVAL && invalid == nullptr);
    Require(scePthreadBarrierWait(&invalid) == SCE_KERNEL_ERROR_EINVAL);
    Require(scePthreadBarrierDestroy(&invalid) == SCE_KERNEL_ERROR_EINVAL);

    Context context;
    Require(scePthreadBarrierInit(&context.barrier, &attr, WAITERS, "test barrier") == 0 && context.barrier != nullptr);
    Require(scePthreadBarrierattrDestroy(&attr) == 0 && attr == nullptr);
    std::array<Pthread, WAITERS> threads{};
    for (auto& thread : threads) Require(scePthreadCreate(&thread, nullptr, Waiter, &context, nullptr) == 0);
    for (auto& thread : threads) Require(scePthreadJoin(thread, nullptr) == 0);
    Require(context.serial == ROUNDS && context.released == ROUNDS * (static_cast<int>(WAITERS) - 1));
    Require(scePthreadBarrierDestroy(&context.barrier) == 0 && context.barrier == nullptr);

    GuestBarrier* single = nullptr;
    Require(scePthreadBarrierInit(&single, nullptr, 1, "single") == 0);
    Require(scePthreadBarrierWait(&single) == BARRIER_SERIAL_THREAD);
    Require(scePthreadBarrierDestroy(&single) == 0);
}
