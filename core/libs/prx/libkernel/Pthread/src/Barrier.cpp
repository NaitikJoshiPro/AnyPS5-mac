// scePthreadBarrier* and scePthreadBarrierattr*: the SCE forms of the POSIX barriers in
// Pthread/Posix/Barrier.cpp. They share the barrier object (a pointer-sized handle the
// guest owns) and report a POSIX error e as SCE_KERNEL_ERROR_* (0x80020000 | e), as the
// other scePthread* wrappers do. scePthreadBarrierWait returns -1
// (PTHREAD_BARRIER_SERIAL_THREAD) to exactly one thread per cycle and 0 to the others,
// like FreeBSD's _pthread_barrier_wait, which the console's libkernel wraps.
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <cstdint>

struct PthreadBarrierPrivate;
struct PthreadBarrierattrPrivate;

extern "C" {

int APS5_VABI pthread_barrierattr_init_nid_postfix(PthreadBarrierattrPrivate** attr);
int APS5_VABI pthread_barrierattr_destroy_nid_postfix(PthreadBarrierattrPrivate** attr);
int APS5_VABI pthread_barrierattr_getpshared_nid_postfix(PthreadBarrierattrPrivate* const* attr, int* pshared);
int APS5_VABI pthread_barrierattr_setpshared_nid_postfix(PthreadBarrierattrPrivate** attr, int pshared);
int APS5_VABI pthread_barrier_init_nid_postfix(PthreadBarrierPrivate** barrier, PthreadBarrierattrPrivate* const* attr, unsigned count);
int APS5_VABI pthread_barrier_wait_nid_postfix(PthreadBarrierPrivate** barrier);
int APS5_VABI pthread_barrier_destroy_nid_postfix(PthreadBarrierPrivate** barrier);

}

namespace {

constexpr int BARRIER_SERIAL_THREAD = -1;

int ToSce(int result) {
    if (result <= 0) return result;
    return static_cast<int>(0x80020000u | static_cast<std::uint32_t>(result));
}

}

extern "C" {

int APS5_VABI scePthreadBarrierattrInit(PthreadBarrierattrPrivate** attr) {
    return ToSce(pthread_barrierattr_init_nid_postfix(attr));
}

int APS5_VABI scePthreadBarrierattrDestroy(PthreadBarrierattrPrivate** attr) {
    return ToSce(pthread_barrierattr_destroy_nid_postfix(attr));
}

int APS5_VABI scePthreadBarrierattrGetpshared(PthreadBarrierattrPrivate* const* attr, int* pshared) {
    return ToSce(pthread_barrierattr_getpshared_nid_postfix(attr, pshared));
}

int APS5_VABI scePthreadBarrierattrSetpshared(PthreadBarrierattrPrivate** attr, int pshared) {
    return ToSce(pthread_barrierattr_setpshared_nid_postfix(attr, pshared));
}

// The name is a debugging label on the console; the host barrier has no use for it.
int APS5_VABI scePthreadBarrierInit(PthreadBarrierPrivate** barrier, PthreadBarrierattrPrivate* const* attr, unsigned count, const char* name) {
    (void)name;
    return ToSce(pthread_barrier_init_nid_postfix(barrier, attr, count));
}

int APS5_VABI scePthreadBarrierWait(PthreadBarrierPrivate** barrier) {
    const int result = pthread_barrier_wait_nid_postfix(barrier);
    return result == BARRIER_SERIAL_THREAD ? result : ToSce(result);
}

int APS5_VABI scePthreadBarrierDestroy(PthreadBarrierPrivate** barrier) {
    return ToSce(pthread_barrier_destroy_nid_postfix(barrier));
}

}
