// scePthreadGetcpuclockid: the CPU-time clock of a thread. For the calling thread that is
// CLOCK_THREAD_CPUTIME_ID (guest clock 14, which clock_gettime already serves), as FreeBSD's
// pthread_getcpuclockid returns for curthread. Another thread's clock would need a per-thread clock id
// that clock_gettime resolves to that thread; that is not implemented and throws.
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <stdexcept>

namespace {

constexpr int GUEST_CLOCK_THREAD_CPUTIME_ID = 14;
constexpr int SCE_KERNEL_ERROR_ESRCH = static_cast<int>(0x80020003u);
constexpr int SCE_KERNEL_ERROR_EINVAL = static_cast<int>(0x80020016u);

}

extern "C" Pthread APS5_VABI scePthreadSelf();

extern "C" int APS5_VABI scePthreadGetcpuclockid(Pthread thread, int* clock) {
    if (thread == nullptr) return SCE_KERNEL_ERROR_ESRCH;
    if (clock == nullptr) return SCE_KERNEL_ERROR_EINVAL;
    if (thread != scePthreadSelf())
        throw std::runtime_error("scePthreadGetcpuclockid: the CPU-time clock of another thread is not implemented");
    *clock = GUEST_CLOCK_THREAD_CPUTIME_ID;
    return 0;
}
