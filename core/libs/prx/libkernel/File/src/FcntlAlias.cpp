// libkernel's _fcntl: the FreeBSD libc name of fcntl. The oops-sdk's fcntl wrapper calls it
// (oops-apps common/posix/hosted.c) for everything but a socket's F_SETFL. It forwards to fcntl
// with its one optional argument, as sceKernelFcntl does.
#include "SceTypes.hpp"
#include "prx/libc/include/VarArgsAbi.hpp"
#include "prx/libc/include/WindowsFormatting.hpp"
#include <cstdarg>
#include <cstdint>

extern "C" int APS5_VABI fcntl_nid_postfix(int descriptor, int command, ...);

extern "C" int APS5_VABI _fcntl_nid_postfix(int descriptor, int command, ...) {
#if !APS5_GUEST_VA_LIST_IS_HOST
    APS5_VA_BEGIN(command);
    const auto argument = LibcDetail::FormatArguments(args).Next<std::int64_t>();
    APS5_VA_END();
#else
    std::va_list arguments;
    va_start(arguments, command);
    const auto argument = va_arg(arguments, std::int64_t);
    va_end(arguments);
#endif
    return fcntl_nid_postfix(descriptor, command, argument);
}
