#include "prx/libc/include/GuestStacks.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libc/include/VarArgsAbi.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libkernel/System/include/UserContext.hpp"
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#if !APS5_GUEST_VA_LIST_IS_HOST
#include "prx/libc/include/WindowsFormatting.hpp"
#endif

extern "C" int APS5_VABI setcontext_nid_postfix(const UserUcontext* context);

namespace {

using ContextFunction = void (APS5_VABI*)(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);

constexpr int MaximumArguments = 6;
constexpr std::uint64_t MinimumStackSize = 2048;

}

extern "C" {

[[noreturn]] void APS5_VABI Aps5MakecontextStart_nid_no_patch(UserUcontext* context, ContextFunction function, const std::uint64_t* arguments) {
    function(arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
    if (context->link == nullptr) LibcExit_nid_no_patch(0);
    setcontext_nid_postfix(context->link);
    std::abort();
}

void APS5_VABI makecontext_nid_postfix(UserUcontext* context, void (APS5_VABI* start)(), int count, ...) {
    if (context == nullptr || context->mcontext.len != sizeof(UserMcontext)) return;
    if (count < 0 || count > MaximumArguments || context->stackPointer == nullptr || context->stackSize < MinimumStackSize) {
        context->mcontext.len = 0;
        return;
    }
    const auto top = (reinterpret_cast<std::uintptr_t>(context->stackPointer) + context->stackSize) & ~static_cast<std::uintptr_t>(15);
    auto* arguments = reinterpret_cast<std::uint64_t*>(top) - MaximumArguments;
    auto* stack = arguments - 1;
#if !APS5_GUEST_VA_LIST_IS_HOST
    APS5_VA_BEGIN(count);
    LibcDetail::FormatArguments list(args);
    for (int index = 0; index < count; ++index) arguments[index] = list.Next<std::uint64_t>();
    APS5_VA_END();
#else
    std::va_list list;
    va_start(list, count);
    for (int index = 0; index < count; ++index) arguments[index] = va_arg(list, std::uint64_t);
    va_end(list);
#endif
    for (int index = count; index < MaximumArguments; ++index) arguments[index] = 0;
    *stack = 0;
    auto& machine = context->mcontext;
    machine.rdi = reinterpret_cast<std::uint64_t>(context);
    machine.rsi = reinterpret_cast<std::uint64_t>(start);
    machine.rdx = reinterpret_cast<std::uint64_t>(arguments);
    machine.rbp = 0;
    machine.rbx = reinterpret_cast<std::uint64_t>(stack);
    machine.rsp = reinterpret_cast<std::uint64_t>(stack);
    machine.rip = reinterpret_cast<std::uint64_t>(&Aps5MakecontextStart_nid_no_patch);
    GuestStacks::GuestStackRegister_nid_no_patch(context->stackPointer, context->stackSize);
}

}
