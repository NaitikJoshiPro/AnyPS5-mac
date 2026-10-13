#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALVMTRACE_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALVMTRACE_H

// Debug aid (APS5_METAL_TRACE): the VM object and share mode of the page holding `address`, to tell
// a copy-on-write page apart from a range the guest arena remapped under a no-copy MTLBuffer.

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>

namespace AgcDriver::Metal {

inline std::string VmRegionInfo(std::uint64_t address) {
    mach_vm_address_t region = address;
    mach_vm_size_t size = 0;
    vm_region_top_info_data_t top {};
    mach_msg_type_number_t count = VM_REGION_TOP_INFO_COUNT;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_TOP_INFO, reinterpret_cast<vm_region_info_t>(&top), &count, &object) != KERN_SUCCESS) return "unmapped";
    region = address;
    vm_region_extended_info_data_t extended {};
    count = VM_REGION_EXTENDED_INFO_COUNT;
    mach_vm_region(mach_task_self(), &region, &size, VM_REGION_EXTENDED_INFO, reinterpret_cast<vm_region_info_t>(&extended), &count, &object);
    char text[160];
    std::snprintf(text, sizeof(text), "region 0x%llx+0x%llx object 0x%x share %d (COW %d private %d shared %d) refs %u", static_cast<unsigned long long>(region), static_cast<unsigned long long>(size), top.obj_id, static_cast<int>(extended.share_mode), SM_COW, SM_PRIVATE, SM_SHARED, top.ref_count);
    return text;
}

// Debug aid (APS5_METAL_GDS_WATCH=1): write-protects one 16 KiB page and prints a backtrace of every
// CPU store to it (the page is made writable again after the first one, which then retries).
namespace WatchDetail {
inline std::atomic<std::uint64_t> page{0};
inline struct sigaction previousSegv {};
inline struct sigaction previousBus {};
inline void Handler(int signal, siginfo_t* info, void* context) {
    const auto address = reinterpret_cast<std::uint64_t>(info->si_addr);
    const auto watched = page.load();
    if (watched != 0 && address >= watched && address < watched + 0x4000u) {
        static const char text[] = "[metal] GDS watch: CPU store to the watched page, backtrace:\n";
        write(2, text, sizeof(text) - 1);
        void* frames[32];
        backtrace_symbols_fd(frames, backtrace(frames, 32), 2);
        mprotect(reinterpret_cast<void*>(watched), 0x4000u, PROT_READ | PROT_WRITE);
        page.store(0);
        return;
    }
    const auto& previous = signal == SIGBUS ? previousBus : previousSegv;
    if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) previous.sa_sigaction(signal, info, context);
    else if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN && previous.sa_handler != nullptr) previous.sa_handler(signal);
    else ::signal(signal, SIG_DFL);
}
}

inline void WatchWrites(std::uint64_t address) {
    static const bool installed = [] {
        struct sigaction action {};
        action.sa_sigaction = WatchDetail::Handler;
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&action.sa_mask);
        sigaction(SIGSEGV, &action, &WatchDetail::previousSegv);
        sigaction(SIGBUS, &action, &WatchDetail::previousBus);
        return true;
    }();
    static_cast<void>(installed);
    const auto page = address & ~std::uint64_t{0x3fff};
    WatchDetail::page.store(page);
    mprotect(reinterpret_cast<void*>(page), 0x4000u, PROT_READ);
}

// Wired/resident counts of the region holding `address` (VM_REGION_SUBMAP_INFO_64).
inline std::string VmWiredInfo(std::uint64_t address) {
    mach_vm_address_t region = address;
    mach_vm_size_t size = 0;
    natural_t depth = 0;
    vm_region_submap_info_data_64_t info {};
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    if (mach_vm_region_recurse(mach_task_self(), &region, &size, &depth, reinterpret_cast<vm_region_recurse_info_t>(&info), &count) != KERN_SUCCESS) return "unmapped";
    char text[200];
    std::snprintf(text, sizeof(text), "region 0x%llx+0x%llx object 0x%x wired %u resident %u dirty %u shadow-depth %u prot %d/%d", static_cast<unsigned long long>(region), static_cast<unsigned long long>(size), info.object_id, info.user_wired_count, info.pages_resident, info.pages_dirtied, static_cast<unsigned>(info.shadow_depth), info.protection, info.max_protection);
    return text;
}

}

#endif
