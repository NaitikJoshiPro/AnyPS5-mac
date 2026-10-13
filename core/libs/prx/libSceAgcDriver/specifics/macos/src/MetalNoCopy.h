#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALNOCOPY_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALNOCOPY_H

// No-copy MTLBuffers over guest (host) pages (work package D3). A page that is read-only when
// newBufferWithBytesNoCopy wraps it is not shared with the GPU for writing: the GPU's stores land
// in pages the CPU never sees again (agc_driver_gds_append_consume: the guest write watch had armed
// the GDS page with mprotect, the shader's atomics reached the import, the CPU kept reading the
// seed). So the range is made writable while the buffer is created: armed write-watch pages through
// the watch itself (BeginHostWrite, which also counts them as written), pages the guest mapped
// read-only by a temporary mprotect that is undone afterwards. Re-arming the pages later is fine:
// the GPU mapping already holds the CPU's pages (agc_metal_noncopy_import checks both).

#import <Metal/Metal.h>
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace AgcDriver::Metal {
// Test seam (specifics/macos/tests/HostImportRetireTwin.mm, the Metal twin of tests/HostImportRetireHold.cpp,
// whose MockVulkan counts vkDestroyBuffer): while set, the probe is told each host import's buffer handle (the
// VkBuffer the import carries) as the import is destroyed, before its no-copy MTLBuffer is released. One probe at
// a time; nullptr clears it. GuestBufferMemory.mm defines it.
using ImportReleaseProbe = void (*)(const void* buffer);
void SetImportReleaseProbe(ImportReleaseProbe probe);

inline id<MTLBuffer> NewNoCopyBuffer(id<MTLDevice> device, std::uint64_t begin, std::uint64_t bytes, MTLResourceOptions options) {
    const auto end = begin + bytes;
    const bool watched = GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(begin, bytes);
    if (watched) GuestWriteWatch::GuestWriteWatchBeginHostWrite_nid_postfix(reinterpret_cast<const void*>(begin), bytes);
    // Readable pages without write access that may be given it (guest read-only mappings).
    std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>, int>> restore;
    for (std::uint64_t cursor = begin; cursor < end;) {
        mach_vm_address_t region = cursor;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info {};
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || region >= end) break;
        const auto first = std::max<std::uint64_t>(cursor, region);
        const auto last = std::min<std::uint64_t>(end, region + size);
        const bool readOnly = (info.protection & VM_PROT_READ) != 0 && (info.protection & (VM_PROT_WRITE | VM_PROT_EXECUTE)) == 0 && (info.max_protection & VM_PROT_WRITE) != 0;
        if (readOnly && ::mprotect(reinterpret_cast<void*>(first), last - first, PROT_READ | PROT_WRITE) == 0) restore.push_back({{first, last}, PROT_READ});
        cursor = last;
    }
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:reinterpret_cast<void*>(begin) length:bytes options:options deallocator:nil];
    for (const auto& [range, protection] : restore) ::mprotect(reinterpret_cast<void*>(range.first), range.second - range.first, protection);
    if (watched) GuestWriteWatch::GuestWriteWatchEndHostWrite_nid_postfix(reinterpret_cast<const void*>(begin), bytes);
    return buffer;
}

}

#endif
