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
#include "prx/libc/include/GuestDisplaced.hpp"
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
// Test seam (tests/ReplacedDeviceQueue.mm): the command queue the host-import residency set of `device`
// (an MTLDevice) is attached to, null when it has none. GuestBufferMemory.mm defines it.
const void* HostImportResidencyQueue(const void* device);

// guestBegin is a guest address: the write watch keys pages by guest address, while the region scan, the temporary
// mprotect and the wrap itself use the host address, the alias for a displaced range (GuestDisplaced.hpp, D31/D33;
// offsets into the buffer are the same).
inline id<MTLBuffer> NewNoCopyBuffer(id<MTLDevice> device, std::uint64_t guestBegin, std::uint64_t bytes, MTLResourceOptions options) {
    const bool watched = GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(guestBegin, bytes);
    if (watched) GuestWriteWatch::GuestWriteWatchBeginHostWrite_nid_postfix(reinterpret_cast<const void*>(guestBegin), bytes);
    const std::uint64_t begin = GuestArena::HostAddress(guestBegin);
    const auto end = begin + bytes;
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
    if (watched) GuestWriteWatch::GuestWriteWatchEndHostWrite_nid_postfix(reinterpret_cast<const void*>(guestBegin), bytes);
    return buffer;
}

}

#endif
