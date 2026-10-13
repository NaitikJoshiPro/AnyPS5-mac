#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALVMTRACE_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALVMTRACE_H

// Debug aid (APS5_METAL_TRACE): the VM object and share mode of the page holding `address`, to tell
// a copy-on-write page apart from a range the guest arena remapped under a no-copy MTLBuffer.

#include <mach/mach.h>
#include <mach/mach_vm.h>
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

}

#endif
