// Metal twin of tests/execution/GuestMemoryLifetime.cpp (agc_driver_guest_memory_lifetime and its copy, direct,
// locked, heap and heap_uncached modes; upstream 304798015b), swapped in by twins/UPSYNC3-tests.cmake. The upstream
// file runs unchanged; only its capability query differs. tests/GuestMemoryLifetimeSupport.hpp asks libvulkan
// (SDL_LoadObject("libvulkan.so.1")) whether the device has VK_EXT_external_memory_host, which throws on macOS.
// The Metal device always imports guest memory (no-copy buffers over registered allocations,
// GuestBufferMemory.mm HostImportFor), so the query answers true unless APS5_NO_HOST_IMPORT disables imports.
// The header's include guard is defined here so the upstream file picks up this definition.

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <cstdio>
#include <cstdlib>

#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_GUESTMEMORYLIFETIMESUPPORT_HPP

static inline bool GuestMemoryLifetimeHasHostImports(VkDevice) {
    if (std::getenv("APS5_NO_HOST_IMPORT") != nullptr) {
        std::puts("skipped, host imports are disabled by APS5_NO_HOST_IMPORT");
        return false;
    }
    return true;
}

#include "../../../tests/execution/GuestMemoryLifetime.cpp"
