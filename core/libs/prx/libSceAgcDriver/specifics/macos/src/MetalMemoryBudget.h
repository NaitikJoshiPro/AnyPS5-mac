#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALMEMORYBUDGET_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALMEMORYBUDGET_H

// The memory a Metal device's Context describes (work packages D1/D5): what upstream's Vulkan device
// takes from vkGetPhysicalDeviceMemoryProperties and VK_EXT_memory_budget, which the sampled and
// storage texture caches size themselves by (ShaderResources.mm TextureCacheBudget, SampledTextureBudget).
// - Heap: a unified-memory Mac has one device-local heap, the device's recommendedMaxWorkingSetSize.
// - Budget report (VK_EXT_memory_budget semantics: heapUsage is what this process has allocated on
//   the heap, heapBudget what it can have allocated in all before allocations fail or degrade the
//   system): usage = currentAllocatedSize; budget = min(working set, usage + headroom), where the
//   headroom is the memory the system can hand out without paging (free, speculative and inactive
//   pages), halved while the OS reports memory pressure WARN and zero at CRITICAL
//   (DISPATCH_SOURCE_TYPE_MEMORYPRESSURE). So the texture caches shrink towards their floor under
//   pressure and grow back when it passes, as they do under a Vulkan driver's budget.
// Objective-C++ only.

#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>

namespace AgcDriver::Graphics::Metal {

enum class MemoryPressure : std::uint8_t { Normal, Warn, Critical };

// VK_EXT_memory_budget's heapBudget for the device-local heap (all values in bytes).
std::uint64_t MemoryBudget(std::uint64_t workingSet, std::uint64_t usage, std::uint64_t headroom, MemoryPressure pressure);

// The OS memory pressure level, as last reported to the process (Normal until a report arrives).
MemoryPressure CurrentMemoryPressure();

// Bytes the system can hand out without paging: free, speculative and inactive pages.
std::uint64_t SystemMemoryHeadroom();

// Fills `context.memory` (the heap), `context.physical` (the device, as the budget report's handle)
// and `context.memoryProperties2` (the budget report) for `device`, and starts the pressure watch.
void DescribeMemory(Context& context, id<MTLDevice> device);

}

#endif
