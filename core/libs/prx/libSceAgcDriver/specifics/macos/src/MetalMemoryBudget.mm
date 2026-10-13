// The Metal device's memory heap and VK_EXT_memory_budget report (MetalMemoryBudget.h).

#include "MetalMemoryBudget.h"

#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <algorithm>
#include <atomic>

namespace AgcDriver::Graphics::Metal {

namespace {

std::atomic<MemoryPressure> pressureLevel{MemoryPressure::Normal};

// One process-wide watch of the OS memory pressure level, started by the first device.
void WatchMemoryPressure() {
    static dispatch_source_t source = [] {
        const auto mask = DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL;
        dispatch_source_t watch = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0, mask, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
        if (watch == nullptr) return watch;
        dispatch_source_set_event_handler(watch, ^{
            const auto level = dispatch_source_get_data(watch);
            pressureLevel.store((level & DISPATCH_MEMORYPRESSURE_CRITICAL) != 0 ? MemoryPressure::Critical : (level & DISPATCH_MEMORYPRESSURE_WARN) != 0 ? MemoryPressure::Warn : MemoryPressure::Normal, std::memory_order_relaxed);
        });
        dispatch_resume(watch);
        return watch;
    }();
    (void)source;
}

// vkGetPhysicalDeviceMemoryProperties2 of the Metal device: `physical` is its bridged id<MTLDevice>.
void VKAPI_PTR ReportMemory(VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2* properties) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)reinterpret_cast<void*>(physical);
    const auto workingSet = static_cast<std::uint64_t>(device.recommendedMaxWorkingSetSize);
    properties->memoryProperties = {};
    properties->memoryProperties.memoryHeapCount = 1;
    properties->memoryProperties.memoryHeaps[0] = {workingSet, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    for (auto* next = static_cast<VkBaseOutStructure*>(properties->pNext); next != nullptr; next = next->pNext) {
        if (next->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) continue;
        auto* reported = reinterpret_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(next);
        std::fill(std::begin(reported->heapBudget), std::end(reported->heapBudget), VkDeviceSize{0});
        std::fill(std::begin(reported->heapUsage), std::end(reported->heapUsage), VkDeviceSize{0});
        const auto usage = static_cast<std::uint64_t>(device.currentAllocatedSize);
        reported->heapUsage[0] = usage;
        reported->heapBudget[0] = MemoryBudget(workingSet, usage, SystemMemoryHeadroom(), CurrentMemoryPressure());
    }
}

}

std::uint64_t MemoryBudget(std::uint64_t workingSet, std::uint64_t usage, std::uint64_t headroom, MemoryPressure pressure) {
    if (pressure == MemoryPressure::Warn) headroom /= 2u;
    if (pressure == MemoryPressure::Critical) headroom = 0u;
    const auto reachable = usage > UINT64_MAX - headroom ? UINT64_MAX : usage + headroom;
    return std::min(workingSet, reachable);
}

MemoryPressure CurrentMemoryPressure() {
    return pressureLevel.load(std::memory_order_relaxed);
}

std::uint64_t SystemMemoryHeadroom() {
    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const auto host = mach_host_self();
    const auto result = host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&statistics), &count);
    mach_port_deallocate(mach_task_self(), host);
    // Unknown: no limit from the system, the working set bounds the budget.
    if (result != KERN_SUCCESS) return UINT64_MAX;
    const auto pages = static_cast<std::uint64_t>(statistics.free_count) + statistics.speculative_count + statistics.inactive_count;
    return pages * static_cast<std::uint64_t>(vm_kernel_page_size);
}

void DescribeMemory(Context& context, id<MTLDevice> device) {
    context.memory = {};
    context.memory.memoryHeapCount = 1;
    context.memory.memoryHeaps[0] = {static_cast<VkDeviceSize>(device.recommendedMaxWorkingSetSize), VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    context.physical = reinterpret_cast<VkPhysicalDevice>((__bridge void*)device);
    context.memoryProperties2 = &ReportMemory;
    WatchMemoryPressure();
}

}
