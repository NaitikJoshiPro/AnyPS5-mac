#include "prx/libSceAgcDriver/Submit/include/Queue.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

// Queue objects as the oops-sdk uses them on retail firmware 12.40 (oops-sdk include/agc/driver.h,
// src/gl/gl_context.c, docs/hardware/agc-gl-cube-oracle-fw1240.md): CreateQueue(type, &handle, 0)
// with type 0 (the GL universal queue) or 3 (a direct command buffer queue), then
// SubmitCommandBuffer(handle, &descriptor) with the 16-byte {address, dwords, flags} descriptor
// that sceAgcDriverSubmitDcb reads. Both types are graphics queues; the driver has one graphics
// ring (queue 0), so both submit there in order with sceAgcDriverSubmitDcb.

namespace {

struct Queue {
    std::uint32_t type;
};

std::mutex queueMutex;
std::unordered_map<const void*, std::unique_ptr<Queue>> queues;
// Handles sceAgcDriverDestroyQueue released: still ours, so a late submission throws instead of being read
// as a guest queue context.
std::unordered_set<const void*> destroyedQueues;

}

bool AgcDriver::IsQueueHandle(const void* handle) {
    std::lock_guard lock(queueMutex);
    return handle != nullptr && (queues.contains(handle) || destroyedQueues.contains(handle));
}

std::uint32_t AgcDriver::QueueFromHandle(const void* handle, const char* function) {
    std::lock_guard lock(queueMutex);
    if (!handle || !queues.contains(handle))
        throw std::runtime_error(std::string(function) + ": the queue handle was not created by sceAgcDriverCreateQueue or was destroyed");
    return 0;
}

extern "C" {

int APS5_VABI sceAgcDriverCreateQueue(std::uint32_t type, void** queue, std::uint32_t flags) {
    if (type != 0 && type != 3)
        throw std::runtime_error("sceAgcDriverCreateQueue: unsupported queue type " + std::to_string(type) + " (only 0 and 3 are known)");
    if (flags != 0)
        throw std::runtime_error("sceAgcDriverCreateQueue: nonzero queue flags are not implemented");
    if (!queue) throw std::runtime_error("sceAgcDriverCreateQueue: null queue output");
    AgcDriver::GuestMemory::CheckRange(queue, sizeof(*queue), alignof(void*), true);
    auto created = std::make_unique<Queue>(Queue{type});
    void* handle = created.get();
    {
        std::lock_guard lock(queueMutex);
        destroyedQueues.erase(handle);
        queues.emplace(handle, std::move(created));
    }
    *queue = handle;
    return 0;
}

int APS5_VABI sceAgcDriverDestroyQueue(void* queue) {
    std::lock_guard lock(queueMutex);
    if (!queue || queues.erase(queue) == 0)
        throw std::runtime_error("sceAgcDriverDestroyQueue: the queue handle was not created by sceAgcDriverCreateQueue or was already destroyed");
    destroyedQueues.insert(queue);
    return 0;
}

}
