#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SUBMIT_INCLUDE_QUEUE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SUBMIT_INCLUDE_QUEUE_HPP

#include "SceTypes.hpp"

namespace AgcDriver {

// The driver queue a handle from sceAgcDriverCreateQueue submits to. Throws for a handle
// that sceAgcDriverCreateQueue did not return or that was destroyed.
std::uint32_t QueueFromHandle(const void* handle, const char* function);
// True for a live handle from sceAgcDriverCreateQueue (sceAgcDriverSubmitCommandBuffer also takes
// upstream's guest queue contexts, whose queue id is the dword at offset 4).
bool IsQueueHandle(const void* handle);

}

extern "C" int APS5_VABI sceAgcDriverCreateQueue(std::uint32_t type, void** queue, std::uint32_t flags);
extern "C" int APS5_VABI sceAgcDriverDestroyQueue(void* queue);
extern "C" int APS5_VABI sceAgcDriverSubmitCommandBuffer(void* queue, const Packet* packet);

#endif
