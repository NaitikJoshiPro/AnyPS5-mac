#include "prx/libSceAgcDriver/Submit/include/CommandBuffer.hpp"
#include "prx/libSceAgcDriver/Submit/include/Queue.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"

#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

// (queue handle, descriptor) as the oops-sdk calls it on retail firmware 12.40, where its GL
// path renders through it (oops-sdk docs/hardware/agc-gl-cube-oracle-fw1240.md). OpenAGC
// declares (uint32 queue, dcb, size in dwords) instead; see TechnicalDebt.
int APS5_VABI sceAgcDriverSubmitCommandBuffer(void* queue_context, const Packet* packet) {
    AgcDriver::Submit(packet, AgcDriver::QueueFromHandle(queue_context, __func__));
    return 0;
}

int APS5_VABI sceAgcDriverSubmitMultiCommandBuffers(void* queue_context, uint32_t* const* command_buffers, const uint32_t* sizes_in_dwords, uint32_t count) {
 (void)queue_context;
 (void)command_buffers;
 (void)sizes_in_dwords;
 (void)count;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

}
