#include "prx/libSceAgcDriver/Submit/include/CommandBuffer.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Queue.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"

#include <cerrno>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

// (queue handle, descriptor) as the oops-sdk calls it on retail firmware 12.40, where its GL
// path renders through it (oops-sdk docs/hardware/agc-gl-cube-oracle-fw1240.md). OpenAGC
// declares (uint32 queue, dcb, size in dwords) instead; see TechnicalDebt.
int APS5_VABI sceAgcDriverSubmitCommandBuffer(void* queue_context, const Packet* packet) {
    if (!queue_context || !packet) return EINVAL;
    if (AgcDriver::IsQueueHandle(queue_context)) {
        AgcDriver::Submit(packet, AgcDriver::QueueFromHandle(queue_context, __func__));
        return 0;
    }
    const auto queue = *reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(queue_context) + 4);
    if (queue != 0 && (queue < 0x20 || queue >= 0x58)) {
        throw std::runtime_error(std::string(__func__) + ": unsupported queue");
    }
    AgcDriver::Submit(packet, queue);
    return 0;
}

int APS5_VABI sceAgcDriverSubmitMultiCommandBuffers(void* queue_context, uint32_t* const* command_buffers, const uint32_t* sizes_in_dwords, uint32_t count) {
    if (count == 0) return 0;
    if (!queue_context || !command_buffers || !sizes_in_dwords) return EINVAL;
    const auto queue = *reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(queue_context) + 4);
    if (queue != 0 && (queue < 0x20 || queue >= 0x58)) {
        throw std::runtime_error(std::string(__func__) + ": unsupported queue");
    }
    return AgcDriver::SubmitMany(command_buffers, sizes_in_dwords, count, queue);
}

}
