// Presentation members of the Metal device (work package D6, docs/mac/M2_DESIGN.md): CAMetalLayer
// drawables from the SDL Metal view, a headless offscreen target when there is no window, the
// scaler pass and the present slots.

#include "MetalDeviceState.h"

namespace AgcDriver {

using Metal::Unimplemented;

void* VulkanDevice::Window() const {
    return nullptr;
}

void VulkanDevice::Resize(std::uint32_t, std::uint32_t) {
}

bool VulkanDevice::Presentable() const {
    return false;
}

std::size_t VulkanDevice::FlipInFlight() {
    return 1u;
}

double VulkanDevice::RetirePresents(std::size_t) {
    return 0.0;
}

bool VulkanDevice::PresentWaitsForSlots(const DisplayBuffer*) const {
    return false;
}

bool VulkanDevice::AcquireImage() {
    Unimplemented("presentation (AcquireImage)", "work package D6");
}

bool VulkanDevice::PresentClear(std::uint32_t, std::uint32_t, bool) {
    Unimplemented("presentation (PresentClear)", "work package D6");
}

void VulkanDevice::PresentPixels(std::uint32_t, std::uint32_t, std::span<const std::byte>) {
    Unimplemented("presentation (PresentPixels)", "work package D6");
}

bool VulkanDevice::PresentDisplayBuffer(const DisplayBuffer&) {
    Unimplemented("presentation (PresentDisplayBuffer)", "work package D6");
}

double VulkanDevice::FinishPresent() {
    return 0.0;
}

void VulkanDevice::QueuePresent() {
    Unimplemented("presentation (QueuePresent)", "work package D6");
}

void VulkanDevice::FlipBatches(std::uint64_t& submissions, std::uint64_t& unsignaled) const {
    submissions = state->recorder != nullptr ? state->recorder->Submissions() : 0u;
    unsignaled = state->recorder != nullptr ? state->recorder->InFlightBatches() : 0u;
}

VulkanDevice::PresentStatistics VulkanDevice::PresentCounts() {
    return {};
}

}
