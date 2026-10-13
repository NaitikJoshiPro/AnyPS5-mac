// Metal twin of tests/Recorder.cpp's --storage-upload-lifetime-only and --storage-upload-failure-only modes
// (agc_storage_upload_lifetime, agc_storage_upload_failure; storageUploadLifetimeTests, Recorder.cpp:3126-3263).
// A storage texture whose guest memory is a DCC fast clear is created by recording a clear of its image
// on the open batch. The image must outlive that recorded clear when the caller drops the texture at once
// and when creating its storage view fails after the clear was recorded (failure injection), and it must
// be released by recorder teardown. On Metal this matters as on Vulkan: an MTL4 command buffer does not
// retain the textures it uses, and DestroyImage removes the texture from the image residency set.
// Upstream intercepts the Vulkan device functions; the twin observes the same calls through
// Metal::ImageProbe (MetalVulkanImages.h). Metal textures own their memory (VkDeviceMemory is a token,
// FreeMemory does nothing), so "released" is DestroyImage of the image the clear targeted. Expected values
// are upstream's.

#include "RecorderTwinHarness.h"
#include "RecorderTwins.h"
#include "../src/MetalVulkanImages.h"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

namespace {

class StorageUploadProbe final : public Metal::ImageProbe {
public:
    StorageUploadProbe(const Context& context, bool failView) : context(context), failView(failView) {
        Require(active == nullptr, "nested storage upload probe");
        active = this;
        Metal::SetImageProbe(this);
    }
    ~StorageUploadProbe() override {
        Metal::SetImageProbe(nullptr);
        if (imageDestroyed) Metal::Vk::DestroyImage(context.device, image, nullptr);
        active = nullptr;
    }
    VkImage Image() const { return image; }
    bool Destroyed() const { return imageDestroyed; }
    bool Released() const { return imageDestroyed; }
    bool FailedView() const { return failedView; }
    std::uint32_t Clears() const { return clears; }

    void Created(VkImage created) override { image = created; }
    bool FailView(const VkImageViewCreateInfo& info) override {
        if (info.image != image || !failView || clears == 0) return false;
        failedView = true;
        return true;
    }
    bool Destroying(VkImage destroyed) override {
        if (destroyed != image) return false;
        imageDestroyed = true;
        return true;
    }
    void Cleared(VkImage cleared) override {
        if (cleared == image) ++clears;
    }

private:
    static inline StorageUploadProbe* active = nullptr;
    const Context& context;
    bool failView;
    VkImage image = VK_NULL_HANDLE;
    bool imageDestroyed = false;
    bool failedView = false;
    std::uint32_t clears = 0;
};

void storageUploadLifetimeTests(const Device& device, Recorder& previous, bool failView) {
    auto context = device.GetContext();
    context.hostImportAlignment = 0;
    TextureDetiler detiler(context);
    context.detiler = &detiler;
    GuestTextureResource resource{};
    resource.width = 64;
    resource.height = 64;
    resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kLinear;
    resource.dimension = TextureDimension::k2D;
    resource.format = 56;
    const auto guestBytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
    std::vector<std::byte> texels(guestBytes + 256, std::byte{0x55});
    std::vector<std::byte> keys(guestBytes / 256, std::byte{0});
    resource.baseAddress = (reinterpret_cast<std::uintptr_t>(texels.data()) + 255) & ~std::uintptr_t{255};
    resource.dccAddress = reinterpret_cast<std::uintptr_t>(keys.data());
    DccKeyProof proof;
    Require(ProvedClearKeys(resource, guestBytes, proof) == DccKeys::Clear0000, "storage lifetime fixture has no DCC clear");
    StorageUploadProbe probe(context, failView);
    Buffer readback(device.GetContext(), resource.width * resource.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(readback.Bytes().data(), 0x55, readback.Bytes().size());
    std::weak_ptr<StorageTexture> lifetime;
    bool premature = false;
    bool retained = true;
    bool threw = false;
    {
        Recorder recorder(context);
        recorder.Activate();
        try {
            auto image = StorageTexture::Create(context, detiler, resource, 0);
            lifetime = image;
            image.reset();
            retained = !lifetime.expired();
        } catch (const std::exception& error) {
            if (!probe.FailedView() || std::string_view(error.what()).find("vkCreateImageView storage") == std::string_view::npos) throw;
            threw = true;
        }
        premature = probe.Destroyed();
        Require(probe.Clears() == 1 && recorder.Recording(), "storage creation did not record its initial clear");
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {resource.width, resource.height, 1};
        Metal::Vk::CmdCopyImageToBuffer(commands, probe.Image(), VK_IMAGE_LAYOUT_GENERAL, readback.Handle(), 1, &copy);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Sync();
    }
    previous.Activate();
    Require(threw == failView && probe.FailedView() == failView, "storage view failure injection did not take the requested path");
    Require(std::all_of(readback.Bytes().begin(), readback.Bytes().end(), [](std::byte value) { return value == std::byte{0}; }), "recorded storage clear did not reach every texel");
    Require(lifetime.expired() && probe.Released(), "storage upload resources outlived recorder teardown");
    Require(!premature && retained, failView ? "failed storage creation released an image still referenced by its recorded clear" : "storage creation did not retain its image through the recorded clear");
    std::cout << "Storage upload lifetime" << (failView ? " after view failure" : " after caller release") << ": passed\n";
}

const bool registered = [] {
    AgcDriver::MetalTwins::RegisterRecorderMode("--storage-upload-lifetime-only", [] { return RunMode([](const Device& device, Recorder& recorder) { storageUploadLifetimeTests(device, recorder, false); }); });
    AgcDriver::MetalTwins::RegisterRecorderMode("--storage-upload-failure-only", [] { return RunMode([](const Device& device, Recorder& recorder) { storageUploadLifetimeTests(device, recorder, true); }); });
    return true;
}();

}
