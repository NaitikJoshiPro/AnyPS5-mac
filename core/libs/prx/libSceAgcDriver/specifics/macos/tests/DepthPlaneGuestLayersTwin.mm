// Metal twin of tests/execution/DepthPlaneGuestLayers.cpp (agc_driver_depth_plane_guest_layers), swapped in by
// twins/T1-tests.cmake. The upstream file is compiled here unchanged with its main renamed, and its Run() runs as
// written: a two-slice D32 array over a guest block whose slice 0 a depth surface holds, sampled through
// DepthSurfaceTexture (DepthSurface.mm's plane copy), with upstream's expected guest reads and copy counts. Only the
// device differs: instead of a Vulkan device whose vkCmdCopyImageToBuffer / vkCmdCopyBufferToImage are counted
// through an intercepted vkGetDeviceProcAddr, the Context is the Metal image cache's and the same two counters are
// fed by the Metal::ImageProbe copy hooks (depth surface plane copies to the staging buffer, staging to plane copy
// fills: MetalVulkanImages.h NoteImageCopy). Upstream sync 3: main runs upstream's later cases too, as upstream's
// main does: RecordedUploads (875ecc139: a recorded plane upload counts toward Recorder::OpenKeptBytes),
// RunCubeGather and RunCubeTiledSlice (03b7cb5a1: a cube gathered at the descriptor's layer offsets, 6 surface copies
// and 1 plane fill). Upstream's main gives them a Context with a TextureDetiler of its own; the image cache's Context
// already carries the Metal detiler, so the twin passes it as is (RunCubeGather and RunCubeTiledSlice still make
// their own detiler over it, as written).

#define main AgcDriverDepthPlaneGuestLayersVulkanMain
#include "../../../tests/execution/DepthPlaneGuestLayers.cpp"
#undef main

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalImages.h"
#include "../src/MetalVulkanImages.h"
#include "MetalTestDevice.h"

namespace MetalDepthPlaneGuestLayersTwin {

using namespace AgcDriver::Graphics;

// Upstream's CountImageToBuffer / CountBufferToImage, reported by the Metal copies.
class CopyCounter final : public Metal::ImageProbe {
public:
    void CopiedToBuffer(VkImage) override { imageToBufferCopies.fetch_add(1); }
    void CopiedFromBuffer(VkImage) override { bufferToImageCopies.fetch_add(1); }
};

// The Metal test device as upstream's Device: GetContext() is the image cache's Context (detiler, placeholders,
// sampler and texture caches); no recorder is active outside RecordedUploads, as upstream's (the copies run in
// one-shot CommandBatches).
class Device {
public:
    Device() : cache(test.device, test.queue) {
        test.context.hostImportAlignment = 16384u;
        cache.Attach(test.context);
    }
    ~Device() {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        ClearDepthSurfaces(test.context.device);
        cache.Teardown();
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    const Context& GetContext() const { return *cache.ImageContext(); }

private:
    MetalTest::Device test;
    mutable Metal::ImageCache cache;
};

}

// upstream main with the Metal device.
int main() {
    try {
        if (!AgcDriver::GuestMemory::WriteWatched()) {
            std::puts("skipped, guest memory has no write watch");
            return Skipped;
        }
        MetalDepthPlaneGuestLayersTwin::CopyCounter counter;
        {
            MetalDepthPlaneGuestLayersTwin::Device device;
            WatchedBlock block;
            Require(AgcDriver::GuestMemory::Watched(block.Address(), BlockBytes), "depth plane guest layers: the guest block is not write-watched");
            AgcDriver::Graphics::Metal::SetImageProbe(&counter);
            try {
                const auto& context = device.GetContext();
                Run(context, block);
                RecordedUploads(context, block);
                RunCubeGather(context, block);
                RunCubeTiledSlice(context, block);
            } catch (...) {
                AgcDriver::Graphics::Metal::SetImageProbe(nullptr);
                throw;
            }
            AgcDriver::Graphics::Metal::SetImageProbe(nullptr);
        }
        std::puts("depth plane guest layer tests passed on Metal");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
