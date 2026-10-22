// Metal twin of tests/execution/DepthSurfaceLastWriter.cpp (agc_driver_depth_surface_last_writer; upstream
// 626cff7d26, a depth surface retires once another writer covers its memory), swapped in by twins/UPSYNC3-tests.cmake.
// The upstream file is compiled here with its main renamed and its Run runs as written: binding (DepthSurfaceView),
// RetireDepthSurfaces, DepthSurfaceTexture and the write-watch generations are backend-neutral code over the Metal
// depth surfaces (DepthSurface.mm). Only the device differs: the Context is the Metal image cache's (as
// DepthStorageTwin.mm makes it) instead of a Vulkan instance's (upstream's Device stays compiled but unused), and
// Run holds the GPU mutex as the driver's callers do.

#define main AgcDriverDepthSurfaceLastWriterVulkanMain
#include "../../../tests/execution/DepthSurfaceLastWriter.cpp"
#undef main

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalImages.h"
#include "MetalTestDevice.h"

namespace MetalDepthSurfaceLastWriterTwin {

using namespace AgcDriver::Graphics;

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
    @autoreleasepool {
        try {
            MetalDepthSurfaceLastWriterTwin::Device device;
            {
                std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
                Run(device.GetContext());
                ClearDepthSurfaces(device.GetContext().device);
            }
            std::puts("depth surface last writer tests passed on Metal");
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
