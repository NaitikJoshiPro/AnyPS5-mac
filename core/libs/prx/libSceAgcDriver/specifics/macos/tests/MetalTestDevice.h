#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_TESTS_METALTESTDEVICE_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_TESTS_METALTESTDEVICE_H

// A Metal device and MTL4 command queue for the D2 tests, and the Context that carries them the way
// the Metal driver's device fills it (Metal::DeviceOf, Metal::QueueOf). Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalCommandStream.h"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

namespace AgcDriver::Graphics::MetalTest {

struct Device {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    Context context{};

    // Exits 77 (ctest skip) when the machine has no Metal 4 GPU.
    Device() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::printf("skipped, the machine has no Metal 4 device\n");
            std::exit(77);
        }
        queue = [device newMTL4CommandQueue];
        if (queue == nil) {
            std::fprintf(stderr, "FAIL: no MTL4 command queue\n");
            std::exit(1);
        }
        context.device = Metal::DeviceHandle(device);
        context.queue = Metal::QueueHandle(queue);
        context.bufferDeviceAddress = true;
        // What MetalDevice.mm promises: guest MSAA storage goes through the target's storage shadow (D22/D28), so
        // the storage MSAA placeholder heaps exist (upstream fd872264e2 omits them only where this is false).
        context.storageImageMultisample = true;
        // One pool per device, as the device's setup makes it (copies of the Context share it).
        context.bufferPool = std::make_shared<BufferPool>(context);
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

}

#endif
