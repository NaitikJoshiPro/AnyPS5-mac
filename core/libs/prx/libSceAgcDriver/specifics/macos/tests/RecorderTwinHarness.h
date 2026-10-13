#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_TESTS_RECORDERTWINHARNESS_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_TESTS_RECORDERTWINHARNESS_H

// What the Metal ports of tests/Recorder.cpp modes share (StorageUploadTwin.mm, DrawSnapshotCapTwin.mm,
// DrawInputsInPlaceTwin.mm, MultisampleGuestAccessTwin.mm): the test Device (the Metal test device with
// the image cache attached as device creation attaches it, as D3RecorderModes.mm makes it), the
// write-watched blocks of Recorder.cpp:70-97, and Run, which does what upstream main does around a mode
// (GPU mutex, import watch, an active recorder). Everything is in an unnamed namespace: each port is
// its own translation unit of agc_driver_recorder_tests. Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalTestDevice.h"
#include "../src/MetalImages.h"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include <sys/mman.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {

using namespace AgcDriver::Graphics;
using AgcDriver::GuestMemory::GpuMutex;

class Device {
public:
    Device() : cache(test.device, test.queue) {
        // The limits the device's setup gives its Context (MetalDevice.mm), which the test device leaves at zero.
        const auto bufferLength = static_cast<std::uint64_t>(test.device.maxBufferLength);
        test.context.limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(bufferLength, 0xffffffffu));
        test.context.limits.maxUniformBufferRange = test.context.limits.maxStorageBufferRange;
        test.context.limits.maxComputeWorkGroupCount[0] = test.context.limits.maxComputeWorkGroupCount[1] = test.context.limits.maxComputeWorkGroupCount[2] = 65535u;
        // Color sample counts the device can render (Apple9: 1, 2, 4; 8 on Apple10, TechnicalDebt metal-msaa-8x).
        VkSampleCountFlags samples = 0;
        for (const std::uint32_t count : {1u, 2u, 4u, 8u}) {
            if ([test.device supportsTextureSampleCount:count]) samples |= count;
        }
        test.context.limits.framebufferColorSampleCounts = samples;
        // Host imports as the device's setup enables them: no-copy buffers over whole 16 KiB pages.
        test.context.hostImportAlignment = 16384u;
        cache.Attach(test.context);
    }
    const Context& GetContext() const { return *cache.ImageContext(); }
    void WaitQueue() const {
        if (auto* recorder = Recorder::Active()) recorder->Sync();
    }

private:
    MetalTest::Device test;
    mutable Metal::ImageCache cache;
};

[[maybe_unused]] void* AllocateWatched(std::size_t bytes, std::size_t alignment) {
    if (!AgcDriver::GuestMemory::WriteWatched()) return nullptr;
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
    if (!AgcDriver::GuestMemory::Watched(reinterpret_cast<std::uint64_t>(block), bytes)) throw std::runtime_error("the watched block is not watched");
    return block;
}

[[maybe_unused]] void ReleaseWatched(void* block, std::size_t bytes) {
    munmap(block, bytes);
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, bytes);
}

// Runs one ported mode as upstream main runs it: 0 passed, 1 failed (a "FAIL:" line).
template <typename Test>
int RunMode(Test test) {
    @autoreleasepool {
        try {
            Device device;
            std::unique_lock gpu(GpuMutex());
            std::cout << "host imports " << (PrepareImportWatch(device.GetContext()) == ImportWatch::Unwatch ? "are compared" : "stay watched") << '\n';
            Recorder recorder(device.GetContext());
            recorder.Activate();
            test(device, recorder);
            recorder.Sync();
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}

}

#endif
