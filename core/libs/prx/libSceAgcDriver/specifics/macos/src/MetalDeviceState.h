#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALDEVICESTATE_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALDEVICESTATE_H

// The Metal device's state, shared by the Objective-C++ files that implement the VulkanDevice API
// on macOS (MetalDevice.mm: device, submission, compute; MetalPresentation.mm; MetalDraw.mm).
// Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace AgcDriver {

namespace Metal {

[[noreturn]] inline void Unimplemented(const char* what, const char* package) {
    throw std::runtime_error(std::string("Metal device: ") + what + " is not implemented yet (" + package + ", docs/mac/M2_DESIGN.md)");
}

inline std::string Describe(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

}

struct VulkanDevice::State {
    id<MTLDevice> device = nil;
    id<MTL4CommandQueue> queue = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTLSharedEvent> event = nil;
    id<MTLResidencySet> residency = nil;
    id<MTL4ArgumentTable> arguments = nil;
    id<MTLBuffer> ring = nil;
    id<MTLBuffer> faults = nil;
    std::size_t ringUsed = 0;
    std::uint64_t submitted = 0;
    std::vector<std::uint32_t> capabilities;
    std::vector<std::string_view> extensions;

    struct Batch {
        std::uint64_t value;
        id<MTL4CommandAllocator> allocator;
    };
    std::deque<Batch> inflight;
    std::vector<id<MTL4CommandAllocator>> freeAllocators;
    // Command buffer errors reported by commit feedback, which may arrive after the device is gone.
    struct Errors {
        std::mutex mutex;
        std::vector<std::string> messages;
    };
    std::shared_ptr<Errors> errors = std::make_shared<Errors>();

    struct Wrapper {
        std::uint64_t begin;
        std::uint64_t end;
        id<MTLBuffer> buffer;
    };
    std::vector<Wrapper> wrappers;

    std::mutex pipelinesMutex;
    std::unordered_map<std::string, id<MTLLibrary>> libraries;
    std::unordered_map<std::string, id<MTLComputePipelineState>> computePipelines;

    std::uint64_t Allocate(std::size_t bytes, std::size_t alignment, void** host);
    std::uint64_t Wrap(std::uint64_t address, std::uint64_t bytes);
    id<MTLLibrary> Library(const ShaderRecompiler::MslShader& shader);
    id<MTLComputePipelineState> ComputePipeline(const ShaderRecompiler::RecompileResult& shader);
    void Retire(bool wait);
    void CheckFaults();
};

}

#endif
