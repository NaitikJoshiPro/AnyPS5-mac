// Metal twin of tests/execution/SparseBufferHoles.cpp (agc_driver_sparse_buffer_holes), swapped in by
// twins/T1-tests.cmake. Upstream builds a Vulkan device of its own and copies the guest buffer a sparse upload
// produced into a readback buffer with a SPIR-V compute pipeline, then checks that the GPU sees the backed guest
// bytes unchanged and zeros (not the bytes a pooled buffer's earlier user left) in the inaccessible holes. Here the
// same guest pages, upload and checks run on the Metal test device with the image cache attached (as the device
// makes its Context), and the copy is an MSL kernel recorded with CommandStream::DispatchHelper: it reads the
// Metal buffer the guest-memory descriptor names at the descriptor's offset through its GPU address and writes
// the readback buffer, one word per thread in 64-thread groups, as upstream's pipeline does. Expected values are
// upstream's.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalTestDevice.h"
#include "MslAbi.hpp"
#include "../src/MetalImages.h"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <sys/mman.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

constexpr std::size_t PageBytes = 65536;
constexpr std::size_t BufferBytes = 3 * PageBytes;
constexpr std::size_t ReservedBytes = 4 * PageBytes;
constexpr std::uint32_t WorkgroupSize = 64;
constexpr std::array<std::uint32_t, 3> Patterns{0x11223344u, 0x8899aabbu, 0x55667788u};

// Three 64 KiB guest pages of a four-page reservation; the middle one stays inaccessible unless `committed`.
class GuestPages {
public:
    explicit GuestPages(bool committed) {
        const auto block = mmap(nullptr, ReservedBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (block != MAP_FAILED) memory = static_cast<std::byte*>(block);
        Require(memory != nullptr, "cannot reserve sparse guest pages");
        try {
            for (std::size_t page = 0; page < Patterns.size(); ++page) {
                if (page == 1 && !committed) continue;
                auto* begin = memory + page * PageBytes;
                Require(mprotect(begin, PageBytes, PROT_READ | PROT_WRITE) == 0, "cannot commit sparse guest page");
                for (std::size_t offset = 0; offset < PageBytes; offset += sizeof(std::uint32_t)) std::memcpy(begin + offset, &Patterns[page], sizeof(std::uint32_t));
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~GuestPages() { release(); }
    GuestPages(const GuestPages&) = delete;
    GuestPages& operator=(const GuestPages&) = delete;
    std::uint64_t Address() const { return reinterpret_cast<std::uint64_t>(memory); }

private:
    void release() noexcept {
        if (memory != nullptr) munmap(memory, ReservedBytes);
        memory = nullptr;
    }

    std::byte* memory = nullptr;
};

constexpr const char* CopySource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct CopyParameters {
    ulong source;
    ulong destination;
};
kernel void aps5_sparse_copy(constant CopyParameters& parameters [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    device const uint* source = reinterpret_cast<device const uint*>(parameters.source);
    device uint* destination = reinterpret_cast<device uint*>(parameters.destination);
    destination[index] = source[index];
}
)MSL";

struct CopyParameters {
    std::uint64_t source;
    std::uint64_t destination;
};

class CopyPipeline {
public:
    explicit CopyPipeline(const Context& context) : context(context) {
        id<MTLDevice> device = Metal::DeviceOf(context);
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = static_cast<MTLLanguageVersion>((ShaderRecompiler::MslAbi::LanguageMajor << 16u) | ShaderRecompiler::MslAbi::LanguageMinor);
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(CopySource) options:options error:&error];
        Require(library != nil, "the sparse buffer copy kernel does not compile: " + Metal::Describe(error));
        id<MTLFunction> function = [library newFunctionWithName:@"aps5_sparse_copy"];
        pipeline = function != nil ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        Require(pipeline != nil, "no pipeline for the sparse buffer copy kernel: " + Metal::Describe(error));
        Require(pipeline.maxTotalThreadsPerThreadgroup >= WorkgroupSize, "the sparse buffer copy kernel cannot run 64 threads per group");
    }

    void Copy(const VkDescriptorBufferInfo& source, Buffer& destination) const {
        Require(source.range == BufferBytes && destination.Bytes().size() == BufferBytes, "sparse buffer holes GPU copy has an unexpected extent");
        const id<MTLBuffer> metal = Metal::BufferOf(source.buffer);
        Require(metal != nil && source.offset + source.range <= metal.length, "the sparse buffer descriptor does not name a Metal buffer covering its range");
        const CopyParameters parameters{metal.gpuAddress + source.offset, destination.DeviceAddress()};
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        CommandBatch batch(context);
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        Metal::Stream(batch.Handle()).DispatchHelper(pipeline, std::as_bytes(std::span(&parameters, 1)), MTLSizeMake(BufferBytes / sizeof(std::uint32_t) / WorkgroupSize, 1, 1), MTLSizeMake(WorkgroupSize, 1, 1));
        RecordMemoryBarrier(context, batch.Handle(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        batch.SubmitAndWait();
        destination.Invalidate();
    }

private:
    const Context& context;
    id<MTLComputePipelineState> pipeline = nil;
};

void Run(const Context& context, const CopyPipeline& pipeline, bool writable, const std::string& shape) {
    const auto label = std::string("sparse buffer holes GPU: ") + (writable ? "writable " : "readable ") + shape;
    const bool committed = shape == "committed";
    const auto requested = shape == "shortened" ? ReservedBytes : BufferBytes;
    GuestPages pages(committed);
    const auto ranges = AgcDriver::GuestMemory::DescribeCommitted(pages.Address(), requested);
    Require(ranges.whole == committed, label + ": unexpected guest commitment");
    if (!committed) {
        Require(ranges.ranges == std::vector<std::pair<std::uint64_t, std::uint64_t>>{{pages.Address(), pages.Address() + PageBytes}, {pages.Address() + 2 * PageBytes, pages.Address() + BufferBytes}}, label + ": guest hole is not inaccessible");
    }
    Buffer readback(context, BufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memset(readback.Bytes().data(), 0xcc, BufferBytes);
    VkBuffer primed = VK_NULL_HANDLE;
    {
        Buffer earlier(context, BufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        primed = earlier.Handle();
        std::memset(earlier.Bytes().data(), 0xa5, BufferBytes);
    }
    GuestBufferMemory memory(context);
    if (writable) memory.AddWritable(pages.Address(), requested);
    else memory.AddReadable(pages.Address(), requested);
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        memory.Upload(false);
    }
    std::uint32_t adjustment = 0;
    const auto descriptor = memory.Descriptor(pages.Address(), requested, adjustment);
    Require(descriptor.buffer == primed && descriptor.offset == 0 && descriptor.range == BufferBytes && adjustment == 0, label + ": the upload did not reuse the primed buffer at the expected extent");
    std::cout << label << " reused primed buffer; requested=" << requested << "; allocated=" << descriptor.range << '\n';
    pipeline.Copy(descriptor, readback);
    const auto bytes = readback.Bytes();
    std::size_t mismatches = 0;
    std::size_t staleWords = 0;
    for (std::size_t offset = 0; offset < BufferBytes; offset += sizeof(std::uint32_t)) {
        std::uint32_t actual = 0;
        std::memcpy(&actual, bytes.data() + offset, sizeof(actual));
        const auto page = offset / PageBytes;
        if (page == 1 && !committed) {
            mismatches += actual != 0;
            staleWords += actual == 0xa5a5a5a5u;
        } else {
            Require(actual == Patterns[page], label + ": GPU readback changed backed guest bytes");
        }
    }
    std::cout << label << " readback checked; words=" << BufferBytes / sizeof(std::uint32_t) << "; nonzeroHoleWords=" << mismatches << "; staleHoleWords=" << staleWords << '\n';
    Require(mismatches == 0, label + ": GPU readback exposed stale pooled bytes in guest holes");
    std::cout << label << " passed\n";
}

}

int main() {
    try {
        bool passed = true;
        @autoreleasepool {
            MetalTest::Device test;
            Metal::ImageCache cache(test.device, test.queue);
            // The limits the device's setup gives its Context (MetalDevice.mm), which the test device leaves at zero.
            auto& limits = test.context.limits;
            limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(test.device.maxBufferLength, 0xffffffffu));
            limits.maxUniformBufferRange = limits.maxStorageBufferRange;
            limits.minStorageBufferOffsetAlignment = 4u;
            limits.nonCoherentAtomSize = 1u;
            cache.Attach(test.context);
            const auto& context = *cache.ImageContext();
            std::printf("sparse buffer holes GPU: Metal device %s\n", test.device.name.UTF8String);
            Require(context.limits.maxStorageBufferRange >= BufferBytes, "Metal limits cannot support the bounded sparse buffer readback");
            {
                CopyPipeline pipeline(context);
                for (const bool writable : {false, true}) {
                    for (const auto* shape : {"interior", "shortened", "committed"}) {
                        try {
                            Run(context, pipeline, writable, shape);
                        } catch (const std::exception& error) {
                            if (std::string(error.what()).find(": GPU readback exposed stale pooled bytes in guest holes") == std::string::npos) throw;
                            std::cerr << error.what() << '\n';
                            passed = false;
                        }
                    }
                }
            }
            cache.Teardown();
        }
        if (!passed) return 1;
        std::puts("Sparse buffer hole GPU readback tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
