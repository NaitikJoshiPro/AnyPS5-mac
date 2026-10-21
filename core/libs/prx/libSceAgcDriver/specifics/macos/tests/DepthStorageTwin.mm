// Metal twin of tests/execution/DepthStorage.cpp (agc_driver_depth_storage), swapped in by twins/T1-tests.cmake.
// The upstream file is compiled here with its main renamed; its helpers (PlaneResource, CountFloats, ExpectFailure,
// the guest block) are used as written, and MetalRun is upstream's Run with two device differences only:
// - the Context is the Metal image cache's (detiler, placeholders, caches) instead of a Vulkan device's;
// - upstream's submit (Recorder::Submit, then vkQueueWaitIdle) is Submit then Recorder::Sync, which waits for the
//   batch and runs its completions (the storage images' write-backs into guest memory).
// Every expectation is upstream's: the extent/format/level refusals, the cleared depth (0.25) and stencil (7)
// reaching the storage images, a store back through DepthSurfaceStoreBack reaching the surface (0.75, then 0.5),
// one-layer 2D array views, the replaced-view refusal, the two-layer and base-slice layer-pitch refusals (upstream
// 696d1660c dd7d649c3), a store back of a storage image the surface did not hand out (a84afe931) and
// MetalRunPairings, upstream's RunPairings: a store back goes into the newest pairing of the plane (b3e5acc8c).
// Upstream sync 6 (6b6c701636): MetalDepthCopyCase, upstream's DepthCopyCase (DB depth copies, see there).

#define main AgcDriverDepthStorageVulkanMain
#include "../../../tests/execution/DepthStorage.cpp"
#undef main

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalImages.h"
#include "MetalTestDevice.h"

namespace MetalDepthStorageTwin {

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

void MetalRun(Context context, std::uint8_t* block) {
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto stencilAddress = address + BlockBytes / 2;
    const auto submit = [&](Recorder& recorder) {
        recorder.Submit();
        recorder.Sync();
    };
    Recorder recorder(context);
    recorder.Activate();
    const auto readBack = [&](StorageTexture& texture) {
        texture.MarkDirty();
        texture.WriteBack();
        submit(recorder);
    };
    const DepthTarget target{address, stencilAddress, {Side, Side}, VK_FORMAT_D32_SFLOAT_S8_UINT, 0.25f, 7};
    Require(DepthSurfaceView(context, target) != VK_NULL_HANDLE, "depth surface was not created");
    Require(DepthSurfaceStorage(context, {}, PlaneResource(address, Format32Float, Side / 2), 0) == nullptr, "a storage view of another extent over the depth memory was taken for the depth surface");
    Require(DepthSurfaceTexture(context, {}, PlaneResource(address, Format32Float, Side / 2), {}) == nullptr, "a sampled view of another extent over the depth memory was taken for the depth surface");
    Require(DepthSurfaceStorage(context, {}, PlaneResource(address, Format8Uint), 0) == nullptr, "an 8-bit storage view of the depth plane was taken for the depth surface");
    // Upstream 5d9dcc42d2..0d078f75d3: a later level over the depth memory is declined (nullptr), not refused.
    Require(DepthSurfaceStorage(context, {}, PlaneResource(address, Format32Float), 1) == nullptr, "a storage access to a later level over the depth memory was taken for the depth surface");
    auto later = PlaneResource(address, Format32Float);
    later.mipCount = 2;
    later.baseLevel = 1;
    later.lastLevel = 1;
    Require(DepthSurfaceStorage(context, {}, later, 0) == nullptr, "a storage view of a later level over the depth memory was taken for the depth surface");
    Require(DepthSurfaceTexture(context, {}, later, {}) == nullptr, "a sampled view of a later level over the depth memory was taken for the depth surface");
    const auto depth = DepthSurfaceStorage(context, {}, PlaneResource(address, Format32Float), 0);
    Require(depth != nullptr, "the depth plane has no storage image");
    submit(recorder);
    readBack(*depth);
    Require(CountFloats(block, 0.25f) == Side * Side, "the storage image did not receive the cleared depth");
    const std::array<std::uint32_t, 4> pattern{std::bit_cast<std::uint32_t>(0.75f), std::bit_cast<std::uint32_t>(0.75f), std::bit_cast<std::uint32_t>(0.75f), std::bit_cast<std::uint32_t>(0.75f)};
    const char* refusal = nullptr;
    Require(depth->FillClear(pattern, StorageTexture::WholeImage, refusal), refusal != nullptr ? refusal : "the storage image could not be filled");
    Require(DepthSurfaceStoreBack(context, *depth), "the depth plane's storage image was not stored back");
    submit(recorder);
    Require(DepthSurfaceStorage(context, {}, PlaneResource(address, Format32Float), 0) == depth, "the depth plane's storage image was not reused");
    submit(recorder);
    readBack(*depth);
    Require(CountFloats(block, 0.75f) == Side * Side, "the stored depth did not come back from the depth surface");
    const auto stencil = DepthSurfaceStorage(context, {}, PlaneResource(stencilAddress, Format8Uint), 0);
    Require(stencil != nullptr && stencil != depth, "the stencil plane has no storage image of its own");
    submit(recorder);
    readBack(*stencil);
    const auto stencilBytes = DescribeSurface(PlaneResource(stencilAddress, Format8Uint)).guestBytes;
    Require(stencilBytes <= BlockBytes / 2 && std::count(block + BlockBytes / 2, block + BlockBytes / 2 + stencilBytes, 7) == static_cast<std::ptrdiff_t>(Side * Side), "the storage image did not receive the cleared stencil");
    const std::array<std::uint32_t, 4> half{std::bit_cast<std::uint32_t>(0.5f), std::bit_cast<std::uint32_t>(0.5f), std::bit_cast<std::uint32_t>(0.5f), std::bit_cast<std::uint32_t>(0.5f)};
    Require(depth->FillClear(half, StorageTexture::WholeImage, refusal), refusal != nullptr ? refusal : "the storage image could not be filled");
    Require(DepthSurfaceStoreBack(context, *depth), "the depth plane's storage image was not stored back");
    submit(recorder);
    auto layeredDepth = PlaneResource(address, Format32Float);
    layeredDepth.dimension = TextureDimension::k2DArray;
    const auto arrayDepth = DepthSurfaceStorage(context, {}, layeredDepth, 0);
    Require(arrayDepth != nullptr && arrayDepth != depth, "a one-layer 2D array view of the depth plane was not taken for the depth surface as a storage image of its own");
    ExpectFailure([&] { static_cast<void>(DepthSurfaceStoreBack(context, *depth)); }, "through a view that another view of the plane", "a store back through the 2D view that the one-layer 2D array view replaced was accepted");
    submit(recorder);
    readBack(*arrayDepth);
    Require(CountFloats(block, 0.5f) == Side * Side, "the one-layer 2D array view did not receive the depth plane");
    std::fill(block + BlockBytes / 2, block + BlockBytes / 2 + stencilBytes, std::uint8_t{0});
    auto layeredStencil = PlaneResource(stencilAddress, Format8Uint);
    layeredStencil.dimension = TextureDimension::k2DArray;
    const auto arrayStencil = DepthSurfaceStorage(context, {}, layeredStencil, 0);
    Require(arrayStencil != nullptr, "a one-layer 2D array view of the stencil plane was not taken for the depth surface");
    submit(recorder);
    readBack(*arrayStencil);
    Require(std::count(block + BlockBytes / 2, block + BlockBytes / 2 + stencilBytes, 7) == static_cast<std::ptrdiff_t>(Side * Side), "the one-layer 2D array view did not receive the stencil plane");
    layeredStencil.depthOrLastArray = 1;
    ExpectFailure([&] { static_cast<void>(DepthSurfaceStorage(context, {}, layeredStencil, 0)); }, "storage image access to depth/stencil surface", "a two-layer 2D array view of the stencil plane was accepted");
    layeredDepth.depthOrLastArray = 1;
    ExpectFailure([&] { static_cast<void>(DepthSurfaceStorage(context, {}, layeredDepth, 0)); }, "with a layer pitch of", "a two-layer 2D array view of the depth plane with a layer pitch other than the depth slice size was accepted");
    layeredDepth.baseArray = 1;
    ExpectFailure([&] { static_cast<void>(DepthSurfaceStorage(context, {}, layeredDepth, 0)); }, "with a layer pitch of", "a 2D array view of layer 1 of the depth plane with a layer pitch other than the depth slice size was accepted");
    const auto own = StorageTexture::Create(context, *context.detiler, PlaneResource(address, Format32Float), 0);
    const std::array<std::uint32_t, 4> eighth{std::bit_cast<std::uint32_t>(0.125f), std::bit_cast<std::uint32_t>(0.125f), std::bit_cast<std::uint32_t>(0.125f), std::bit_cast<std::uint32_t>(0.125f)};
    Require(own->FillClear(eighth, StorageTexture::WholeImage, refusal), refusal != nullptr ? refusal : "the storage image could not be filled");
    Require(DepthSurfaceStoreBack(context, *own), "a storage image over the depth plane that the depth surface did not hand out was not stored back");
    submit(recorder);
    const auto reread = DepthSurfaceStorage(context, {}, PlaneResource(address, Format32Float), 0);
    Require(reread != nullptr, "the depth plane has no storage image");
    submit(recorder);
    readBack(*reread);
    Require(CountFloats(block, 0.125f) == Side * Side, "a storage image written over the depth plane did not reach the depth surface");
    Require(!DepthSurfaceStoreBack(context, *StorageTexture::Create(context, *context.detiler, PlaneResource(address, Format32Float, Side / 2), 0)), "a storage image of another extent over the depth memory was stored into the depth surface");
    recorder.Sync();
}

// Upstream's RunPairings (b3e5acc8c) with the same submit as MetalRun.
void MetalRunPairings(Context context, std::uint8_t* block) {
    const auto address = reinterpret_cast<std::uint64_t>(block);
    const auto submit = [&](Recorder& recorder) {
        recorder.Submit();
        recorder.Sync();
    };
    Recorder recorder(context);
    recorder.Activate();
    const auto readBack = [&](StorageTexture& texture) {
        texture.MarkDirty();
        texture.WriteBack();
        submit(recorder);
    };
    const DepthTarget first{address, address + BlockBytes / 2, {Side, Side}, VK_FORMAT_D32_SFLOAT_S8_UINT, 0.25f, 7};
    const DepthTarget second{address, address + BlockBytes / 2 + 8192, {Side, Side}, VK_FORMAT_D32_SFLOAT_S8_UINT, 0.5f, 3};
    Require(DepthSurfaceView(context, first) != VK_NULL_HANDLE && DepthSurfaceView(context, second) != VK_NULL_HANDLE, "the two pairings of the depth plane were not created");
    const auto depth = DepthSurfaceStorage(context, {}, PlaneResource(address, Format32Float), 0);
    Require(depth != nullptr, "the depth plane has no storage image");
    submit(recorder);
    const std::array<std::uint32_t, 4> pattern{std::bit_cast<std::uint32_t>(0.75f), std::bit_cast<std::uint32_t>(0.75f), std::bit_cast<std::uint32_t>(0.75f), std::bit_cast<std::uint32_t>(0.75f)};
    const char* refusal = nullptr;
    Require(depth->FillClear(pattern, StorageTexture::WholeImage, refusal), refusal != nullptr ? refusal : "the storage image could not be filled");
    Require(DepthSurfaceStoreBack(context, *depth), "the newest pairing's storage image was not stored back");
    submit(recorder);
    const auto reread = DepthSurfaceStorage(context, {}, PlaneResource(address, Format32Float), 0);
    Require(reread != nullptr, "the depth plane has no storage image");
    submit(recorder);
    readBack(*reread);
    Require(CountFloats(block, 0.75f) == Side * Side, "a storage image stored back went into an older pairing instead of the pairing that handed it out");
    auto layered = PlaneResource(address, Format32Float);
    layered.dimension = TextureDimension::k2DArray;
    Require(DepthSurfaceStorage(context, {}, layered, 0) != nullptr, "a one-layer 2D array view of the depth plane was not taken");
    ExpectFailure([&] { static_cast<void>(DepthSurfaceStoreBack(context, *depth)); }, "through a view that another view of the plane", "a store back through a replaced view was taken by an older pairing that covers the plane");
    recorder.Sync();
}

// Upstream DepthCopyCase (191c8a8d2b 120900a342 223e90289b 67cea1c314): a DB depth copy (RunDepthCopyPass) of
// each sample of a cleared depth surface into an R32 float color target reaches guest memory exactly, its
// refusals (no surface, a sample past the last, another extent) are upstream's. Device differences: every
// Metal depth surface is shader-readable, so the sample counts are 2, 4 and 8 (8 is two 4x slices on Apple9,
// DECISIONS D28, which exercises the slice addressing) and upstream's unsampleable two-sample refusal
// (67cea1c314) has no Metal case; submit is Submit then Recorder::Sync.
void MetalDepthCopyCase(Context context, std::uint8_t* block) {
    const auto colorAddress = reinterpret_cast<std::uint64_t>(block);
    const auto depthAddress = colorAddress + BlockBytes / 4;
    Recorder recorder(context);
    recorder.Activate();
    const auto submit = [&] {
        recorder.Submit();
        recorder.Sync();
    };
    ColorTarget color{};
    color.address = colorAddress;
    color.surfaceAddress = colorAddress;
    color.extent = {Side, Side};
    color.surfaceExtent = color.extent;
    color.format = VK_FORMAT_R32_SFLOAT;
    color.elementBytes = 4;
    color.tileMode = ColorTileMode::RenderTarget;
    color.bytes = ColorTargetLayout(Side, Side, ColorTileMode::RenderTarget, 4).Bytes();
    const auto copy = [&](std::uint32_t samples, std::uint32_t sample, VkExtent2D extent) {
        RunDepthCopyPass(context, DepthCopyPass{depthAddress, extent, samples, sample, color});
        submit();
        const auto resident = CachedStorageSurface(context, SurfaceForTarget(color));
        resident->MarkDirty();
        resident->WriteBack();
        submit();
    };
    const auto countBlock = [&](float value) {
        std::size_t count = 0;
        for (std::size_t offset = 0; offset < BlockBytes; offset += sizeof(float)) {
            float texel = 0;
            std::memcpy(&texel, block + offset, sizeof(texel));
            if (texel == value) ++count;
        }
        return count;
    };
    std::fill(block, block + BlockBytes, std::uint8_t{0});
    ExpectFailure([&] { copy(1, 0, {Side, Side}); }, "no draw rendered", "a DB depth copy from memory without a depth surface was accepted");
    for (const std::uint32_t samples : {2u, 4u, 8u}) {
        Require(DepthSurfaceView(context, DepthTarget{depthAddress, 0, {Side, Side}, VK_FORMAT_D32_SFLOAT, 0.375f, 0, 0, false, samples}) != VK_NULL_HANDLE, "the multisample depth surface was not created");
        for (std::uint32_t sample = 0; sample < samples; ++sample) {
            std::fill(block, block + BlockBytes, std::uint8_t{0});
            copy(samples, sample, {Side, Side});
            Require(countBlock(0.375f) == Side * Side, "the DB depth copy of sample " + std::to_string(sample) + " of " + std::to_string(samples) + " did not store the cleared depth in every texel of the R32 float target");
        }
        ExpectFailure([&] { copy(samples, samples, {Side, Side}); }, "does not have", "a DB depth copy of a sample past the last of a multisample surface was accepted");
        ExpectFailure([&] { copy(samples, 0, {Side / 2, Side / 2}); }, "of another size", "a DB depth copy of another extent than the depth surface was accepted");
        ClearDepthSurfaces(context.device);
    }
    Require(DepthSurfaceView(context, DepthTarget{depthAddress, 0, {Side, Side}, VK_FORMAT_D32_SFLOAT, 0.625f, 0, 0, false, 1}) != VK_NULL_HANDLE, "the one-sample depth surface was not created");
    std::fill(block, block + BlockBytes, std::uint8_t{0});
    copy(1, 0, {Side, Side});
    Require(countBlock(0.625f) == Side * Side, "the DB depth copy of a one-sample depth surface did not store the cleared depth in every texel of the R32 float target");
    ExpectFailure([&] { copy(1, 1, {Side, Side}); }, "does not have", "a DB depth copy of a second sample of a one-sample surface was accepted");
    recorder.Sync();
}

}

// upstream main with the Metal device.
int main() {
    try {
        MetalDepthStorageTwin::Device device;
        auto context = device.GetContext();
        auto* block = Storage.data() + (BlockBytes - reinterpret_cast<std::uintptr_t>(Storage.data()) % BlockBytes) % BlockBytes;
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            MetalDepthStorageTwin::MetalRun(context, block);
            auto* copyBlock = CopyStorage.data() + (BlockBytes - reinterpret_cast<std::uintptr_t>(CopyStorage.data()) % BlockBytes) % BlockBytes;
            GuestAllocations::Mutation().Add(copyBlock, BlockBytes, true, true, true);
            MetalDepthStorageTwin::MetalDepthCopyCase(context, copyBlock);
            ClearCachedTextures(context.device);
            GuestAllocations::Mutation().Remove(copyBlock);
            ClearDepthSurfaces(context.device);
            MetalDepthStorageTwin::MetalRunPairings(context, block);
            ClearDepthSurfaces(context.device);
        }
        GuestAllocations::Mutation().Remove(block);
        std::cout << "depth surface storage access tests passed on Metal\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
