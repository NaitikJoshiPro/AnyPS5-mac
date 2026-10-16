// Metal twin of tests/Graphics.cpp (agc_driver_graphics and its flags), swapped in by twins/T1-tests.cmake.
// The upstream file is compiled here unchanged with its main renamed, so its checks keep tracking upstream (see
// tools/mac/port_ledger.py). Its backend-neutral checks of the graphics state, layouts, tiling and validation run
// on Metal as they are. A check that drives Vulkan objects through MockVulkan either has a Metal twin below (MetalTwins:
// the same behaviour of the Metal code, checked without the mock) or is listed in VulkanOnly with the reason it has
// no Metal counterpart here and the work package that owns its port; listed checks are not run. `--probe` runs every
// upstream check in its own process and prints which pass: the owner of a twin that breaks a check moves it to
// MetalTwins or VulkanOnly with a reason, or ports it.
// Flag modes: --covered-centroid-state (agc_driver_covered_centroid_state) runs upstream's multisampleTests;
// --descriptor-layout-flags (agc_compute_descriptor_layout_flags) runs the Metal twin of
// descriptorLayoutFlagTests; --descriptor-pool-reset-failure (agc_descriptor_pool_reset_failure) skips: Metal
// descriptor pools are tokens whose reset cannot fail (TechnicalDebt metal-vulkan-internal-tests);
// --sparse-buffer-holes (agc_sparse_buffer_holes) runs upstream's RunSparseBufferHoleTests over the Metal device's
// Context, reading the copied bytes through the MTLBuffer the descriptor names; --address-space-teardown
// (agc_driver_address_space_teardown) runs the Metal twin of addressSpaceTeardownTests; --alpha-to-coverage-state
// (agc_driver_alpha_to_coverage_state), --undithered-alpha-to-coverage-state
// (agc_driver_undithered_alpha_to_coverage_state) and --resource-content-key / --benchmark-resource-content-key run
// upstream's checks as they are (register decoding and ShaderResources::ContentKey only); --buffer-budget-placement
// skips as Vulkan-only (memory-type and heap choice of a buffer); --image-table-allocation, --image-table-snapshot
// and --sampler-table-{lod-bias,anisotropy,reduction-compare} skip: they check Vulkan descriptor-set allocation and
// vkCreateSampler on MockVulkan, while Metal's image and sampler tables are MslAbi v15 arrays of MTLResourceIDs
// (exercised by the upstream ImageTable* execution tests); --unused-stencil-reference runs upstream's
// stencilReferenceUnusedTests as it is (State decoding only). Of upstream's 109 checks (the 107 of Checks() plus
// RunBdaResourceTests and RunTextureDetilerTests), 95 run on Metal, 5 of them through Metal twins; 14 are listed in
// VulkanOnly.

#define main AgcDriverGraphicsVulkanMain
#include "../../../tests/Graphics.cpp"
#undef main

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalTestDevice.h"
#include "../src/MetalImages.h"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include <sys/wait.h>
#include <set>
#include <unistd.h>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Check {
    const char* name;
    std::function<void()> run;
};

void DrawValidationChecks() {
    const AgcDriver::Graphics::Context context{};
    const AgcDriver::Graphics::State state{};
    AgcDriver::Pm4::DrawParameters draw{0, 0, 0, 1, 0, false};
    AgcDriver::Graphics::Draw(context, state, draw, {});
    draw.indexCount = 3;
    draw.instanceCount = 0;
    AgcDriver::Graphics::Draw(context, state, draw, {});
    draw.instanceCount = 2;
    draw.firstInstance = 0xffffffffu;
    expectFailure([&] { AgcDriver::Graphics::Draw(context, state, draw, {}); }, "instance range overflow");
    draw.firstInstance = 0;
    draw.firstVertex = 0xffffffffu;
    expectFailure([&] { AgcDriver::Graphics::Draw(context, state, draw, {}); }, "vertex range overflow");
    draw.firstVertex = 0;
    draw.indexAddress = 1;
    expectFailure([&] { AgcDriver::Graphics::Draw(context, state, draw, {}); }, "must not reference an index buffer");
    draw.indexAddress = 0;
    draw.flags = 1;
    expectFailure([&] { AgcDriver::Graphics::Draw(context, state, draw, {}); }, "draw modifiers");
    auto bounded = state;
    bounded.depthBoundsTest = true;
    bounded.minDepthBounds = 0.25f;
    AgcDriver::Graphics::ValidateDepthBounds(context, bounded);
    bounded.minDepthBounds = 1.5f;
    expectFailure([&] { AgcDriver::Graphics::ValidateDepthBounds(context, bounded); }, "depth bounds outside [0, 1]");
    auto unrestricted = context;
    unrestricted.depthRangeUnrestricted = true;
    AgcDriver::Graphics::ValidateDepthBounds(unrestricted, bounded);
}

// Upstream main's pipeline cache lock check (it is inline in main, not a function): backend-neutral, as it is.
void PipelineCacheSynchronizationChecks() {
    AgcDriver::Graphics::PipelineCacheSynchronization synchronization;
    auto pipelineUse = synchronization.LockForPipelineUse();
    bool snapshotAcquired = false;
    std::thread snapshot([&] { snapshotAcquired = synchronization.TryLockForSnapshot().owns_lock(); });
    snapshot.join();
    Require(!snapshotAcquired, "pipeline cache snapshot overlapped pipeline creation");
    pipelineUse.unlock();
    Require(synchronization.TryLockForSnapshot().owns_lock(), "pipeline cache snapshot stayed blocked after pipeline creation finished");
}

// Upstream main's order; RunBdaResourceTests and RunTextureDetilerTests take MockVulkan callbacks in main and
// are listed in VulkanOnly only.
std::vector<Check> Checks() {
    return {
        {"pipelineCacheSynchronization", PipelineCacheSynchronizationChecks},
        {"drawValidation", DrawValidationChecks},
        {"RunGuestLeaseWaitTests", RunGuestLeaseWaitTests},
        {"ResourceContentKeyTests", [] { ResourceContentKeyTests(false); }},
        {"stateTests", stateTests},
        {"retainedPixelStageTests", retainedPixelStageTests},
        {"pipelineLibrarySetTests", pipelineLibrarySetTests},
        {"hdrComponentSwapTests", hdrComponentSwapTests},
        {"depthMaintenanceTests", depthMaintenanceTests},
        {"hardwareScreenOffsetTests", hardwareScreenOffsetTests},
        {"srgb8TargetTests", srgb8TargetTests},
        {"DepthClipTests", DepthClipTests},
        {"DepthStencilTests", DepthStencilTests},
        {"OneDimensionalColorTests", OneDimensionalColorTests},
        {"htileDepthClearTests", htileDepthClearTests},
        {"memoryValueFillTests", memoryValueFillTests},
        {"ZExportTests", ZExportTests},
        {"unorm10_11_11TargetTests", unorm10_11_11TargetTests},
        {"DepthBoundsBiasTests", DepthBoundsBiasTests},
        {"conservativeZExportTests", conservativeZExportTests},
        {"earlyFragmentTestsTests", earlyFragmentTestsTests},
        {"orderedPixelShaderTests", orderedPixelShaderTests},
        {"stencilRefExportTests", stencilRefExportTests},
        {"stencilReferenceUnusedTests", stencilReferenceUnusedTests},
        {"aliasedBufferViewTests", aliasedBufferViewTests},
        {"LineStripTests", LineStripTests},
        {"ProvokingVertexTests", ProvokingVertexTests},
        {"ProvokingVertexRenderPassTests", ProvokingVertexRenderPassTests},
        {"ConservativeRasterizationTests", ConservativeRasterizationTests},
        {"DisabledColorTests", DisabledColorTests},
        {"OpaqueDestinationAlphaTests", OpaqueDestinationAlphaTests},
        {"Color1555TargetTests", Color1555TargetTests},
        {"BlendBypassTests", BlendBypassTests},
        {"CompactedExportTests", CompactedExportTests},
        {"ReversedComponentOrderTests", ReversedComponentOrderTests},
        {"ReversedComponentOrderBlendTests", ReversedComponentOrderBlendTests},
        {"ReversedBlendRoutingTests", ReversedBlendRoutingTests},
        {"ShadersWriteMemoryTests", ShadersWriteMemoryTests},
        {"DualSourceBlendTests", DualSourceBlendTests},
        {"ReversedBlendShaderTests", ReversedBlendShaderTests},
        {"metadataPassTests", metadataPassTests},
        {"cmaskTests", cmaskTests},
        {"uint16ExportTests", uint16ExportTests},
        {"uint8x4TargetTests", uint8x4TargetTests},
        {"multisampleTests", multisampleTests},
        {"MsaaEnableSingleSampleTests", MsaaEnableSingleSampleTests},
        {"clipDistanceTests", clipDistanceTests},
        {"framebufferLayerTests", framebufferLayerTests},
        {"UnwrittenSampleLocationTests", UnwrittenSampleLocationTests},
        {"quadPixelMaskTests", quadPixelMaskTests},
        {"ShaderStageTests", ShaderStageTests},
        {"NggVertexWorkgroupTests", NggVertexWorkgroupTests},
        {"inactiveVertexPointSizeTests", inactiveVertexPointSizeTests},
        {"TuningFieldTests", TuningFieldTests},
        {"PixelInputLayoutTests", PixelInputLayoutTests},
        {"UnexportedPixelInputTests", UnexportedPixelInputTests},
        {"ColorPipeBankXorTests", ColorPipeBankXorTests},
        {"HeaderlessVertexStageTests", HeaderlessVertexStageTests},
        {"HiddenUserWordVertexTablesTests", HiddenUserWordVertexTablesTests},
        {"ComputeScratchTests", ComputeScratchTests},
        {"shaderUserDataTailPaddingTests", shaderUserDataTailPaddingTests},
        {"InitialContextTests", InitialContextTests},
        {"AlphaToCoverageTests", AlphaToCoverageTests},
        {"descriptorPoolTests", descriptorPoolTests},
        {"pushConstantTests", pushConstantTests},
        {"resourceTests", resourceTests},
        {"oversizedDescriptorTests", oversizedDescriptorTests},
        {"descriptorSnapshotTests", descriptorSnapshotTests},
        {"descriptorCacheTests", descriptorCacheTests},
        {"bufferBudgetPlacementTests", bufferBudgetPlacementTests},
        {"imageTableAllocationTests", imageTableAllocationTests},
        {"imageTableSnapshotAllocationTests", imageTableSnapshotAllocationTests},
        {"samplerTableNonSeamlessCubeTests", samplerTableNonSeamlessCubeTests},
        {"samplerTableDeviceLimitTests", [] { samplerTableDeviceLimitTests(); }},
        {"misalignedShaderDataTests", misalignedShaderDataTests},
        {"debugBranchTests", debugBranchTests},
        {"textureCacheBudgetTests", textureCacheBudgetTests},
        {"sampledTextureBudgetTests", sampledTextureBudgetTests},
        {"storageTextureBudgetTests", storageTextureBudgetTests},
        {"sampledBudgetReportTests", sampledBudgetReportTests},
        {"meshArgumentTests", meshArgumentTests},
        {"meshIndexBufferTests", meshIndexBufferTests},
        {"stagingPoolBudgetTests", stagingPoolBudgetTests},
        {"validationTests", validationTests},
        {"vertexCopyTests", vertexCopyTests},
        {"highestDrawIndexTests", highestDrawIndexTests},
        {"storeAtFlipTests", storeAtFlipTests},
        {"storeAtLabelTests", storeAtLabelTests},
        {"nullVertexDescriptorTests", nullVertexDescriptorTests},
        {"vertexZeroFillTests", vertexZeroFillTests},
        {"vertexZeroFillMergeTests", vertexZeroFillMergeTests},
        {"parallelCompareTests", parallelCompareTests},
        {"pixelParameterSlotTests", pixelParameterSlotTests},
        {"rectListTests", rectListTests},
        {"floatControlsModeTests", floatControlsModeTests},
        {"barycentricEmulationTests", barycentricEmulationTests},
        {"copiedWriterIndirectOverlapTests", copiedWriterIndirectOverlapTests},
        {"registeredReadableCoverTests", registeredReadableCoverTests},
        {"RunGuestAllocationTests", RunGuestAllocationTests},
        {"RunUnmappedGapTests", RunUnmappedGapTests},
        {"RunColorTargetLayoutTests", RunColorTargetLayoutTests},
        {"RunLiveStackAccessTests", RunLiveStackAccessTests},
        {"RunTextureFormatTests", RunTextureFormatTests},
        {"RunTextureTilingTests", RunTextureTilingTests},
        {"RunGuestTextureResourceTests", RunGuestTextureResourceTests},
        {"RunDepthSurfaceReuseTests", RunDepthSurfaceReuseTests},
        {"RunGuestSamplerResourceTests", RunGuestSamplerResourceTests},
    };
}

// Metal descriptor sets, layouts and pools are tokens (MetalVulkanImages.mm): the descriptor tables are written
// into the per-batch upload ring, so a pool never runs out and its reset cannot fail. DescriptorCache keeps
// upstream's bookkeeping on top of them (ShaderResources.mm), which these twins check through its handles and
// Counters() instead of MockVulkan's pool records.
void MetalDescriptorPoolTests() {
    using namespace AgcDriver::Graphics;
    DescriptorCache cache(mockContext());
    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    const std::array<std::uint32_t, 4> key{binding.binding, binding.descriptorType, binding.descriptorCount, binding.stageFlags};
    const auto layout = cache.Layout(key, std::span(&binding, 1));
    Require(layout != VK_NULL_HANDLE && cache.Layout(key, std::span(&binding, 1)) == layout, "a descriptor set layout missed the layout cache");
    const VkDescriptorPoolSize size{binding.descriptorType, binding.descriptorCount};
    const auto allocate = [&] { return cache.Allocate(layout, std::span(&size, 1)); };
    // More sets than one Vulkan chain pool holds (1024): a Metal pool never reports exhaustion.
    constexpr std::size_t Sets = 2 * 1024 + 5;
    std::vector<DescriptorCache::SetAllocation> sets;
    std::set<VkDescriptorSet> handles;
    for (std::size_t index = 0; index < Sets; ++index) {
        sets.push_back(allocate());
        Require(sets.back().set != VK_NULL_HANDLE && handles.insert(sets.back().set).second, "a descriptor set handle was null or reused while live");
        Require(sets.back().pool == sets.front().pool, "a Metal descriptor pool reported exhaustion and the chain advanced");
    }
    Require(cache.Counters().pools == 1 && cache.Counters().sets == Sets && cache.Counters().dedicatedPools == 0, "the descriptor cache counted other than one chain pool for its sets");
    for (std::size_t index = 0; index + 1 < sets.size(); ++index) cache.Free(sets[index]);
    const auto again = allocate();
    Require(again.pool == sets.front().pool && cache.Counters().pools == 1, "a pool with a live set was not kept for new sets");
    cache.Free(sets.back());
    cache.Free(again);
    const auto recycled = allocate();
    Require(recycled.pool == sets.front().pool && cache.Counters().pools == 1, "an empty descriptor pool was not recycled");
    cache.Free(recycled);
    cache.Free(DescriptorCache::SetAllocation{});
}

// descriptorCacheTests on Metal: sets above the chain pool's capacity get a dedicated pool of their own, freed with
// the set; the device's per-set limits and the supported descriptor types are enforced as upstream.
void MetalDescriptorCacheTests() {
    auto context = mockContext();
    context.limits.maxDescriptorSetStorageBuffers = 8192;
    context.limits.maxPerStageDescriptorStorageBuffers = 8192;
    context.limits.maxPerStageResources = 8192;
    context.limits.maxDescriptorSetSampledImages = 2048;
    AgcDriver::Graphics::DescriptorCache cache(context);
    const auto layout = [&](std::uint32_t count) {
        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        const std::array<std::uint32_t, 4> key{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count, VK_SHADER_STAGE_FRAGMENT_BIT};
        return cache.Layout(key, std::span(&binding, 1));
    };
    const auto large = layout(4097);
    const std::array<VkDescriptorPoolSize, 2> oversized{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4097}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3}}};
    const auto dedicated = cache.Allocate(large, oversized);
    Require(dedicated.set != VK_NULL_HANDLE && dedicated.pool != VK_NULL_HANDLE, "a set above the chain pool's capacity got no set");
    Require(cache.Counters().pools == 0 && cache.Counters().dedicatedPools == 1 && cache.Counters().sets == 1, "an oversized set does not get a pool of its own outside the chain");
    cache.Free(dedicated);
    const std::array<VkDescriptorPoolSize, 1> regular{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096}}};
    const auto chained = cache.Allocate(layout(4096), regular);
    Require(chained.set != VK_NULL_HANDLE && chained.pool != dedicated.pool && cache.Counters().pools == 1 && cache.Counters().dedicatedPools == 1, "a set the chain pool holds left the chain");
    cache.Free(chained);
    const auto first = cache.Allocate(layout(4096), regular);
    const auto oversizedMixed = cache.Allocate(large, oversized);
    const auto second = cache.Allocate(layout(4096), regular);
    Require(first.pool == chained.pool && second.pool == chained.pool && oversizedMixed.pool != chained.pool && cache.Counters().dedicatedPools == 2 && cache.Counters().pools == 1, "an oversized set disturbed the chain pool");
    cache.Free(oversizedMixed);
    cache.Free(first);
    cache.Free(second);
    const std::array<VkDescriptorPoolSize, 1> beyondDevice{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8193}}};
    expectFailure([&] { cache.Allocate(large, beyondDevice); }, "descriptor set exceeds the device's per-set descriptor limit");
    const std::array<VkDescriptorPoolSize, 1> storageImages{{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}}};
    expectFailure([&] { cache.Allocate(large, storageImages); }, "descriptor set exceeds the device's per-set descriptor limit");
    const std::array<VkDescriptorPoolSize, 1> uniformBuffers{{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}}};
    expectFailure([&] { cache.Allocate(large, uniformBuffers); }, "descriptor set uses an unsupported descriptor type 6");
    // Upstream's update-after-bind sets with a variable image table (2026-10): the same allocations and per-set
    // limits; the layout and pool create flags and the variable count are Vulkan create-info fields MockVulkan
    // records, which Metal's token layouts and pools do not keep, so the twin checks the cache's handles,
    // counters and SetAllocation::updateAfterBind instead.
    context.limits.maxDescriptorSetStorageBuffers = 16;
    context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindStorageBuffers = 8192;
    context.descriptorIndexingLimits.maxPerStageUpdateAfterBindResources = 16384;
    context.limits.maxPerStageDescriptorSampledImages = 2048;
    {
        AgcDriver::Graphics::DescriptorCache combined(context);
        const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4097, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}, {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 32, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
        constexpr auto tableFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
        const std::array<VkDescriptorBindingFlags, 2> flags{VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, tableFlags};
        const std::array<std::uint32_t, 9> key{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4097, VK_SHADER_STAGE_FRAGMENT_BIT, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 32, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT};
        const auto combinedLayout = combined.Layout(key, bindings, true, flags);
        Require(combinedLayout != VK_NULL_HANDLE && combined.Layout(key, bindings, true, flags) == combinedLayout && combined.Counters().layoutHits == 1, "the combined descriptor layout was not reused");
        const std::array<VkDescriptorPoolSize, 2> oversizedTable{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4097}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3}}};
        const auto dedicatedTable = combined.Allocate(combinedLayout, oversizedTable, true, 3u);
        Require(dedicatedTable.set != VK_NULL_HANDLE && combined.Counters().dedicatedPools == 1 && combined.Counters().sets == 1, "an oversized update-after-bind set with a variable image table did not get a dedicated pool");
        combined.Free(dedicatedTable);
        auto fittingBindings = bindings;
        fittingBindings[0].descriptorCount = 17;
        auto fittingKey = key;
        fittingKey[2] = 17;
        const auto fittingLayout = combined.Layout(fittingKey, fittingBindings, true, flags);
        const std::array<VkDescriptorPoolSize, 2> fitting{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 17}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3}}};
        const auto chainedTable = combined.Allocate(fittingLayout, fitting, true, 3u);
        Require(chainedTable.set != VK_NULL_HANDLE && chainedTable.updateAfterBind && chainedTable.pool != dedicatedTable.pool && combined.Counters().dedicatedPools == 1, "a fitting update-after-bind set with a variable image table left the update-after-bind chain");
        combined.Free(chainedTable);
        const auto reused = combined.Allocate(fittingLayout, fitting, true, 0u);
        Require(reused.set != VK_NULL_HANDLE && reused.pool == chainedTable.pool && reused.updateAfterBind, "reusing an update-after-bind pool lost an empty variable image table");
        combined.Free(reused);
        expectFailure([&] { combined.Allocate(fittingLayout, fitting, false, 3u); }, "descriptor set exceeds the device's per-set descriptor limit");
    }
}

// descriptorLayoutFlagTests on Metal: the layout cache is keyed by the binding flags, and ShaderResources exports
// one flag per layout binding (update-after-bind exactly when the set needs it), as upstream; there is no native
// Vulkan layout to compare them with.
void MetalDescriptorLayoutFlagTests() {
    constexpr VkDescriptorBindingFlags indexed = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    {
        AgcDriver::Graphics::DescriptorCache cache(mockContext());
        const VkDescriptorSetLayoutBinding binding{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        const std::array<std::uint32_t, 4> key{3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, VK_SHADER_STAGE_COMPUTE_BIT};
        const std::array<VkDescriptorBindingFlags, 1> flags{indexed};
        const auto layout = cache.Layout(key, std::span(&binding, 1), false, flags);
        Require(cache.Layout(key, std::span(&binding, 1), false, flags) == layout && cache.Counters().layoutHits == 1, "identical binding flags missed the layout cache");
        Require(cache.Layout(key, std::span(&binding, 1)) != layout && cache.Counters().layoutMisses == 2, "layout cache ignored binding flags");
        const std::array<VkDescriptorBindingFlags, 2> mismatched{indexed, 0};
        expectFailure([&] { cache.Layout(key, std::span(&binding, 1), false, mismatched); }, "descriptor binding flag count differs from layout");
    }
    for (const bool updateAfterBind : {false, true}) {
        auto context = mockContext();
        context.emptyBuffer = reinterpret_cast<VkBuffer>(std::uintptr_t{0x1000});
        context.descriptorIndexingLimits.maxPerStageDescriptorUpdateAfterBindStorageBuffers = 17;
        context.descriptorIndexingLimits.maxPerStageUpdateAfterBindResources = 128;
        context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindStorageBuffers = 32;
        ShaderRecompiler::RecompileResult program;
        const std::uint32_t count = updateAfterBind ? 17 : 1;
        program.bindings.push_back(makeBinding(Role::GuestBuffers, 0, count, std::vector<std::uint32_t>(4 * count)));
        const AgcDriver::Graphics::CompiledShader compute{ShaderRecompiler::ShaderStage::Compute, &program, 0};
        AgcDriver::Graphics::ShaderResources resources(context, compute);
        const auto flags = resources.LayoutBindingFlags();
        Require(flags.size() == resources.LayoutBindings().size() && flags.size() == 1, "exported binding flag count differs from the layout");
        Require(flags[0] == (updateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0), "exported binding flags differ from the layout's");
    }
}

// The Metal test device with the image cache attached as device creation attaches it, and the Context limits the
// device's setup gives (MetalDevice.mm), for the twins below that need guest memory on a real Metal device.
class MetalGraphicsDevice {
public:
    MetalGraphicsDevice() : cache(test.device, test.queue) {
        auto& limits = test.context.limits;
        const auto bufferLength = static_cast<std::uint64_t>(test.device.maxBufferLength);
        limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(bufferLength, 0xffffffffu));
        limits.maxUniformBufferRange = limits.maxStorageBufferRange;
        limits.minStorageBufferOffsetAlignment = 4u;
        limits.nonCoherentAtomSize = 1u;
        limits.maxBoundDescriptorSets = 4u;
        limits.maxPerStageDescriptorStorageBuffers = 16;
        limits.maxPerStageResources = 128;
        limits.maxDescriptorSetStorageBuffers = 32;
        test.context.nullDescriptors = true;
    }
    // Attaches the image cache once the test has set its Context fields; the Context it returns is the one
    // image and buffer builds use.
    const AgcDriver::Graphics::Context& Attach(const std::function<void(AgcDriver::Graphics::Context&)>& adjust = {}) {
        if (adjust) adjust(test.context);
        cache.Attach(test.context);
        return *cache.ImageContext();
    }

private:
    AgcDriver::Graphics::MetalTest::Device test;
    AgcDriver::Graphics::Metal::ImageCache cache;
};

// copiedWriterIndirectOverlapTests on Metal (upstream 0f2683cb7): the same guest blocks and checks over the Metal
// device's Context instead of MockVulkan's; the copied region is staged in a Metal buffer, the registered one is
// a Metal host import (no-copy buffer over its 16 KiB pages).
void MetalCopiedWriterIndirectOverlapTests() {
    using namespace ShaderRecompiler;
    using namespace AgcDriver::Graphics;
    constexpr std::size_t half = 65536;
    void* directBlock = ::operator new(half, std::align_val_t{half});
    std::memset(directBlock, 0xaa, half);
    const auto directAddress = reinterpret_cast<std::uintptr_t>(directBlock);
    void* copiedBlock = ::operator new(half, std::align_val_t{half});
    std::memset(copiedBlock, 0xbb, half);
    const auto copiedAddress = reinterpret_cast<std::uintptr_t>(copiedBlock);
    {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, directBlock, half, true, true, true);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    }
    {
        MetalGraphicsDevice device;
        const auto& context = device.Attach([](Context& context) {
            context.hostImportAlignment = half;
            context.limits.minStorageBufferOffsetAlignment = 16;
            context.limits.maxStorageBufferRange = 1u << 27;
        });
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        {
            GuestBufferMemory memory(context);
            memory.AcquireRegistered();
            Require(HostImportCovers(context, directAddress, half), "direct registered range must be covered by host import");
            Require(!HostImportCovers(context, copiedAddress, half), "copied unregistered range must NOT be covered by host import");
            memory.AddWritable(copiedAddress, 1024);
            memory.AddWritable(directAddress, 1024);
            memory.Upload(false);
            Require(memory.HasCopiedWrites(), "writer with one copied region must report HasCopiedWrites() == true");
            Require(memory.WritesOverlap(directAddress, 16), "WritesOverlap must return true for direct written region");
            Require(memory.WritesOverlap(copiedAddress, 16), "WritesOverlap must return true for copied written region");
            Require(!memory.WritesOverlap(directAddress + 2048, 16), "WritesOverlap must return false for unwritten range");
            Require(!memory.CopiedWritesOverlap(directAddress, 16), "CopiedWritesOverlap must return false for direct written region");
            Require(memory.CopiedWritesOverlap(copiedAddress, 16), "CopiedWritesOverlap must return true for copied written region");
            Require(!memory.CopiedWritesOverlap(directAddress + 2048, 16), "CopiedWritesOverlap must return false for unwritten range");
            memory.WriteBack();
        }
        {
            RecompileResult shader;
            shader.runtimeAbiVersion = RuntimeAbi::Version;
            auto copiedBinding = makeBinding(Role::GuestBuffers, 0, 1, vsharp(copiedBlock, 1024));
            copiedBinding.bufferWritten = {true};
            auto directBinding = makeBinding(Role::GuestBuffers, 1, 1, vsharp(directBlock, 1024));
            directBinding.bufferWritten = {true};
            shader.bindings = {copiedBinding, directBinding};
            CompiledShader compiled{ShaderStage::Compute, &shader, 0};
            auto writer = std::make_shared<ShaderResources>(context, compiled);
            Require(writer->HasCopiedWrites(), "ShaderResources with copied region must report HasCopiedWrites() == true");
            Require(writer->WritesOverlap(directAddress, 16), "ShaderResources must report WritesOverlap for direct region");
            Require(!writer->CopiedWritesOverlap(directAddress, 16), "ShaderResources must NOT report CopiedWritesOverlap for direct region");
            Require(writer->CopiedWritesOverlap(copiedAddress, 16), "ShaderResources must report CopiedWritesOverlap for copied region");
            auto copiedWriters = DrawCopiedWriters();
            copiedWriters->push_back(writer);
            const bool mainOverlaps = std::any_of(copiedWriters->begin(), copiedWriters->end(), [&](const auto& w) { return w->WritesOverlap(directAddress, 16); });
            const bool prOverlaps = std::any_of(copiedWriters->begin(), copiedWriters->end(), [&](const auto& w) { return w->CopiedWritesOverlap(directAddress, 16); });
            Require(mainOverlaps, "main's check (WritesOverlap) must flag an overlap over direct arguments");
            Require(!prOverlaps, "PR's check (CopiedWritesOverlap) must disambiguate and NOT flag an overlap over direct arguments");
            Require(std::any_of(copiedWriters->begin(), copiedWriters->end(), [&](const auto& w) { return w->CopiedWritesOverlap(copiedAddress, 16); }), "PR's check must flag an overlap over copied arguments");
            copiedWriters->erase(std::remove(copiedWriters->begin(), copiedWriters->end(), writer), copiedWriters->end());
            writer->WriteBack();
        }
        ClearHostImports(context.device);
    }
    {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, directBlock);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    }
    ::operator delete(directBlock, std::align_val_t{half});
    ::operator delete(copiedBlock, std::align_val_t{half});
}

// oversizedDescriptorTests on Metal (upstream e9c98900e7): a V# whose size passes the storage buffer range limit
// is bound up to the end of the registered allocation behind it when that is under the limit, refused when the
// allocation reaches the limit, and bound as the empty buffer when no allocation is registered behind it. Upstream
// also compares the bound range and bytes with the MockVulkan descriptor write; Metal writes the descriptor
// tables into the per-batch upload ring, so the twin checks what ShaderResources accepts and refuses over the
// Metal device's Context, with mockContext's 16 KiB range limit.
void MetalOversizedDescriptorTests() {
    using namespace ShaderRecompiler;
    using namespace AgcDriver::Graphics;
    constexpr std::size_t blockBytes = 65536;
    constexpr std::size_t tailBytes = 8192;
    void* block = ::operator new(blockBytes, std::align_val_t{blockBytes});
    std::memset(block, 0x5a, blockBytes);
    {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, blockBytes, true, true, true);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    }
    const auto oversized = [](const void* pointer) { return makeBinding(Role::GuestBuffers, 0, 1, vsharp(pointer, 0xffffffffu)); };
    const auto* tail = static_cast<const std::byte*>(block) + blockBytes - tailBytes;
    {
        MetalGraphicsDevice device;
        const auto& context = device.Attach([](Context& context) {
            context.hostImportAlignment = blockBytes;
            context.limits.minStorageBufferOffsetAlignment = 16;
            context.limits.maxStorageBufferRange = 16384;
        });
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto color = AgcDriver::Graphics::DecodeState(makeState()).color;
        const auto build = [&](const void* pointer) {
            RecompileResult vertex;
            RecompileResult fragment;
            vertex.runtimeAbiVersion = RuntimeAbi::Version;
            fragment.runtimeAbiVersion = RuntimeAbi::Version;
            vertex.bindings.push_back(oversized(pointer));
            ShaderResources resources(context, vertex, fragment, color, 0, 0);
        };
        build(tail);
        expectFailure([&] { build(block); }, "descriptor range limit");
        build(reinterpret_cast<const void*>(0x1000));
        ClearHostImports(context.device);
    }
    {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    }
    ::operator delete(block, std::align_val_t{blockBytes});
}

// agc_sparse_buffer_holes on Metal: upstream's RunSparseBufferHoleTests (tests/BdaResources.cpp) over the Metal
// device's Context; the bytes of a copy are the contents of the shared MTLBuffer its descriptor names.
int MetalSparseBufferHoles() {
    bool passed = false;
    @autoreleasepool {
        MetalGraphicsDevice device;
        const auto& context = device.Attach([](AgcDriver::Graphics::Context& context) { context.limits.maxStorageBufferRange = 1u << 27; });
        passed = RunSparseBufferHoleTests(context, {
            [](VkBuffer buffer) -> std::span<std::byte> {
                const id<MTLBuffer> metal = AgcDriver::Graphics::Metal::BufferOf(buffer);
                Require(metal != nil && metal.contents != nullptr, "a sparse buffer copy is not a CPU-visible Metal buffer");
                return {static_cast<std::byte*>(metal.contents), static_cast<std::size_t>(metal.length)};
            },
            {}, {}, {}, {}
        });
    }
    std::cout << "sparse buffer hole tests " << (passed ? "passed" : "failed") << '\n';
    return passed ? 0 : 1;
}

// agc_driver_address_space_teardown on Metal (upstream abc997b0d, ported to GuestBufferMemory.mm's
// ClearImageMirrors): upstream's addressSpaceTeardownTests over the Metal device's Context instead of a mock one
// with a made-up VkDevice. context.device is the Metal device's handle, the one MetalDevice.mm's teardown passes to
// ClearImageMirrors. Deviations: the build runs under GuestMemory::GpuMutex as the Metal driver's builds do
// (released before the guest block is freed, so the free sees the same state as upstream's), and the host imports
// are cleared at the end so no no-copy buffer outlives the device; upstream's MockVulkan leak check has no
// counterpart.
void MetalAddressSpaceTeardownTests() {
    setenv("APS5_NO_LEASE_MIRROR", "1", 1);
    @autoreleasepool {
        MetalGraphicsDevice device;
        const auto& context = device.Attach([](AgcDriver::Graphics::Context& context) {
            context.bufferDeviceAddress = true;
            context.limits.maxStorageBufferRange = 1u << 27;
        });
        void* block = GuestHeap::GuestHeapAlignForGpu_nid_postfix(16, 64);
        const auto before = AgcDriver::Graphics::LeaseCounters().cacheDrops;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            {
                AgcDriver::Graphics::GuestBufferMemory memory(context);
                memory.AcquireRegistered();
                Require(memory.HoldsLease(), "address-based build holds no lease before device teardown");
                memory.Upload(true);
                memory.WriteBack();
                Require(!memory.HoldsLease(), "completed address-based build kept its own lease");
            }
            AgcDriver::Graphics::ClearImageMirrors(context.device);
        }
        GuestHeap::GuestHeapFree_nid_postfix(block);
        Require(AgcDriver::Graphics::LeaseCounters().cacheDrops == before, "device teardown left the cached address space pinning guest allocations");
        AgcDriver::Graphics::ClearHostImports(context.device);
    }
}

// Upstream modes this twin does not run, with the skip line ctest shows (exit 77).
struct SkippedMode {
    const char* flag;
    const char* line;
};

const std::vector<SkippedMode> SkippedModes{
    {"--buffer-budget-placement", "skipped, bufferBudgetPlacementTests is Vulkan-only: it checks which VkMemoryType and heap a host-coherent Buffer allocates from under VK_EXT_memory_budget; a Metal Buffer is one shared MTLBuffer in unified memory, with no memory types or heaps to choose between"},
    {"--image-table-allocation", "skipped, the Metal twin of --image-table-allocation (imageTableAllocationTests) is not written: it checks the variable-count Vulkan descriptor set, while Metal's image tables are MslAbi v15 MTLResourceID arrays (ImageTable* execution tests)"},
    {"--image-table-snapshot", "skipped, the Metal twin of --image-table-snapshot (imageTableSnapshotAllocationTests) is not written: it checks the variable-count Vulkan descriptor set, while Metal's image tables are MslAbi v15 MTLResourceID arrays (ImageTable* execution tests)"},
    {"--sampler-table-lod-bias", "skipped, the Metal twin of --sampler-table-lod-bias (samplerTableDeviceLimitTests) is not written: it checks vkCreateSampler limits on MockVulkan, while Metal's sampler tables are MslAbi v15 MTLResourceID arrays (ImageTable* execution tests)"},
    {"--sampler-table-anisotropy", "skipped, the Metal twin of --sampler-table-anisotropy (samplerTableDeviceLimitTests) is not written: it checks vkCreateSampler limits on MockVulkan, while Metal's sampler tables are MslAbi v15 MTLResourceID arrays (ImageTable* execution tests)"},
    {"--sampler-table-reduction-compare", "skipped, the Metal twin of --sampler-table-reduction-compare (samplerTableDeviceLimitTests) is not written: it checks vkCreateSampler limits on MockVulkan, while Metal's sampler tables are MslAbi v15 MTLResourceID arrays (ImageTable* execution tests)"},
};

// earlyFragmentTestsTests on Metal (upstream fdefd7660): the MSL fragment function carries
// [[early_fragment_tests]] (reflection earlyFragmentTests) for the same DB_SHADER_CONTROL values that
// declare EarlyFragmentTests in SPIR-V, except that Metal refuses the attribute on a function returning
// [[depth]], so DEPTH_BEFORE_SHADER with Z_EXPORT_ENABLE keeps late tests (TechnicalDebt
// metal-early-depth-export). Each request is also checked after a serializer round trip.
// upstream's depthExportRequest builds a SPIR-V target; the twin compiles the same request for MSL.
bool MetalDeclaresEarlyFragmentTests(ShaderRecompiler::RecompileRequest request) {
    request.target.backend = ShaderRecompiler::ShaderBackend::Msl;
    const auto result = ShaderRecompiler::Recompile(request);
    Require(!result.msl.empty(), "the depth-export pixel program produced no MSL");
    const bool reflected = result.msl.Reflection().earlyFragmentTests;
    const bool attributed = std::string(result.msl.Source()).find("[[early_fragment_tests]]") != std::string::npos;
    Require(reflected == attributed, "the MSL reflection and the stage function disagree on early fragment tests");
    return reflected;
}

void MetalEarlyFragmentTestsTests() {
    const std::array<std::pair<std::uint32_t, bool>, 9> cases{{
        {0x0000u, false},
        {0x0020u, false},
        {0x0010u, true},
        {0x0030u, true},
        {0x0050u, false},
        {0x0011u, false},
        {0x1000u, true},
        {0x1050u, true},
        {0x1031u, false},  // upstream: true; Z_EXPORT_ENABLE keeps late tests on Metal
    }};
    const ShaderRecompiler::RequestSerializer serializer;
    for (const auto& [shaderControl, expected] : cases) {
        const auto request = depthExportRequest(shaderControl);
        const auto replayed = serializer.Deserialize(serializer.Serialize(request));
        Require(MetalDeclaresEarlyFragmentTests(request) == expected && MetalDeclaresEarlyFragmentTests(replayed.request) == expected, "DB_SHADER_CONTROL " + std::to_string(shaderControl) + (expected ? " did not declare" : " declared") + " early fragment tests");
    }
    Require(depthExportRequest(0x1031u).context.pixel->earlyZ, "DEPTH_BEFORE_SHADER with a depth export did not decode early tests (only the Metal stage function keeps them late)");
    for (const auto shaderControl : {0x0010u, 0x1000u}) {
        auto request = depthExportRequest(shaderControl);
        request.context.pixel->alphaToCoverage = true;
        const auto replayed = serializer.Deserialize(serializer.Serialize(request));
        Require(!MetalDeclaresEarlyFragmentTests(request) && !MetalDeclaresEarlyFragmentTests(replayed.request), "DB_SHADER_CONTROL " + std::to_string(shaderControl) + " with alpha-to-coverage declared early fragment tests");
        // Upstream 72518f64f: a quad-masked pixel function discards uncovered pixels, so it tests late.
        request = depthExportRequest(shaderControl);
        request.context.pixel->quadPixelMask = 0x1u;
        const auto masked = serializer.Deserialize(serializer.Serialize(request));
        Require(!MetalDeclaresEarlyFragmentTests(request) && !MetalDeclaresEarlyFragmentTests(masked.request), "DB_SHADER_CONTROL " + std::to_string(shaderControl) + " with a quad pixel mask declared early fragment tests");
        // Upstream reads the driver's view (DeclaresEarlyFragmentTests over the SPIR-V); the Metal driver reads the
        // MSL reflection, which MetalDeclaresEarlyFragmentTests checks against the stage function.
        Require(MetalDeclaresEarlyFragmentTests(depthExportRequest(shaderControl)), "the driver did not read early fragment tests from a recompiled pixel shader");
    }
}

struct Twin {
    const char* name;
    const char* what;
    void (*run)();
};

// Upstream checks of MockVulkan objects whose Metal behaviour is checked by a twin instead.
const std::vector<Twin> MetalTwins{
    {"descriptorPoolTests", "pools never exhaust, one chain pool is recycled", MetalDescriptorPoolTests},
    {"descriptorCacheTests", "dedicated pools for oversized sets, per-set limits, update-after-bind sets with a variable image table", MetalDescriptorCacheTests},
    {"copiedWriterIndirectOverlapTests", "copied vs direct written ranges over a Metal host import", MetalCopiedWriterIndirectOverlapTests},
    {"earlyFragmentTestsTests", "[[early_fragment_tests]] in the MSL stage function, late with a depth export or a quad pixel mask", MetalEarlyFragmentTestsTests},
    {"oversizedDescriptorTests", "a V# past the range limit bound up to its registered allocation, refused at the limit, empty when unregistered", MetalOversizedDescriptorTests},
};

const Twin* TwinOf(const std::string& name) {
    for (const auto& twin : MetalTwins) {
        if (name == twin.name) return &twin;
    }
    return nullptr;
}

// Checks of Vulkan objects through MockVulkan with no Metal twin here, with the work package whose Metal twin
// replaces what they test.
const std::vector<std::pair<std::string, std::string>> VulkanOnly{
    {"resourceTests", "Vulkan buffers, images and pool slabs on MockVulkan: D2 (BufferPool, Resources twins)"},
    {"descriptorSnapshotTests", "descriptor snapshots recorded into a Vulkan recorder: D5 (ShaderResources twin), D2"},
    {"misalignedShaderDataTests", "shader data uploaded through the Vulkan buffer pool: D5 (ShaderResources twin), D2"},
    {"vertexZeroFillTests", "zero-filled vertex buffers from the Vulkan buffer pool: D6 (Draw twin), D2"},
    {"RunBdaResourceTests", "BDA buffers and descriptors on MockVulkan: D3 (BdaResources twin)"},
    {"RunDepthSurfaceReuseTests", "depth surface copies recorded into Vulkan command batches: D6 (DepthSurface twin), D2"},
    {"RunTextureDetilerTests", "detile dispatch descriptors on MockVulkan: D4 (TextureDetiler twin)"},
    {"pipelineLibrarySetTests", "graphics pipeline library layouts through MockVulkan: Metal has no pipeline libraries (TechnicalDebt metal-graphics-pipeline-libraries)"},
    {"RunGuestSamplerResourceTests", "checks the VkSamplerCreateInfo chain through a captured vkCreateSampler (Context.deviceProc); Sampler.mm builds MTLSamplerDescriptors instead, checked by agc_metal_sampler_reduction and agc_metal_images: D5"},
    {"bufferBudgetPlacementTests", "Vulkan-only: the VkMemoryType and heap a host-coherent Buffer allocates from under VK_EXT_memory_budget; a Metal Buffer is one shared MTLBuffer in unified memory"},
    {"imageTableAllocationTests", "the bounded variable-count image table set on MockVulkan; Vulkan descriptor sets only (Metal tables: MslAbi v15 arrays)"},
    {"imageTableSnapshotAllocationTests", "an empty image table in a draw snapshot on MockVulkan; Vulkan descriptor sets only (Metal tables: MslAbi v15 arrays)"},
    {"samplerTableNonSeamlessCubeTests", "sampler table entries and their vkCreateSampler flags on MockVulkan; Vulkan samplers only (Metal tables: MslAbi v15 arrays)"},
    {"samplerTableDeviceLimitTests", "sampler table entries refused by device limits on MockVulkan; Vulkan samplers only (Metal tables: MslAbi v15 arrays)"},
};

bool Listed(const std::string& name) {
    for (const auto& [listed, reason] : VulkanOnly) {
        if (listed == name) return true;
    }
    return false;
}

std::string Run(const Check& check) {
    try {
        check.run();
        return {};
    } catch (const std::exception& error) {
        return error.what();
    }
}

int Probe() {
    std::size_t failed = 0;
    for (const auto& check : Checks()) {
        std::fflush(stdout);
        const pid_t child = fork();
        if (child == 0) {
            const auto error = Run(check);
            if (!error.empty()) std::fprintf(stderr, "%s\n", error.c_str());
            std::fflush(stderr);
            std::_Exit(error.empty() ? 0 : 1);
        }
        int status = 0;
        waitpid(child, &status, 0);
        const bool passed = WIFEXITED(status) && WEXITSTATUS(status) == 0;
        failed += passed ? 0u : 1u;
        std::printf("%s %s%s\n", passed ? "PASS" : "FAIL", check.name, Listed(check.name) ? " (listed as Vulkan-only)" : TwinOf(check.name) != nullptr ? " (replaced by its Metal twin)" : "");
    }
    std::printf("%zu check(s) failed in isolation\n", failed);
    return 0;
}

// One upstream mode: its checks, then the line upstream prints when they pass (none for the silent modes).
int RunMode(const Check& check, const char* passed) {
    const auto error = Run(check);
    if (!error.empty()) {
        std::cerr << error << '\n';
        return 1;
    }
    if (passed != nullptr) std::cout << passed << '\n';
    return 0;
}

}

int main(int argc, char** argv) {
    // As upstream's main, for every mode.
    setenv("APS5_PIN_WAIT_MS", "200", 1);
    setenv("APS5_HEAP_MIRROR_MIB", "4", 1);
    const std::string_view mode = argc == 2 ? std::string_view(argv[1]) : std::string_view{};
    // agc_driver_graphics --unused-stencil-reference (upstream 3490838653): State decoding only, as upstream; the
    // check prints its own summary line.
    if (mode == "--unused-stencil-reference") return RunMode({"stencilReferenceUnusedTests", stencilReferenceUnusedTests}, nullptr);
    if (mode == "--descriptor-layout-flags") return RunMode({"descriptorLayoutFlagTests", MetalDescriptorLayoutFlagTests}, "Descriptor layout cache and exported binding flags passed on Metal");
    if (mode == "--sparse-buffer-holes") {
        try {
            return MetalSparseBufferHoles();
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    if (mode == "--address-space-teardown") return RunMode({"addressSpaceTeardownTests", MetalAddressSpaceTeardownTests}, "address-space device teardown releases cached guest allocations (Metal device)");
    if (mode == "--undithered-alpha-to-coverage-state") return RunMode({"UnditheredAlphaToCoverageTests", UnditheredAlphaToCoverageTests}, "Existing undithered alpha-to-coverage states passed");
    if (mode == "--resource-content-key" || mode == "--benchmark-resource-content-key") {
        const bool benchmark = mode == "--benchmark-resource-content-key";
        return RunMode({"ResourceContentKeyTests", [benchmark] { ResourceContentKeyTests(benchmark); }}, nullptr);
    }
    if (mode == "--alpha-to-coverage-state") {
        return RunMode({"alphaToCoverageState", [] {
            MsaaEnableSingleSampleTests();
            AlphaToCoverageTests();
        }}, "Alpha-to-coverage state and refusal tests passed");
    }
    for (const auto& skipped : SkippedModes) {
        if (mode == skipped.flag) {
            std::cout << skipped.line << '\n';
            return 77;
        }
    }
    if (mode == "--descriptor-pool-reset-failure") {
        std::cout << "skipped, the device has no Vulkan descriptor pools: Metal descriptor pools are tokens whose reset cannot fail, so the termination on a failed reset is unreachable\n";
        return 77;
    }
    // agc_driver_covered_centroid_state (upstream fdefd7660): register decoding only, as upstream.
    if (mode == "--covered-centroid-state") return RunMode({"multisampleTests", multisampleTests}, "Covered-centroid and multisample state checks passed");
    if (mode == "--probe") return Probe();
    if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [--probe | --unused-stencil-reference | --descriptor-layout-flags | --descriptor-pool-reset-failure | --sparse-buffer-holes | --covered-centroid-state | --address-space-teardown | --alpha-to-coverage-state | --undithered-alpha-to-coverage-state | --resource-content-key | --benchmark-resource-content-key";
        for (const auto& skipped : SkippedModes) std::cerr << " | " << skipped.flag;
        std::cerr << "]\n";
        return 2;
    }
    std::size_t ran = 0;
    std::string twins;
    for (const auto& check : Checks()) {
        if (Listed(check.name)) continue;
        const auto* twin = TwinOf(check.name);
        const auto error = twin != nullptr ? Run({check.name, twin->run}) : Run(check);
        if (!error.empty()) {
            std::cerr << check.name << (twin != nullptr ? " (Metal twin)" : "") << ": " << error << '\n';
            return 1;
        }
        ++ran;
        if (twin != nullptr) twins += std::string(twins.empty() ? "" : ", ") + twin->name + " (" + twin->what + ")";
    }
    std::cout << "Graphics validation tests passed on Metal: " << ran << " checks, through Metal twins: " << twins << "; not run (Vulkan-only, or a Metal gap named with its owner):";
    for (const auto& [name, reason] : VulkanOnly) std::cout << ' ' << name << " (" << reason << ')';
    std::cout << '\n';
    return 0;
}
