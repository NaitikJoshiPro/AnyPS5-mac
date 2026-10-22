// Work packages D4 (textures) and D5 (image/sampler descriptors): see MetalImages.h.
#include "MetalImages.h"
#include "prx/libSceAgcDriver/Graphics/include/GuestSamplerResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "RuntimeAbi.hpp"
#include "MetalResidency.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics::Metal {

namespace {

[[noreturn]] void Fail(const std::string& message) {
    throw std::runtime_error("Metal textures (work package D4): " + message);
}

[[noreturn]] void FailD5(const std::string& message) {
    throw std::runtime_error("Metal descriptors (work package D5): " + message);
}

}

// --- ImageCache ----------------------------------------------------------------------------------

namespace {

struct CacheRegistry {
    std::mutex mutex;
    std::vector<std::pair<void*, ImageCache*>> caches;
};

CacheRegistry& Caches() {
    static CacheRegistry registry;
    return registry;
}

}

ImageCache* ImageCache::For(id<MTLDevice> device) {
    if (device == nil) return nullptr;
    auto& registry = Caches();
    std::lock_guard lock(registry.mutex);
    // The newest cache of the device (a test may keep an older, queue-less one alive).
    for (auto entry = registry.caches.rbegin(); entry != registry.caches.rend(); ++entry) {
        if (entry->first == (__bridge void*)device) return entry->second;
    }
    return nullptr;
}

void ImageCache::Track(id<MTLTexture> texture) {
    const std::array<id<MTLTexture>, 1> one{texture};
    MakeResident(one);
}

void ImageCache::TrackBuffer(id<MTLBuffer> buffer) {
    std::lock_guard lock(mutex);
    if (tornDown || buffer == nil) return;
    auto& count = residentCounts[(__bridge void*)buffer];
    if (count++ == 0u) {
        ::AgcDriver::Metal::Residency::Add(residency, buffer);
        ::AgcDriver::Metal::Residency::Commit(residency);
    }
}

void ImageCache::UntrackBuffer(id<MTLBuffer> buffer) {
    std::lock_guard lock(mutex);
    if (tornDown || buffer == nil) return;
    const auto found = residentCounts.find((__bridge void*)buffer);
    if (found == residentCounts.end()) return;
    if (--found->second == 0u) {
        ::AgcDriver::Metal::Residency::Remove(residency, buffer);
        residentCounts.erase(found);
        ::AgcDriver::Metal::Residency::Commit(residency);
    }
}

void ImageCache::Untrack(id<MTLTexture> texture) {
    const std::array<id<MTLTexture>, 1> one{texture};
    Release(one);
}

ImageCache::ImageCache(id<MTLDevice> device, id<MTL4CommandQueue> queue) : device(device), queue(queue) {
    {
        auto& registry = Caches();
        std::lock_guard lock(registry.mutex);
        registry.caches.emplace_back((__bridge void*)device, this);
    }
    MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
    descriptor.label = @"aps5 images";
    NSError* error = nil;
    residency = [device newResidencySetWithDescriptor:descriptor error:&error];
    if (residency == nil) Fail("cannot create the image residency set");
    if (queue != nil) ::AgcDriver::Metal::Residency::Attach(queue, residency);
}

ImageCache::~ImageCache() {
    Teardown();
}

void ImageCache::Attach(const Context& deviceContext) {
    if (imageContext != nullptr) return;
    imageContext = std::make_unique<Context>(deviceContext);
    auto& context = *imageContext;
    detiler = std::make_unique<TextureDetiler>(context);
    context.detiler = detiler.get();
    context.textureCompressionBC = [device supportsBCTextureCompression];
    context.srgbDecodeFormats = SrgbDecodeFormats();
    context.nullDescriptors = false;
    context.descriptorCache = nullptr;
    context.descriptorIndexing = false;
    context.imageViewMinLod = false;
    // The limits a ShaderResources build checks (Metal argument tables hold the heaps; the MslAbi
    // capacities bound them).
    auto& limits = context.limits;
    limits.maxBoundDescriptorSets = std::max(limits.maxBoundDescriptorSets, 4u);
    limits.maxPerStageResources = std::max(limits.maxPerStageResources, 1u << 20u);
    limits.maxPerStageDescriptorStorageBuffers = std::max(limits.maxPerStageDescriptorStorageBuffers, 1u << 20u);
    limits.maxPerStageDescriptorSampledImages = std::max(limits.maxPerStageDescriptorSampledImages, 1u << 20u);
    limits.maxPerStageDescriptorStorageImages = std::max(limits.maxPerStageDescriptorStorageImages, 1u << 20u);
    limits.maxPerStageDescriptorSamplers = std::max(limits.maxPerStageDescriptorSamplers, 1u << 20u);
    limits.maxDescriptorSetStorageBuffers = std::max(limits.maxDescriptorSetStorageBuffers, 1u << 20u);
    limits.maxDescriptorSetSampledImages = std::max(limits.maxDescriptorSetSampledImages, 1u << 20u);
    limits.maxDescriptorSetStorageImages = std::max(limits.maxDescriptorSetStorageImages, 1u << 20u);
    limits.maxDescriptorSetSamplers = std::max(limits.maxDescriptorSetSamplers, 1u << 20u);
    limits.maxSamplerAnisotropy = std::max(limits.maxSamplerAnisotropy, 16.0f);
    limits.maxSamplerLodBias = std::max(limits.maxSamplerLodBias, 16.0f);
    limits.maxImageDimension2D = std::max(limits.maxImageDimension2D, 16384u);
    limits.maxImageDimension3D = std::max(limits.maxImageDimension3D, 2048u);
    limits.maxImageArrayLayers = std::max(limits.maxImageArrayLayers, 2048u);
    context.samplerAnisotropy = true;
    context.samplerFilterMinmax = true;
    // Indexed lists with primitive restart expand on the CPU (MetalDraw.mm), as PrimitiveListRestart says.
    context.primitiveListRestart = true;
    placeholderImages = std::make_unique<PlaceholderImages>(context);
    context.placeholderImages = placeholderImages.get();
    upstreamSamplers = std::make_unique<SamplerCache>();
    context.samplerCache = upstreamSamplers.get();
    textureCache = std::make_unique<TextureCache>(context);
    context.textureCache = textureCache.get();
}

void ImageCache::Teardown() noexcept {
    if (imageContext != nullptr) {
        // Every ShaderResources and cached texture of the device goes while the image cache's
        // residency set is still on the queue (the recorder is idle: the device tore it down).
        try {
            SharedResourceCache().Clear();
            ClearCachedTextures(imageContext->device);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[metal-textures] teardown: %s\n", error.what());
        }
        textureCache.reset();
        upstreamSamplers.reset();
        placeholderImages.reset();
        detiler.reset();
        imageContext.reset();
    }
    {
        auto& registry = Caches();
        std::lock_guard lock(registry.mutex);
        std::erase_if(registry.caches, [&](const auto& entry) { return entry.second == this; });
    }
    std::lock_guard lock(mutex);
    if (tornDown) return;
    tornDown = true;
    if (queue != nil && residency != nil) ::AgcDriver::Metal::Residency::Detach(queue, residency);
    ::AgcDriver::Metal::Residency::RemoveAll(residency);
    ::AgcDriver::Metal::Residency::Commit(residency);
    residentCounts.clear();
}

void ImageCache::MakeResident(std::span<const id<MTLTexture>> textures) {
    std::lock_guard lock(mutex);
    bool changed = false;
    for (id<MTLTexture> texture : textures) {
        if (texture == nil) continue;
        auto& count = residentCounts[(__bridge void*)texture];
        if (count++ == 0u) {
            ::AgcDriver::Metal::Residency::Add(residency, texture);
            changed = true;
        }
    }
    if (changed) ::AgcDriver::Metal::Residency::Commit(residency);
}

void ImageCache::Release(std::span<const id<MTLTexture>> textures) {
    std::lock_guard lock(mutex);
    if (tornDown) return;
    bool changed = false;
    for (id<MTLTexture> texture : textures) {
        if (texture == nil) continue;
        const auto found = residentCounts.find((__bridge void*)texture);
        if (found == residentCounts.end()) continue;
        if (--found->second == 0u) {
            ::AgcDriver::Metal::Residency::Remove(residency, texture);
            residentCounts.erase(found);
            changed = true;
        }
    }
    if (changed) ::AgcDriver::Metal::Residency::Commit(residency);
}

namespace {

bool ResourceCacheEnabled() {
    static const bool noResourceCache = std::getenv("APS5_NO_RESOURCE_CACHE") != nullptr;
    static const bool noTextureCache = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    return !noResourceCache && !noTextureCache;
}

bool TemplateDataRefresh() {
    static const bool enabled = std::getenv("APS5_NO_TEMPLATE_DATA_REFRESH") == nullptr;
    return enabled;
}

// VulkanDevice.cpp's DispatchContentKey: the build's content plus the device.
ResourceCache::Key DispatchContentKey(const CompiledShader& shader, VkDevice device) {
    auto key = ShaderResources::ContentKey(shader, !TemplateDataRefresh());
    const auto deviceHandle = reinterpret_cast<std::uint64_t>(device);
    key.push_back(static_cast<std::uint32_t>(deviceHandle));
    key.push_back(static_cast<std::uint32_t>(deviceHandle >> 32u));
    return key;
}

bool HasImages(const ShaderRecompiler::RecompileResult& shader) {
    // MslAbi Version 15: image and sampler tables (and their map) come from the build too.
    return std::any_of(shader.bindings.begin(), shader.bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) {
        using Role = ShaderRecompiler::DescriptorRole;
        return binding.role == Role::GuestImages || binding.role == Role::GuestSamplers || binding.role == Role::ImageTable || binding.role == Role::ImageStorageTable || binding.role == Role::SamplerTable || binding.role == Role::ImageTableMap;
    });
}

}

// `settle` served the removed bring-up uploads; ShaderResources settles through GuestMemory.
ImageBindings::ImageBindings(ImageCache& cache, std::function<void(std::uint64_t, std::uint64_t)>) : cache(cache) {}

ImageBindings::~ImageBindings() = default;

void ImageBindings::Prepare(const ShaderRecompiler::RecompileResult& shader) {
    // A draw prepares each stage in turn: every stage writes its entries from its own build (the
    // previous stage's stays in `builds` for Commit).
    resources = nullptr;
    if (!HasImages(shader)) return;
    const auto* context = cache.ImageContext();
    if (context == nullptr) FailD5("a dispatch binds guest images before the image cache is attached to the device's Context");
    // Upstream's dispatch build (VulkanDevice::dispatch): the resource cache entry of the content
    // when it revalidates, else a new build, cached when reusable.
    const CompiledShader compiled{ShaderRecompiler::ShaderStage::Compute, &shader, 0u};
    const bool cacheable = ResourceCacheEnabled() && shader.PipelineVariantId() != 0;
    ResourceCache::Key key;
    if (cacheable) {
        key = DispatchContentKey(compiled, context->device);
        if (auto cached = SharedResourceCache().Find(key)) {
            if (cached->Revalidate(compiled)) {
                resources = std::move(cached);
            } else {
                SharedResourceCache().Remove(key, cached.get());
                if (auto* recorder = Recorder::Active()) recorder->Keep(std::move(cached));
            }
        }
    }
    if (resources == nullptr) {
        resources = std::make_shared<ShaderResources>(*context, compiled, ShaderResources::MetalImagesOnly{});
        if (cacheable && resources->Reusable()) {
            std::vector<std::shared_ptr<ShaderResources>> evicted;
            SharedResourceCache().Insert(key, resources, &evicted);
            if (auto* recorder = Recorder::Active()) {
                for (auto& object : evicted) recorder->Keep(std::move(object));
            }
        }
    }
    if (std::find(builds.begin(), builds.end(), resources) == builds.end()) builds.push_back(resources);
}

void ImageBindings::Write(const ShaderRecompiler::DescriptorBinding& binding, std::span<std::uint64_t> entries) {
    if (committed) FailD5("image bindings written after Commit");
    if (resources == nullptr) FailD5("guest image or sampler entries written without Prepare's ShaderResources build");
    resources->WriteTableEntries(binding, entries);
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ImageBindings::TableRanges() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& build : builds) ranges.insert(ranges.end(), build->ImageTableRanges().begin(), build->ImageTableRanges().end());
    return ranges;
}

void ImageBindings::Commit(Recorder& recorder) {
    if (committed) return;
    committed = true;
    // ShaderResources::MarkGpuWrites marks the written storage images dirty (pending: stored when a
    // reader needs their memory) and notes their ranges; the batch keeps each stage's build.
    for (auto& build : builds) {
        build->MarkGpuWrites(recorder);
        recorder.Keep(build);
    }
}

}
