#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALPIPELINECACHE_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALPIPELINECACHE_H

// Work package D5 (docs/mac/M2_DESIGN.md §7): the Metal device's shader libraries and pipeline
// states. One cache per device owns the device's MTL4Compiler (created with a binaries-only
// pipeline data set serializer), keeps every library and pipeline in memory, looks pipelines up in
// the per-title MTL4Archive persisted under <ShaderCacheDirectory>/metal before compiling them, and
// compiles in the background on a small pool of utility-QoS workers so a submission only waits for
// the one pipeline it needs. "The second run compiles 0 pipelines" = Statistics().pipelinesCompiled
// stays 0 when every pipeline came from the archive (archive lookups cannot compile).
//
// Environment:
//   APS5_METAL_PIPELINE_CACHE=<dir>  archive directory (default <ShaderCacheDirectory>/metal; none
//                                    when the shader cache is disabled)
//   APS5_NO_PIPELINE_CACHE=1         no archive (memory cache only)
//   APS5_PROFILE_PIPELINES=1         one line per compiled pipeline and a summary at destruction
// Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "Recompiler.hpp"
#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics::Metal {

struct PipelineCacheStatistics {
    // MSL front-end compilations (MTL4Compiler newLibraryWithDescriptor) and in-memory library hits.
    std::uint64_t librariesCompiled = 0;
    std::uint64_t libraryHits = 0;
    // Pipelines the compiler built (archive misses) and pipelines served by the persisted archive.
    std::uint64_t pipelinesCompiled = 0;
    std::uint64_t archiveHits = 0;
    // Requests served by a pipeline already in memory (or by a finished background task).
    std::uint64_t memoryHits = 0;
    // Background tasks started by Prepare*, and requests that waited for an unfinished task.
    std::uint64_t backgroundTasks = 0;
    std::uint64_t waits = 0;
    double waitMilliseconds = 0.0;
    double compileMilliseconds = 0.0;
    // Archive persistence.
    bool archiveLoaded = false;
    std::uint64_t archiveSaves = 0;
    std::string archivePath;
};

// One shader function of a pipeline: an emitted MSL library, the values of the function constants
// it declares (RecompileResult::specialization) and the preprocessor macros of its library (type
// changing values: colour export numeric classes, dual-source blending, M2_DESIGN.md §4).
struct PipelineFunction {
    std::shared_ptr<const ShaderRecompiler::MslShader> shader;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> constants;
    std::vector<std::pair<std::string, std::string>> macros;

    static PipelineFunction From(const ShaderRecompiler::RecompileResult& result);
};

// Blend state of one colour attachment (MTLRenderPipelineColorAttachmentDescriptor).
struct ColorAttachmentState {
    MTLPixelFormat format = MTLPixelFormatInvalid;
    bool blending = false;
    MTLBlendFactor sourceRgb = MTLBlendFactorOne;
    MTLBlendFactor destinationRgb = MTLBlendFactorZero;
    MTLBlendOperation rgbOperation = MTLBlendOperationAdd;
    MTLBlendFactor sourceAlpha = MTLBlendFactorOne;
    MTLBlendFactor destinationAlpha = MTLBlendFactorZero;
    MTLBlendOperation alphaOperation = MTLBlendOperationAdd;
    MTLColorWriteMask writeMask = MTLColorWriteMaskAll;
};

// Everything a Metal 4 render pipeline state is made from (the draws package D6 fills it from the
// guest state; dynamic state, viewports, depth-stencil states and the like are not part of it).
struct RenderPipelineRequest {
    PipelineFunction vertex;
    // No fragment function: rasterization still runs (depth-only passes) unless `rasterization`.
    std::optional<PipelineFunction> fragment;
    std::array<ColorAttachmentState, 8> colors{};
    MTLPixelFormat depthFormat = MTLPixelFormatInvalid;
    MTLPixelFormat stencilFormat = MTLPixelFormatInvalid;
    std::uint32_t rasterSampleCount = 1;
    bool alphaToCoverage = false;
    bool alphaToOne = false;
    bool rasterization = true;
    MTLPrimitiveTopologyClass topology = MTLPrimitiveTopologyClassUnspecified;
    // Vertex amplification and raster order groups are not used by M2.
};

class PipelineCache {
public:
    // `directory`: where the archive lives (empty: the default from the environment, see above).
    explicit PipelineCache(id<MTLDevice> device, std::filesystem::path directory = {});
    ~PipelineCache();
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    // The device's compiler (created here, with the archive serializer attached).
    id<MTL4Compiler> Compiler() const { return compiler; }

    // The library of an emitted shader (compiled once per source + macro set).
    id<MTLLibrary> Library(const ShaderRecompiler::MslShader& shader, const std::vector<std::pair<std::string, std::string>>& macros = {});

    // The compute pipeline of a kernel with its function constants; waits for a background task
    // that builds it, compiles it on this thread otherwise. Throws std::runtime_error naming D5.
    id<MTLComputePipelineState> Compute(const ShaderRecompiler::RecompileResult& shader);
    id<MTLComputePipelineState> Compute(const PipelineFunction& function);
    // Starts building the pipeline on the background workers (no-op when known or in flight).
    void PrepareCompute(const ShaderRecompiler::RecompileResult& shader);

    id<MTLRenderPipelineState> Render(const RenderPipelineRequest& request);
    void PrepareRender(const RenderPipelineRequest& request);

    // Writes the archive when pipelines were compiled since the last save (atomically renamed).
    // Called every SaveInterval new pipelines and at destruction. Never throws.
    void Save();
    // Waits until no background task runs.
    void Drain();

    PipelineCacheStatistics Statistics() const;

    static constexpr std::uint64_t SaveInterval = 64;
    // The cache key of a request (exposed for tests).
    static std::string ComputeKey(const PipelineFunction& function);
    static std::string RenderKey(const RenderPipelineRequest& request);

private:
    struct Entry;
    std::shared_ptr<Entry> entryFor(const std::string& key, bool* created);
    id<MTLComputePipelineState> buildCompute(const PipelineFunction& function);
    id<MTLRenderPipelineState> buildRender(const RenderPipelineRequest& request);
    MTL4FunctionDescriptor* functionDescriptor(const PipelineFunction& function);
    void submit(std::function<void()> task);
    void worker();
    void recaptureHits();

    id<MTLDevice> device = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTL4PipelineDataSetSerializer> serializer = nil;
    id<MTL4Archive> archive = nil;
    std::filesystem::path archivePath;

    mutable std::mutex mutex;
    std::unordered_map<std::string, id<MTLLibrary>> libraries;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries;
    // Pipelines served by the archive this run: the serializer only captures pipelines its compiler
    // builds, so before a save they are rebuilt through the compiler (archive-accelerated) to keep
    // the new archive a superset of the old one.
    std::vector<std::function<void()>> archiveRecaptures;
    std::uint64_t unsaved = 0;
    PipelineCacheStatistics statistics;

    std::mutex queueMutex;
    std::condition_variable queueWake;
    std::condition_variable queueIdle;
    std::deque<std::function<void()>> queue;
    std::size_t running = 0;
    bool stopping = false;
    std::vector<std::thread> workers;
};

}

#endif
