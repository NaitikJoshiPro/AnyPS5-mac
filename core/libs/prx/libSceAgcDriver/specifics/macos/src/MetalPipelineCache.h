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
//   MTL_SHADER_VALIDATION=1          no archive (instrumented binaries do not reload on 27.0.1)
//   APS5_METAL_UNIQUE_LIBRARIES=1    measurement aid: library sources made unique per process (cold)
// and the library store's (MetalLibraryStore.h, DECISIONS D55): every MSL library source is kept
// under <dir>/libraries and compiled again on a background thread when the next process creates
// the device, so libraries reach the system compiler cache (and this cache) before they are used.
// Objective-C++ only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalLibraryStore.h"
#include "Recompiler.hpp"
#include <array>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
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
    // Pipelines replayed into the serializer before saves (D51) and the time the replays took.
    std::uint64_t archiveReplays = 0;
    double replayMilliseconds = 0.0;
    std::string archivePath;
    // Why there is no archive although a directory was given (empty: there is one, or none asked).
    std::string archiveDisabled;
    // The library store (D55): libraries read from it at startup, compiled ahead of use by the
    // prewarm thread, and requests that found their library compiled or compiling by it.
    std::string libraryStorePath;
    std::uint64_t librariesStored = 0;
    std::uint64_t librariesPrewarmed = 0;
    std::uint64_t prewarmHits = 0;
    bool prewarmFinished = false;
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
    // `vertex` is an MSL mesh function: an MTL4MeshRenderPipelineDescriptor (MetalMesh.mm, B6),
    // `meshThreads` its threads per mesh threadgroup (Reflection::threadsPerThreadgroup).
    bool mesh = false;
    std::uint32_t meshThreads = 0;
    // Tessellated draws (DECISIONS D24, MetalTessellation.h): the object function (local + hull
    // programs) of the mesh pipeline, its threads per object threadgroup, and the most mesh
    // threadgroups one object threadgroup launches.
    std::optional<PipelineFunction> object;
    std::uint32_t objectThreads = 0;
    std::uint32_t meshGroups = 0;
    // Vertex amplification and raster order groups are not used by M2.
};

// A depth-stencil state (MTLDepthStencilDescriptor): no depth test is compare Always without
// writes; no stencil test is `stencil` false.
struct StencilFaceState {
    MTLCompareFunction compare = MTLCompareFunctionAlways;
    MTLStencilOperation fail = MTLStencilOperationKeep;
    MTLStencilOperation depthFail = MTLStencilOperationKeep;
    MTLStencilOperation pass = MTLStencilOperationKeep;
    std::uint32_t readMask = 0xffu;
    std::uint32_t writeMask = 0xffu;
    bool operator==(const StencilFaceState&) const = default;
};

struct DepthStencilRequest {
    MTLCompareFunction depthCompare = MTLCompareFunctionAlways;
    bool depthWrite = false;
    bool stencil = false;
    StencilFaceState front;
    StencilFaceState back;
    bool operator==(const DepthStencilRequest&) const = default;
};

// Guest compute pipelines the caches built (compiled or an archive hit; process-wide), and those
// built by a thread holding the GPU mutex (upstream 7fae4549e's VulkanDevice::ComputePipelineCounts:
// PrepareDispatch starts the build on the cache's workers, outside the mutex).
struct ComputePipelineCreation {
    std::uint64_t created = 0;
    std::uint64_t underGpuMutex = 0;
};
ComputePipelineCreation ComputePipelineCreationCounts();

class PipelineCache {
public:
    // `directory`: where the archive lives (empty: the default from the environment, see above).
    explicit PipelineCache(id<MTLDevice> device, std::filesystem::path directory = {});
    ~PipelineCache();
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    // The device's compiler (created here, with the archive serializer attached).
    id<MTL4Compiler> Compiler() const { return compiler; }

    // The library of an emitted shader (compiled once per source + macro set; a compile the prewarm
    // thread runs is waited for, not repeated).
    id<MTLLibrary> Library(const ShaderRecompiler::MslShader& shader, const std::vector<std::pair<std::string, std::string>>& macros = {});
    // Waits until the prewarm thread has compiled every stored library (tests).
    void WaitForPrewarm();
    // Libraries and prewarms are shared by every cache on one MTLDevice (process-wide); tests that
    // stand for a new process forget them.
    static void ForgetSharedLibraries();

    // The compute pipeline of a kernel with its function constants; waits for a background task
    // that builds it, compiles it on this thread otherwise. Throws std::runtime_error naming D5.
    id<MTLComputePipelineState> Compute(const ShaderRecompiler::RecompileResult& shader);
    id<MTLComputePipelineState> Compute(const PipelineFunction& function);
    // Starts building the pipeline on the background workers (no-op when known or in flight).
    void PrepareCompute(const ShaderRecompiler::RecompileResult& shader);

    id<MTLRenderPipelineState> Render(const RenderPipelineRequest& request);
    // The depth-stencil state of a request, made once per distinct request (thread-safe).
    id<MTLDepthStencilState> DepthStencil(const DepthStencilRequest& request);
    void PrepareRender(const RenderPipelineRequest& request);

    // Writes the archive when pipelines were compiled since the last save (atomically renamed).
    // The serializer's serializeAsArchiveAndFlushToURL empties it, so every save first replays the
    // pipelines an earlier save flushed (and the archive's hits) through the compiler with the
    // archives as lookup archives (no compile), keeping each written archive a superset of the
    // last. Requested after every new pipeline and run by the saver thread, coalesced and spaced
    // (SaveSpacing); also at destruction. Never throws.
    // `waitForPipelines` false (the saver thread; upstream 1d8acd1eda SaveUnlessBusy): when a
    // pipeline creation holds serializerGate, the save is skipped instead of queueing an exclusive
    // lock that would hold every later creation behind that compile; returns false and the saver
    // asks again after the spacing. True (destruction, tests) waits for the gate.
    bool Save(bool waitForPipelines = true);
    // Waits until no background task runs.
    void Drain();

    PipelineCacheStatistics Statistics() const;

    // A title that fail-stops (_Exit) runs no destructor, so the pipelines of its first frames,
    // the slow ones (PPSA99005's first draw compiles for 3 s against the guest's 2 s retirement
    // bound), only persist if they are saved as they are compiled: the first new pipeline is saved
    // at once, later ones at most SaveSpacing (or SaveCostFactor x the last save's duration, when
    // longer) after the previous save.
    static constexpr std::chrono::milliseconds SaveSpacing{1000};
    static constexpr std::uint32_t SaveCostFactor = 4;
    // The cache key of a request (exposed for tests).
    static std::string ComputeKey(const PipelineFunction& function);
    static std::string RenderKey(const RenderPipelineRequest& request);

private:
    struct Entry;
    struct LibraryTask;
    std::shared_ptr<Entry> entryFor(const std::string& key, bool* created);
    // Library(): `prewarm` compiles a stored library ahead of use (returns nil instead of throwing,
    // and skips sources another thread compiles).
    id<MTLLibrary> library(const std::string& source, const std::vector<std::pair<std::string, std::string>>& macros, bool prewarm);
    void prewarm(std::vector<StoredLibrary> stored);
    id<MTLComputePipelineState> buildCompute(const PipelineFunction& function);
    // Compute(): `guest` counts a pipeline built for a guest kernel (ComputePipelineCreationCounts).
    id<MTLComputePipelineState> compute(const PipelineFunction& function, bool guest);
    id<MTLRenderPipelineState> buildRender(const RenderPipelineRequest& request);
    id<MTLRenderPipelineState> buildMeshRender(const RenderPipelineRequest& request);
    MTL4FunctionDescriptor* functionDescriptor(const PipelineFunction& function);
    void submit(std::function<void()> task);
    void worker();
    void saver();
    void requestSave();
    // A pipeline the serializer holds (compiled now) or must get again before the next save (an
    // archive hit, or a pipeline an earlier save flushed); `replay` rebuilds it with lookup archives.
    void noteArchived(std::function<void(NSArray<id<MTL4Archive>>*)> replay, bool captured);
    id<MTLComputePipelineState> obtainCompute(const std::function<MTL4ComputePipelineDescriptor*()>& describe, const char* what);
    id<MTLRenderPipelineState> obtainRender(const std::function<MTL4PipelineDescriptor*()>& describe, const char* what);
    MTL4PipelineDescriptor* renderDescriptor(const RenderPipelineRequest& request);
    MTL4PipelineDescriptor* meshDescriptor(const RenderPipelineRequest& request);

    id<MTLDevice> device = nil;
    id<MTL4Compiler> compiler = nil;
    id<MTL4PipelineDataSetSerializer> serializer = nil;
    // The archive generations this cache looks pipelines up in, newest first (empty: none). Every
    // cache of a process loads the same set: the newest generation of the last run together with
    // the generations its other devices wrote (upstream 8c8cd86cf: a process holds the windowless
    // device next to its windowed replacement, and neither device's pipelines may be dropped).
    NSArray<id<MTL4Archive>>* archives = nil;
    std::filesystem::path archivePath;

    mutable std::mutex mutex;
    // Compiles capture their binaries into `serializer`; Save's serializeAsArchiveAndFlushToURL flushes
    // them. A flush during a compile's capture leaves that compile with released capture data (Metal
    // crashed in -[_MTLBinaryArchive newArchiverIdWithBinaryKey:...] -> dispatch_data_create_map,
    // ASTRO BOT, once saves became eager). Compiles hold this gate shared, the flush exclusively.
    std::shared_mutex serializerGate;
    std::unordered_map<std::string, id<MTLLibrary>> libraries;
    std::unique_ptr<MetalLibraryStore> store;
    std::thread prewarmThread;
    std::atomic<bool> prewarmStop{false};
    std::condition_variable prewarmDone;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries;
    std::vector<std::pair<DepthStencilRequest, id<MTLDepthStencilState>>> depthStencilStates;
    // Every pipeline this cache built or took from the archive (D51). The serializer only captures
    // what its compiler builds, and each save's flush empties it: entries not `captured` (archive
    // hits, and everything once flushed) are replayed through the compiler, with the loaded and the
    // last written archive as lookup archives, before the next save. Guarded by `mutex`; entries are
    // appended under the shared serializerGate so a flush never falls between capture and append.
    struct Archived {
        std::function<void(NSArray<id<MTL4Archive>>*)> replay;
        bool captured = false;
    };
    std::vector<Archived> archived;
    // The archive this process wrote last (opened from the file after the save), or nil.
    id<MTL4Archive> written = nil;
    std::uint64_t unsaved = 0;
    PipelineCacheStatistics statistics;

    std::mutex queueMutex;
    std::condition_variable queueWake;
    std::condition_variable queueIdle;
    std::deque<std::function<void()>> queue;
    std::size_t running = 0;
    bool stopping = false;
    std::vector<std::thread> workers;
    // The saver thread (requestSave/saver): one pending request at a time.
    std::mutex saverMutex;
    std::condition_variable saverWake;
    bool saveRequested = false;
    bool saverStopping = false;
    std::chrono::steady_clock::time_point nextSave{};
    std::thread saverThread;
};

}

#endif
