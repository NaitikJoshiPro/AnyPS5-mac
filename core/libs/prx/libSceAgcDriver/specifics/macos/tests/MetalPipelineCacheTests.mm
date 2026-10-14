// Work package D5: the Metal pipeline cache (src/MetalPipelineCache.h).
//   agc_metal_pipeline_cache            child processes share a fresh archive directory: the first
//                                       compiles every pipeline and saves the archive, the second
//                                       must compile 0 pipelines (all archive hits) and compute the
//                                       same results (MISSION §8, M2_DESIGN.md §7.6); a third adds two
//                                       pipelines with a save between them, and a fourth must find
//                                       the old and the new ones in the archive (each save flushes the
//                                       serializer, D51: the archive used to keep only the pipelines
//                                       since the previous save)
//   agc_metal_pipeline_cache_background background preparation, concurrent requests for one pipeline,
//                                       failure reporting, key separation
//   agc_metal_pipeline_cache_save_race  archive saves while other threads compile pipelines (the
//                                       serializer's flush raced in-flight captures: Metal crashed in
//                                       -[_MTLBinaryArchive newArchiverIdWithBinaryKey:...], ASTRO BOT)
// The `--pipeline-cache-only` mode of agc_driver_recorder_tests (agc_pipeline_cache) runs the
// two-cache check in one process (D5RecorderModes.mm).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalPipelineCache.h"
#include "MetalPipelineCacheChecks.h"

#include <spawn.h>
#include <atomic>
#include <sys/wait.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace {

using namespace AgcDriver::Graphics::Metal;
using namespace AgcDriver::Graphics::MetalPipelineCacheChecks;

int RunChild(const char* self, const std::string& mode, const std::filesystem::path& directory, std::string* output) {
    const auto log = directory / (mode + ".log");
    const std::string command = std::string("'") + self + "' " + mode + " '" + directory.string() + "' > '" + log.string() + "' 2>&1";
    const int status = std::system(command.c_str());
    if (FILE* file = std::fopen(log.c_str(), "r")) {
        char buffer[512];
        while (std::fgets(buffer, sizeof(buffer), file) != nullptr) output->append(buffer);
        std::fclose(file);
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

int Parent(const char* self) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::printf("skipped, the machine has no Metal 4 device\n");
        return 77;
    }
    char pattern[] = "/tmp/aps5-pipeline-cache-XXXXXX";
    const char* made = mkdtemp(pattern);
    if (made == nullptr) return Fail("cannot create a temporary directory");
    const std::filesystem::path directory(made);
    if (ShaderValidation()) {
        PipelineCache cache(device, directory);
        std::vector<std::uint32_t> results;
        const auto failure = BuildWorkload(device, cache, &results);
        const auto statistics = cache.Statistics();
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
        if (!failure.empty()) return Fail(failure);
        if (!statistics.archivePath.empty() || statistics.archiveDisabled.empty()) return Fail("the archive must be disabled under MTL_SHADER_VALIDATION");
        std::printf("PASS: agc_metal_pipeline_cache (shader validation: archive disabled, workload correct)\n");
        return 0;
    }
    std::string first;
    std::string second;
    std::string extend;
    std::string last;
    const int firstStatus = RunChild(self, "--child-first", directory, &first);
    const int secondStatus = firstStatus == 0 ? RunChild(self, "--child-second", directory, &second) : -1;
    const int extendStatus = secondStatus == 0 ? RunChild(self, "--child-extend", directory, &extend) : -1;
    const int finalStatus = extendStatus == 0 ? RunChild(self, "--child-final", directory, &last) : -1;
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    for (const auto* text : {&first, &second, &extend, &last}) std::fputs(text->c_str(), stdout);
    if (firstStatus != 0) return Fail("the first run failed (" + std::to_string(firstStatus) + ")");
    if (secondStatus != 0) return Fail("the second run failed (" + std::to_string(secondStatus) + ")");
    if (extendStatus != 0) return Fail("the extending run failed (" + std::to_string(extendStatus) + ")");
    if (finalStatus != 0) return Fail("the final run failed (" + std::to_string(finalStatus) + ")");
    // Both children print their results; they must agree.
    const auto results = [](const std::string& text) {
        const auto at = text.find("results ");
        return at == std::string::npos ? std::string() : text.substr(at, text.find('\n', at) - at);
    };
    if (results(first).empty() || results(first) != results(second)) return Fail("the two runs computed different results");
    std::printf("PASS: agc_metal_pipeline_cache\n");
    return 0;
}

int Child(bool second, const std::filesystem::path& directory) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    PipelineCacheStatistics statistics;
    std::vector<std::uint32_t> results;
    {
        PipelineCache cache(device, directory);
        if (const auto failure = BuildWorkload(device, cache, &results); !failure.empty()) return Fail(failure);
        statistics = cache.Statistics();
    }
    std::printf("%s run: %llu pipelines compiled, %llu archive hits, archive %s\n", second ? "second" : "first", static_cast<unsigned long long>(statistics.pipelinesCompiled), static_cast<unsigned long long>(statistics.archiveHits),
        statistics.archiveLoaded ? "loaded" : "absent");
    std::printf("results");
    for (const auto value : results) std::printf(" %08x", value);
    std::printf("\n");
    if (!second) {
        if (statistics.pipelinesCompiled != WorkloadPipelines) return Fail("the first run compiled " + std::to_string(statistics.pipelinesCompiled) + " pipelines, expected " + std::to_string(WorkloadPipelines));
        if (!std::filesystem::exists(statistics.archivePath)) return Fail("the first run saved no archive at " + statistics.archivePath);
        return 0;
    }
    if (!statistics.archiveLoaded) return Fail("the second run did not load the archive");
    if (statistics.pipelinesCompiled != 0u) return Fail("the second run compiled " + std::to_string(statistics.pipelinesCompiled) + " pipelines, expected 0");
    if (statistics.archiveHits != WorkloadPipelines) return Fail("the second run had " + std::to_string(statistics.archiveHits) + " archive hits, expected " + std::to_string(WorkloadPipelines));
    return 0;
}

// --child-extend: the workload from the archive plus two new pipelines, saved in between (the second
// save must replay the workload's hits, some made after the first save, and the first new
// pipeline). --child-final: all five from the archive.
int Extend(bool final, const std::filesystem::path& directory) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    PipelineCacheStatistics statistics;
    std::vector<std::uint32_t> results;
    {
        PipelineCache cache(device, directory);
        // One hit, a new pipeline, a save, then the other hits (looked up in the loaded archive
        // after this process wrote its own) and another new pipeline.
        if (cache.Compute(ComputeFunction(3u)) == nil) return Fail("no pipeline for multiplier 3");
        if (cache.Compute(ComputeFunction(7u)) == nil) return Fail("no pipeline for multiplier 7");
        if (!final) cache.Save();
        if (const auto failure = BuildWorkload(device, cache, &results); !failure.empty()) return Fail(failure);
        if (cache.Compute(ComputeFunction(9u)) == nil) return Fail("no pipeline for multiplier 9");
        statistics = cache.Statistics();
    }
    std::printf("%s run: %llu pipelines compiled, %llu archive hits, %llu saves, %llu replays (%.1f ms)\n", final ? "final" : "extend", static_cast<unsigned long long>(statistics.pipelinesCompiled),
        static_cast<unsigned long long>(statistics.archiveHits), static_cast<unsigned long long>(statistics.archiveSaves), static_cast<unsigned long long>(statistics.archiveReplays), statistics.replayMilliseconds);
    if (!statistics.archiveLoaded) return Fail("the archive was not loaded");
    if (!final) {
        if (statistics.pipelinesCompiled != 2u || statistics.archiveHits != WorkloadPipelines) return Fail("the extending run compiled " + std::to_string(statistics.pipelinesCompiled) + " pipelines with " + std::to_string(statistics.archiveHits) + " hits, expected 2 and " + std::to_string(WorkloadPipelines));
        return 0;
    }
    if (statistics.pipelinesCompiled != 0u || statistics.archiveHits != WorkloadPipelines + 2u) return Fail("the final run compiled " + std::to_string(statistics.pipelinesCompiled) + " pipelines with " + std::to_string(statistics.archiveHits) + " hits, expected 0 and " + std::to_string(WorkloadPipelines + 2u));
    return 0;
}

int Background() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::printf("skipped, the machine has no Metal 4 device\n");
        return 77;
    }
    PipelineCache cache(device, {});
    if (!cache.Statistics().archivePath.empty()) return Fail("an empty directory with APS5_NO_PIPELINE_CACHE must disable the archive");
    // Distinct function constants are distinct pipelines; equal ones share one.
    const auto a = ComputeFunction(1u);
    const auto b = ComputeFunction(2u);
    if (PipelineCache::ComputeKey(a) == PipelineCache::ComputeKey(b)) return Fail("function constants are not part of the compute key");
    if (PipelineCache::ComputeKey(a) != PipelineCache::ComputeKey(ComputeFunction(1u))) return Fail("equal compute requests have different keys");
    auto blended = RenderRequest();
    blended.colors[0].blending = true;
    blended.colors[0].sourceRgb = MTLBlendFactorSourceAlpha;
    if (PipelineCache::RenderKey(blended) == PipelineCache::RenderKey(RenderRequest())) return Fail("blend state is not part of the render key");

    // PrepareCompute + several concurrent Compute calls build the pipeline exactly once.
    ShaderRecompiler::RecompileResult result;
    result.msl = ShaderRecompiler::SharedMsl(*a.shader);
    for (const auto& [id, value] : a.constants) result.specialization.push_back({id, value});
    cache.PrepareCompute(result);
    std::vector<std::thread> threads;
    std::vector<id<MTLComputePipelineState>> pipelines(4);
    for (std::size_t index = 0; index < pipelines.size(); ++index) {
        threads.emplace_back([&, index] { pipelines[index] = cache.Compute(a); });
    }
    for (auto& thread : threads) thread.join();
    for (const auto& pipeline : pipelines) {
        if (pipeline == nil || pipeline != pipelines[0]) return Fail("concurrent requests got different pipelines");
    }
    cache.PrepareRender(RenderRequest());
    cache.Drain();
    if (cache.Render(RenderRequest()) == nil) return Fail("no render pipeline");
    auto statistics = cache.Statistics();
    if (statistics.pipelinesCompiled != 2u) return Fail("expected 2 compiled pipelines, got " + std::to_string(statistics.pipelinesCompiled));
    if (statistics.backgroundTasks != 2u) return Fail("expected 2 background tasks, got " + std::to_string(statistics.backgroundTasks));
    if (statistics.librariesCompiled != 3u) return Fail("expected 3 libraries (kernel, vertex, fragment), got " + std::to_string(statistics.librariesCompiled));

    // A shader that does not compile throws (naming the failure) and is retried, not cached.
    PipelineFunction broken = a;
    broken.shader = std::make_shared<const ShaderRecompiler::MslShader>(ShaderRecompiler::MslShader{"kernel void aps5_main( { this is not MSL", {}});
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            (void)cache.Compute(broken);
            return Fail("a broken shader compiled");
        } catch (const std::exception& error) {
            if (std::strstr(error.what(), "MSL compile failed") == nullptr) return Fail(std::string("unexpected failure text: ") + error.what());
        }
    }
    // Depth-stencil states: equal requests share one state, different ones do not.
    DepthStencilRequest less;
    less.depthCompare = MTLCompareFunctionLess;
    less.depthWrite = true;
    DepthStencilRequest stencil = less;
    stencil.stencil = true;
    stencil.front.pass = MTLStencilOperationReplace;
    if (cache.DepthStencil(less) == nil || cache.DepthStencil(less) != cache.DepthStencil(less) || cache.DepthStencil(stencil) == cache.DepthStencil(less)) return Fail("depth-stencil state cache");
    std::printf("PASS: agc_metal_pipeline_cache_background\n");
    return 0;
}

// Saves (serializeAsArchiveAndFlushToURL) racing compiles that capture into the same serializer. The
// cache's own eager saves (one per new pipeline) run on its background thread; one more thread saves
// in a loop. Without the archive (shader validation) the compiles still run and saves are no-ops.
int SaveRace() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
        std::printf("skipped, the machine has no Metal 4 device\n");
        return 77;
    }
    char pattern[] = "/tmp/aps5-pipeline-save-race-XXXXXX";
    const char* made = mkdtemp(pattern);
    if (made == nullptr) return Fail("cannot create a temporary directory");
    const std::filesystem::path directory(made);
    constexpr std::uint32_t Pipelines = 160;
    constexpr std::uint32_t Workers = 4;
    std::atomic<bool> failed{false};
    std::uint64_t compiled = 0;
    std::uint64_t saves = 0;
    {
        PipelineCache cache(device, directory);
        std::atomic<bool> done{false};
        std::thread saver([&] {
            while (!done.load()) cache.Save();
        });
        std::vector<std::thread> workers;
        for (std::uint32_t worker = 0; worker < Workers; ++worker) {
            workers.emplace_back([&, worker] {
                for (std::uint32_t index = worker; index < Pipelines; index += Workers) {
                    @autoreleasepool {
                        if (cache.Compute(ComputeFunction(1000u + index)) == nil) failed = true;
                    }
                }
            });
        }
        for (auto& thread : workers) thread.join();
        done = true;
        saver.join();
        cache.Drain();
        const auto statistics = cache.Statistics();
        compiled = statistics.pipelinesCompiled;
        saves = statistics.archiveSaves;
    }
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    if (failed.load()) return Fail("a pipeline compiled while saving came back nil");
    if (compiled != Pipelines) return Fail("expected " + std::to_string(Pipelines) + " compiled pipelines, got " + std::to_string(compiled));
    std::printf("PASS: agc_metal_pipeline_cache_save_race (%llu pipelines, %llu saves%s)\n", static_cast<unsigned long long>(compiled),
        static_cast<unsigned long long>(saves), ShaderValidation() ? ", archive disabled under shader validation" : "");
    return 0;
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc == 3 && std::strcmp(argv[1], "--child-first") == 0) return Child(false, argv[2]);
            if (argc == 3 && std::strcmp(argv[1], "--child-second") == 0) return Child(true, argv[2]);
            if (argc == 3 && std::strcmp(argv[1], "--child-extend") == 0) return Extend(false, argv[2]);
            if (argc == 3 && std::strcmp(argv[1], "--child-final") == 0) return Extend(true, argv[2]);
            if (argc == 2 && std::strcmp(argv[1], "--background") == 0) return Background();
            if (argc == 2 && std::strcmp(argv[1], "--save-race") == 0) return SaveRace();
            if (argc == 1) return Parent(argv[0]);
            return Fail("usage: agc_metal_pipeline_cache_tests [--background | --save-race]");
        } catch (const std::exception& error) {
            return Fail(error.what());
        }
    }
}
