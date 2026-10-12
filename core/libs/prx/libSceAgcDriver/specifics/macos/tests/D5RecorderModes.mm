// Work package D5: the Metal port of tests/Recorder.cpp's --pipeline-cache-only mode (agc_pipeline_cache).
// Upstream checks that a VkPipelineCache persists across devices; here two pipeline caches in one
// process share an archive directory: the second compiles 0 pipelines and computes the same results.
// agc_metal_pipeline_cache runs the same workload in two processes.
#include "MetalPipelineCacheChecks.h"
#include "RecorderTwins.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>

namespace {

using namespace AgcDriver::Graphics;
using namespace AgcDriver::Graphics::MetalPipelineCacheChecks;

int PipelineCacheOnly() {
    @autoreleasepool {
        try {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
                std::printf("skipped, the machine has no Metal 4 device\n");
                return 77;
            }
            char pattern[] = "/tmp/aps5-pipeline-cache-XXXXXX";
            const char* made = mkdtemp(pattern);
            if (made == nullptr) return Fail("cannot create a temporary directory");
            const std::filesystem::path directory(made);
            std::vector<std::uint32_t> first;
            std::vector<std::uint32_t> second;
            Metal::PipelineCacheStatistics before;
            Metal::PipelineCacheStatistics after;
            std::string failure;
            {
                Metal::PipelineCache cache(device, directory);
                failure = BuildWorkload(device, cache, &first);
                before = cache.Statistics();
            }
            if (failure.empty()) {
                Metal::PipelineCache cache(device, directory);
                failure = BuildWorkload(device, cache, &second);
                after = cache.Statistics();
            }
            std::error_code error;
            std::filesystem::remove_all(directory, error);
            if (!failure.empty()) return Fail(failure);
            if (ShaderValidation()) {
                if (!after.archivePath.empty() || after.archiveDisabled.empty() || after.pipelinesCompiled != WorkloadPipelines) return Fail("the archive must be disabled under MTL_SHADER_VALIDATION");
                if (first != second) return Fail("the two caches' pipelines computed different results");
                std::printf("PASS: agc_pipeline_cache (shader validation: archive disabled, workload correct)\n");
                return 0;
            }
            if (before.pipelinesCompiled != WorkloadPipelines) return Fail("the first cache compiled " + std::to_string(before.pipelinesCompiled) + " pipelines");
            if (!after.archiveLoaded || after.pipelinesCompiled != 0u || after.archiveHits != WorkloadPipelines) {
                return Fail("the second cache compiled " + std::to_string(after.pipelinesCompiled) + " pipelines with " + std::to_string(after.archiveHits) + " archive hits");
            }
            if (first != second) return Fail("the two caches' pipelines computed different results");
            std::printf("PASS: agc_pipeline_cache (Metal archive: 0 pipelines compiled on reuse)\n");
            return 0;
        } catch (const std::exception& error) {
            return Fail(error.what());
        }
    }
}

const bool registered = AgcDriver::MetalTwins::RegisterRecorderMode("--pipeline-cache-only", &PipelineCacheOnly);

}
