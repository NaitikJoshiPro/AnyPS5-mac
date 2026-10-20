// Metal twin of the execution tests that drive Vulkan objects directly (twins/T1-tests.cmake swaps it for each of
// their sources). Each executable is told apart by its name: a Vulkan-only behaviour skips with the reason
// tools/mac/m2_skip_allowlist.txt accepts; a test of a driver feature the Metal driver does not have yet skips naming
// the feature and its docs/dev/TechnicalDebt.md anchor (not allowlisted: tools/check_skips.py --job macos reports it
// until the feature exists); any other skips naming the work package that owns its Metal port. The package that
// ports one moves its source pair from twins/T1-tests.cmake (or twins/UPSYNC3-tests.cmake) to its own twin list.

#include <cstdio>
#include <cstring>
#include <string>

namespace {

struct Pending {
    const char* executable;
    const char* tests;
    const char* owner;
    const char* vulkanOnly;
    const char* metalGap = nullptr;
};

constexpr Pending Tests[] = {
    {"agc_driver_pipeline_optimized_readback_tests", "agc_driver_pipeline_optimized_readback_*", nullptr,
     "skipped, graphics pipeline libraries are unavailable or disabled: Metal has none, so there is no link-time optimized pipeline to read back"},
    // UPSYNC3 round 3 (upstream 0d078f75d3). Vulkan-only behaviours (allowlisted):
    {"agc_driver_dcc_key_presync_tests", "agc_driver_dcc_key_presync", nullptr,
     "skipped, the Metal device has no stage-A dispatch preparation: it binds dispatch buffers by address at the record, so no DCC key writer is pre-synced"},
    {"agc_driver_resource_cache_growth_tests", "agc_driver_resource_cache_growth", nullptr,
     "skipped, buffer-only dispatches never enter the resource cache on Metal: it binds dispatch buffers by address at the record"},
    {"agc_driver_pipeline_journal_tests", "agc_driver_pipeline_journal", nullptr,
     "skipped, the pipeline journal is Vulkan-only: the Metal pipeline cache's MTL4Archive and prewarm thread replay pipelines in its place"},
    {"agc_command_batch_failure_tests", "agc_command_batch_failure", nullptr,
     "skipped, the device has no Vulkan command batches: RunCommandBatchFailureTests injects vkEndCommandBuffer/vkQueueSubmit/fence failures through Context.deviceProc, while a Metal CommandBatch records MTL4 command buffers that never call them"},
    {"agc_descriptor_cache_failure_tests", "agc_descriptor_cache_failure", nullptr,
     "skipped, the device has no Vulkan descriptor pools: DescriptorCacheFailure injects vkCreateDescriptorSetLayout/vkCreateDescriptorPool failures, while Metal descriptor layouts and pools are tokens that cannot fail"},
    // UPSYNC4 (upstream 661f8b7171). Vulkan-only behaviours (allowlisted):
    {"agc_driver_indirect_arguments_pending_producer_tests", "agc_driver_indirect_arguments_pending_producer*", nullptr,
     "skipped, Recorder::HoldNextSubmit holds a Vulkan timeline-semaphore submit: Metal command buffers commit through MTL4 queues with no held submit"},
    // Metal driver features not written yet (not allowlisted):
    {"agc_driver_address_passes_tests", "agc_driver_address_passes*", nullptr, nullptr,
     "skipped, the Metal draw path does not report render pass totals (DrawPassTotals): TechnicalDebt metal-draw-pass-totals"},
    {"agc_driver_rect_list_indirect_tests", "agc_driver_rect_list_indirect", nullptr, nullptr,
     "skipped, indirect draws read their records on the CPU on Metal (no GPU indirect path yet): TechnicalDebt metal-d6-indirect"},
};

}

int main(int, char** argv) {
    const char* slash = std::strrchr(argv[0], '/');
    const std::string executable = slash != nullptr ? slash + 1 : argv[0];
    for (const auto& test : Tests) {
        if (executable != test.executable) continue;
        if (test.vulkanOnly != nullptr) {
            std::puts(test.vulkanOnly);
        } else if (test.metalGap != nullptr) {
            std::puts(test.metalGap);
        } else {
            std::printf("skipped, the Metal twin of %s is not written yet: work package %s\n", test.tests, test.owner);
        }
        return 77;
    }
    std::fprintf(stderr, "PendingTwins.mm does not know the executable %s\n", executable.c_str());
    return 1;
}
