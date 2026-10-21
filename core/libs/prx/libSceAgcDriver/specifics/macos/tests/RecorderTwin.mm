// Metal twin of tests/Recorder.cpp and tests/ResidentPresent.cpp (agc_driver_recorder_tests), swapped in by
// twins/T1-tests.cmake (tests/HostImportLifetime.cpp has a twin of its own, HostImportLifetimeTwin.mm). Upstream, each registered mode builds its own Vulkan instance and drives Vulkan objects
// through the recorder; on Metal a mode runs the Metal port that a work package registered (RecorderTwins.h),
// skips with the allowlisted reason when the behaviour exists only in Vulkan (graphics pipeline libraries,
// descriptor pools), and otherwise skips naming the package that owns its port, which tools/check_skips.py
// --job macos reports as an unexpected skip until the port exists.

#include "RecorderTwins.h"
#include <cstdio>
#include <map>
#include <string>
#include <string_view>

namespace AgcDriver::MetalTwins {

namespace {

std::map<std::string, RecorderMode>& Registry() {
    static std::map<std::string, RecorderMode> modes;
    return modes;
}

struct Mode {
    const char* flag;
    const char* test;
    const char* owner;
    const char* vulkanOnly;
};

constexpr Mode Modes[] = {
    {"--pipeline-cache-only", "agc_pipeline_cache", "D5 (Metal pipeline archive, agc_metal_pipeline_cache)", nullptr},
    {"--pipeline-library-only", "agc_pipeline_library, agc_pipeline_library_off, agc_pipeline_library_fast_only", nullptr,
     "skipped, graphics pipeline libraries are unavailable or disabled: Metal has none"},
    {"--descriptor-pool-only", "agc_descriptor_pool_recycling", nullptr,
     "skipped, the device has no Vulkan descriptor pools: Metal descriptor tables are written into the per-batch upload ring"},
    {"--unchanged-cpu-stamps-only", "agc_unchanged_cpu_stamps", "D3 (guest memory imports and write tracking)", nullptr},
    {"--settled-baseline-only", "agc_storage_baseline_settled", "D3 (guest memory imports and write tracking)", nullptr},
    {"--host-import-unmap-only", "agc_host_import_unmap", "D3 (host imports)", nullptr},
    {"--unit-shadow-only", "agc_unit_shadow", "D3 (unit shadows)", nullptr},
    // Upstream 0f2683cb7 modes: --integer-clear-only, --written-buffer-order-only, --parked-import-unmap-only and
    // --host-import-retirement-only are ported in UpsyncRecorderModes.mm, --copied-direct-only in
    // CopiedDirectOrderingTwin.mm.
    // Upstream 0d078f75d3 (UPSYNC3 round 3) modes: --device-selection-only and --sampled-storage-base-level-usage-only
    // are ported in Upsync3RecorderModesD.mm, --host-import-lifetime-only and --host-import-mutation-race-only in
    // HostImportLifetimeTwin.mm (the twin of tests/HostImportLifetime.cpp).
    // Upstream 6b6c701636 (UPSYNC5) modes: --queued-write-groups-only gains QueuedWriteOrder's checks (eef987ae7b,
    // Upsync3RecorderModesA.mm) and --idle-publish-wait-only (db3af40970) is ported in D3RecorderModes.mm. The DCC
    // clear-key hand-over (92063916d2, b176274da1) belongs to the Texture twin's port.
    {"--clear-hand-over-only", "agc_storage_clear_hand_over", "UPSYNC5 Texture (DCC clear-key hand-over, 92063916d2/b176274da1)", nullptr},
    {"--draw-descriptors-only", "agc_draw_descriptors", nullptr,
     "skipped, the device writes no Vulkan descriptor sets: the checks count vkUpdateDescriptorSets writes per set, and Metal descriptor tables are written into the per-batch upload ring"},
    // Modes upstream registers only under if(WIN32) (Windows host import settles, write-watch probes and mappings);
    // listed so that a manual run says why instead of failing as an unknown mode.
    {"--import-watch-other-family-only", "agc_import_watch_other_family (Windows only)", nullptr,
     "skipped, a Windows-only mode: import settles on another queue family exist only for Windows host imports"},
    {"--deferred-settle-only", "agc_import_settle_deferred (Windows only)", nullptr,
     "skipped, a Windows-only mode: deferred host import settles exist only for Windows host imports (tests/Recorder.cpp #ifdef _WIN32)"},
    {"--deferred-settle-other-family-only", "agc_import_settle_deferred_other_family (Windows only)", nullptr,
     "skipped, a Windows-only mode: deferred host import settles exist only for Windows host imports (tests/Recorder.cpp #ifdef _WIN32)"},
    {"--settle-first-use-only", "agc_import_settle_first_use (Windows only)", nullptr,
     "skipped, a Windows-only mode: host import settles exist only for Windows host imports (tests/Recorder.cpp #ifdef _WIN32)"},
    {"--settle-earlier-import-only", "agc_import_settle_earlier_import (Windows only)", nullptr,
     "skipped, a Windows-only mode: host import settles exist only for Windows host imports (tests/Recorder.cpp #ifdef _WIN32)"},
    {"--settle-retire-only", "agc_import_settle_retire (Windows only)", nullptr,
     "skipped, a Windows-only mode: host import settles exist only for Windows host imports (tests/Recorder.cpp #ifdef _WIN32)"},
    {"--settle-storage-image-only", "agc_import_settle_storage_image (Windows only)", nullptr,
     "skipped, a Windows-only mode: host import settles exist only for Windows host imports (tests/Recorder.cpp #ifdef _WIN32)"},
    {"--storage-cpu-snapshot-only", "agc_storage_cpu_snapshot (Windows only)", nullptr,
     "skipped, a Windows-only mode (upstream registers agc_storage_cpu_snapshot under if(WIN32))"},
    {"--host-import-second-mapping-only", "agc_host_import_second_mapping (Windows only)", nullptr,
     "skipped, a Windows-only mode (upstream registers agc_host_import_second_mapping under if(WIN32))"},
    {"--reimport-bench", "agc_reimport_bench (Windows only)", nullptr,
     "skipped, a Windows-only mode (upstream registers agc_reimport_bench under if(WIN32))"},
    {"--host-import-cut-only", "agc_host_import_cut_lifetime (Windows only)", nullptr,
     "skipped, a Windows-only mode (upstream registers agc_host_import_cut_lifetime under if(WIN32), a9b14023dd)"},
    {"--partial-watch-restoration-only", "agc_partial_watch_restoration (Windows only)", nullptr,
     "skipped, a Windows-only mode (upstream registers agc_partial_watch_restoration under if(WIN32), 823c6a55e1)"},
    {"--host-import-lifetime-other-family-only", "agc_driver_host_import_lifetime_other_family (Windows only)", nullptr,
     "skipped, a Windows-only mode: host import lifetimes on another queue family exist only for Windows host imports (8a1e737182)"},
};

}

bool RegisterRecorderMode(const char* flag, RecorderMode run) {
    return Registry().emplace(flag, run).second;
}

}

int main(int argc, char** argv) {
    using namespace AgcDriver::MetalTwins;
    const std::string flag = argc == 2 ? argv[1] : "";
    if (const auto found = Registry().find(flag); found != Registry().end()) return found->second();
    for (const auto& mode : Modes) {
        if (flag != mode.flag) continue;
        if (mode.vulkanOnly != nullptr) {
            std::puts(mode.vulkanOnly);
        } else {
            std::printf("skipped, the Metal twin of %s (%s) is not written yet: work package %s\n", mode.flag, mode.test, mode.owner);
        }
        return 77;
    }
    std::fprintf(stderr, "agc_driver_recorder_tests on Metal runs one mode per process; known modes:");
    for (const auto& mode : Modes) std::fprintf(stderr, " %s", mode.flag);
    std::fprintf(stderr, "\n");
    return 2;
}
