// Metal twin of tests/Recorder.cpp and tests/ResidentPresent.cpp (agc_driver_recorder_tests), swapped in by
// twins/T1-tests.cmake. Upstream, each registered mode builds its own Vulkan instance and drives Vulkan objects
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
