// Metal twin of the execution tests that drive Vulkan objects directly (twins/T1-tests.cmake swaps it for each of
// their sources). Each executable is told apart by its name: a Vulkan-only behaviour skips with the reason
// tools/mac/m2_skip_allowlist.txt accepts; any other skips naming the work package that owns its Metal port. The
// package that ports one moves its source pair from twins/T1-tests.cmake to its own twin list.

#include <cstdio>
#include <cstring>
#include <string>

namespace {

struct Pending {
    const char* executable;
    const char* tests;
    const char* owner;
    const char* vulkanOnly;
};

constexpr Pending Tests[] = {
    {"agc_driver_pipeline_optimized_readback_tests", "agc_driver_pipeline_optimized_readback_*", nullptr,
     "skipped, graphics pipeline libraries are unavailable or disabled: Metal has none, so there is no link-time optimized pipeline to read back"},
    // Upstream v0.1.2 (0f2683cb7, D29) tests whose harness is a Vulkan instance of their own or a MockVulkan recorder.
    {"agc_driver_depth_storage_tests", "agc_driver_depth_storage (DepthSurfaceStorage / DepthSurfaceStoreBack on a Vulkan device of its own)",
     "UPSYNC follow-up (DepthSurface twin: storage access to depth planes)", nullptr},
    {"agc_driver_depth_plane_guest_layers_tests", "agc_driver_depth_plane_guest_layers (counts image/buffer copies through an intercepted vkGetDeviceProcAddr)",
     "UPSYNC follow-up (DepthSurface twin, a Metal::ImageProbe copy-count seam)", nullptr},
    {"agc_driver_sparse_buffer_holes_tests", "agc_driver_sparse_buffer_holes (GPU readback of a sparse guest heap copy through a SPIR-V pipeline; the CPU half runs on Metal as agc_sparse_buffer_holes)",
     "UPSYNC follow-up (GuestBufferMemory twin, a Metal compute harness for test-made programs)", nullptr},
    {"agc_host_import_retire_tests", "agc_host_import_retire (host import lifetime across a GPU hold, MockVulkan fences and vkDestroyBuffer)",
     "UPSYNC follow-up (GuestBufferMemory twin, a Metal import-lifetime and batch-completion seam)", nullptr},
};

}

int main(int, char** argv) {
    const char* slash = std::strrchr(argv[0], '/');
    const std::string executable = slash != nullptr ? slash + 1 : argv[0];
    for (const auto& test : Tests) {
        if (executable != test.executable) continue;
        if (test.vulkanOnly != nullptr) {
            std::puts(test.vulkanOnly);
        } else {
            std::printf("skipped, the Metal twin of %s is not written yet: work package %s\n", test.tests, test.owner);
        }
        return 77;
    }
    std::fprintf(stderr, "PendingTwins.mm does not know the executable %s\n", executable.c_str());
    return 1;
}
