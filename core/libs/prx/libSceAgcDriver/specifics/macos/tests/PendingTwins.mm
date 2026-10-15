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
