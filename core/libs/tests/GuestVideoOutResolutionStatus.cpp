#include "SceTypes.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "tests/VideoOutTestEnvironment.hpp"
#include "prx/libSceVideoOut/include/DisplayMode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" {
int APS5_VABI sceVideoOutOpen(int userId, int busType, int index, const void* param);
int APS5_VABI sceVideoOutClose(int handle);
int APS5_VABI sceVideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status);
int APS5_VABI sceVideoOutGetResolutionStatus(int handle, VideoOutResolutionStatus* status);
}

static constexpr int SYSTEM_USER = 255;
static constexpr int MAIN_BUS = 0;
static constexpr int NEVER_OPENED_HANDLE = 2;

static void Require(bool value) { if (!value) std::abort(); }

static bool Rejects(int handle, VideoOutResolutionStatus* status) {
    try {
        sceVideoOutGetResolutionStatus(handle, status);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static bool ModeRejected(const char* text) {
    try {
        VideoOutDisplayModeFromText_nid_no_patch(text);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

// The console display mode (D61): APS5_DISPLAY_MODE parsing, then what the resolution queries report.
static void CheckDisplayModeParsing() {
    const auto none = VideoOutDisplayModeFromText_nid_no_patch(nullptr);
    Require(none.width == 1920 && none.height == 1080);
    const auto empty = VideoOutDisplayModeFromText_nid_no_patch("");
    Require(empty.width == 1920 && empty.height == 1080);
    const auto qhd = VideoOutDisplayModeFromText_nid_no_patch("1440p");
    Require(qhd.width == 2560 && qhd.height == 1440 && std::string(qhd.name) == "1440p");
    const auto uhd = VideoOutDisplayModeFromText_nid_no_patch("2160p");
    Require(uhd.width == 3840 && uhd.height == 2160);
    const auto alias = VideoOutDisplayModeFromText_nid_no_patch("4k");
    Require(alias.width == 3840 && alias.height == 2160);
    Require(ModeRejected("720p") && ModeRejected("1080") && ModeRejected("1920x1080") && ModeRejected(" 1080p"));
}

// argv[1], argv[2]: the width and height APS5_DISPLAY_MODE selects for this run (default 1920 1080); argv[3] "hdr"
// when APS5_HDR=1.
int main(int argc, char** argv) {
    CheckDisplayModeParsing();
    const std::uint32_t expectedWidth = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 1920u;
    const std::uint32_t expectedHeight = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 1080u;
    const auto& current = VideoOutCurrentDisplayMode_nid_no_patch();
    Require(current.width == expectedWidth && current.height == expectedHeight);
    VideoOutTestEnvironment environment;
    int handle = 0;
    try {
        handle = sceVideoOutOpen(SYSTEM_USER, MAIN_BUS, 0, nullptr);
    } catch (const std::runtime_error& error) {
        if (std::getenv("ANYPS5_REQUIRE_DISPLAY") != nullptr) throw;
        std::printf("skipped, no display or Vulkan device: %s\n", error.what());
        return 77;
    }
    Require(handle > 0);

    VideoOutResolutionStatus status;
    std::memset(&status, 0xff, sizeof(status));
    Require(sceVideoOutGetResolutionStatus(handle, &status) == 0);
    Require(status.fullWidth == expectedWidth && status.fullHeight == expectedHeight);
    Require(status.paneWidth == status.fullWidth && status.paneHeight == status.fullHeight);
    Require(status.screenSizeInInch == 0.0f && status.flags == 0 && status.reserved0 == 0);
    Require(status.reserved1[0] == 0 && status.reserved1[1] == 0 && status.reserved1[2] == 0);

    VideoOutOutputStatus output{};
    Require(sceVideoOutGetOutputStatus(handle, &output) == 0);
    Require(status.refreshRate == output.refreshRate);
    Require(output.resolution == (expectedHeight >= 2160u ? 2u : 1u));
    // The TV's dynamic range (APS5_HDR, D68): 1 SDR with no flags, or 2 HDR with flag bit 0.
    const bool hdr = argc > 3 && std::string(argv[3]) == "hdr";
    Require(output.dynamicRange == (hdr ? 2u : 1u) && output.flags == (hdr ? 1u : 0u));
    Require(output.refreshRate == 3u);

    Require(Rejects(handle, nullptr));
    Require(Rejects(0, &status));
    Require(Rejects(-1, &status));
    Require(Rejects(NEVER_OPENED_HANDLE, &status));

    Require(sceVideoOutClose(handle) == 0);
    Require(Rejects(handle, &status));
    LibcRunShutdown_nid_postfix();
}
