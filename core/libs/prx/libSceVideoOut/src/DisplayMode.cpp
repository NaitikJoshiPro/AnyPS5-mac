#include "prx/libSceVideoOut/include/DisplayMode.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

VideoOutDisplayMode VideoOutDisplayModeFromText_nid_no_patch(const char* text) {
    const std::string_view value = text != nullptr ? std::string_view(text) : std::string_view();
    if (value.empty() || value == "1080p") return {1920u, 1080u, "1080p"};
    if (value == "1440p") return {2560u, 1440u, "1440p"};
    if (value == "2160p" || value == "4k" || value == "4K") return {3840u, 2160u, "2160p"};
    throw std::runtime_error("APS5_DISPLAY_MODE must be 1080p, 1440p or 2160p (4k), not '" + std::string(value) + "'");
}

const VideoOutDisplayMode& VideoOutCurrentDisplayMode_nid_no_patch() {
    static const VideoOutDisplayMode mode = [] {
        const auto parsed = VideoOutDisplayModeFromText_nid_no_patch(std::getenv("APS5_DISPLAY_MODE"));
        if (VideoOutTraceEnabled_nid_no_patch()) std::fprintf(stderr, "[videoout] display mode %s (%ux%u)\n", parsed.name, parsed.width, parsed.height);
        return parsed;
    }();
    return mode;
}

bool VideoOutTraceEnabled_nid_no_patch() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_TRACE_VIDEOOUT");
        return value != nullptr && *value != '\0' && std::string_view(value) != "0";
    }();
    return enabled;
}
