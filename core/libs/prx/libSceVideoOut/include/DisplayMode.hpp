#ifndef CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_DISPLAYMODE_HPP
#define CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_DISPLAYMODE_HPP

#include <cstdint>

// The console display mode: the output resolution of the TV the emulated PS5 is connected to, which
// sceVideoOutGetResolutionStatus and sceVideoOutGetOutputStatus report. A title that chooses its
// render and display resolution from the system renders at this size, as a real PS5 does on a TV of
// that resolution (D61). APS5_DISPLAY_MODE=1080p|1440p|2160p (also 4k), default 1080p.
struct VideoOutDisplayMode {
    std::uint32_t width;
    std::uint32_t height;
    const char* name;
};

// Parses an APS5_DISPLAY_MODE value; null or empty is the default (1080p); anything else unknown throws.
VideoOutDisplayMode VideoOutDisplayModeFromText_nid_no_patch(const char* text);
// The mode of this process (APS5_DISPLAY_MODE, read once).
const VideoOutDisplayMode& VideoOutCurrentDisplayMode_nid_no_patch();
// APS5_TRACE_VIDEOOUT: one stderr line per resolution query and buffer registration ([videoout] ...).
bool VideoOutTraceEnabled_nid_no_patch();

#endif
