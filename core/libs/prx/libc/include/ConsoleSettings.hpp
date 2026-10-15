#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_CONSOLESETTINGS_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_CONSOLESETTINGS_HPP

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

// The emulated console's user-level settings that the system libraries report to titles: the TV the console is
// connected to and the user's game presets. They are the same for every title (no per-title answers, D61, D66).
//   APS5_DISPLAY_MODE  1080p (default) | 1440p | 2160p: the TV's output resolution (libSceVideoOut DisplayMode.cpp)
//   APS5_HDR           0 (default, an SDR TV) | 1 (an HDR TV)
//   APS5_GAME_PRESET   performance (default) | default: Settings > Game Presets > "Performance Mode or Resolution
//                      Mode". performance reports priority 1, the value a title reads as "favour frame rate";
//                      default reports 0, "Game Default" (the title's own choice).

// HDR output: GetOutputStatus reports dynamic range 2 (HDR) instead of 1 (SDR) and flag bit 0, and
// GetHdrToneMapLuminance reports an HDR panel instead of SDR reference white.
inline bool ConsoleHdrOutputFromText(const char* text) {
    const std::string_view value = text != nullptr ? std::string_view(text) : std::string_view();
    if (value.empty() || value == "0") return false;
    if (value == "1") return true;
    throw std::runtime_error("APS5_HDR must be 0 or 1, not '" + std::string(value) + "'");
}

inline bool ConsoleHdrOutput() {
    static const bool hdr = ConsoleHdrOutputFromText(std::getenv("APS5_HDR"));
    return hdr;
}

// sceUserServiceGetGamePresets priority: 0 is "Game Default", 1 "Performance Mode" (Ghost of Yotei compares the
// field with 1 and then selects its performance mode: PS5PCEM docs/development/game-presets-2026-09-26.md). The
// value of "Resolution Mode" is not known, so it is not offered.
inline constexpr std::uint32_t ConsoleGamePresetPriorityGameDefault = 0;
inline constexpr std::uint32_t ConsoleGamePresetPriorityPerformance = 1;

inline std::uint32_t ConsoleGamePresetPriorityFromText(const char* text) {
    const std::string_view value = text != nullptr ? std::string_view(text) : std::string_view();
    if (value.empty() || value == "performance") return ConsoleGamePresetPriorityPerformance;
    if (value == "default") return ConsoleGamePresetPriorityGameDefault;
    throw std::runtime_error("APS5_GAME_PRESET must be performance or default, not '" + std::string(value) + "'");
}

inline std::uint32_t ConsoleGamePresetPriority() {
    static const std::uint32_t priority = ConsoleGamePresetPriorityFromText(std::getenv("APS5_GAME_PRESET"));
    return priority;
}

// APS5_TRACE_VIDEOOUT=1: one stderr line per answer the system libraries give a title about the console's
// display and user settings: VideoOut resolution and output queries, buffer registrations, SystemService
// parameters, HDR tone map and safe area, UserService game presets. Titles choose their output size from
// these, so the trace shows which answers a title read before it registered its buffers.
inline bool SystemQueryTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_TRACE_VIDEOOUT");
        return value != nullptr && *value != '\0' && std::string_view(value) != "0";
    }();
    return enabled;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
inline void SystemQueryTrace(const char* library, const char* format, ...) {
    if (!SystemQueryTraceEnabled()) return;
    char line[512];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    std::fprintf(stderr, "[%s] %s\n", library, line);
}

#endif
