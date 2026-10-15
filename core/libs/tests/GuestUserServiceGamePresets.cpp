#include "SceTypes.hpp"
#include "prx/libSceUserService/UserService.hpp"
#include "prx/libc/include/ConsoleSettings.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" {
int APS5_VABI sceUserServiceGetGamePresets(int user_id, UserServiceGamePresets* presets);
}

namespace {

void Require(bool value) { if (!value) std::abort(); }

bool PresetRejected(const char* text) {
    try {
        ConsoleGamePresetPriorityFromText(text);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

bool HdrRejected(const char* text) {
    try {
        ConsoleHdrOutputFromText(text);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

// The console settings a title reads (D66): APS5_GAME_PRESET and APS5_HDR parsing.
void CheckParsing() {
    Require(ConsoleGamePresetPriorityFromText(nullptr) == ConsoleGamePresetPriorityPerformance);
    Require(ConsoleGamePresetPriorityFromText("") == ConsoleGamePresetPriorityPerformance);
    Require(ConsoleGamePresetPriorityFromText("performance") == 1);
    Require(ConsoleGamePresetPriorityFromText("default") == 0);
    Require(PresetRejected("resolution") && PresetRejected("1") && PresetRejected("Performance") && PresetRejected(" default"));
    Require(!ConsoleHdrOutputFromText(nullptr) && !ConsoleHdrOutputFromText("") && !ConsoleHdrOutputFromText("0"));
    Require(ConsoleHdrOutputFromText("1"));
    Require(HdrRejected("on") && HdrRejected("2") && HdrRejected("true"));
}

}

// argv[1]: the priority APS5_GAME_PRESET selects for this run (default 1, performance).
int main(int argc, char** argv) {
    CheckParsing();
    const std::uint32_t expected = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 1u;
    Require(ConsoleGamePresetPriority() == expected);

    UserServiceGamePresets presets;
    std::memset(&presets, 0x5a, sizeof(presets));
    presets.this_size = sizeof(presets);
    Require(sceUserServiceGetGamePresets(USER_SERVICE_INITIAL_USER_ID, nullptr) == USER_SERVICE_ERROR_INVALID_ARGUMENT);
    Require(sceUserServiceGetGamePresets(0x12345678, &presets) == USER_SERVICE_ERROR_INVALID_ARGUMENT);
    Require(presets.priority == 0x5a5a5a5au);

    Require(sceUserServiceGetGamePresets(USER_SERVICE_INITIAL_USER_ID, &presets) == USER_SERVICE_OK);
    Require(presets.this_size == sizeof(presets));
    Require(presets.priority == expected);
    Require(presets.difficulty == 0 && presets.display_sub_titles == 0 && presets.audio_language == 0);
    Require(presets.invert_vertical_view_for_1st_person_view == 0 && presets.invert_horizontal_view_for_1st_person_view == 0);
    Require(presets.invert_vertical_view_for_3rd_person_view == 0 && presets.invert_horizontal_view_for_3rd_person_view == 0);

    // A caller block larger than the known layout (Ghost of Yotei passes 48): the bytes past it stay the caller's.
    unsigned char extended[48];
    std::memset(extended, 0x5a, sizeof(extended));
    const std::size_t extendedSize = sizeof(extended);
    std::memcpy(extended, &extendedSize, sizeof(extendedSize));
    Require(sceUserServiceGetGamePresets(USER_SERVICE_INITIAL_USER_ID, reinterpret_cast<UserServiceGamePresets*>(extended)) == USER_SERVICE_OK);
    std::uint32_t priority = 0;
    std::memcpy(&priority, extended + offsetof(UserServiceGamePresets, priority), sizeof(priority));
    Require(priority == expected);
    for (std::size_t index = sizeof(UserServiceGamePresets); index < sizeof(extended); ++index) Require(extended[index] == 0x5a);

    // A shorter block gets only the fields inside it.
    std::memset(&presets, 0x5a, sizeof(presets));
    presets.this_size = offsetof(UserServiceGamePresets, invert_vertical_view_for_1st_person_view);
    Require(sceUserServiceGetGamePresets(USER_SERVICE_INITIAL_USER_ID, &presets) == USER_SERVICE_OK);
    Require(presets.difficulty == 0 && presets.priority == expected);
    Require(presets.invert_vertical_view_for_1st_person_view == 0x5a5a5a5au && presets.audio_language == 0x5a5a5a5au);

    // A block too short to hold any field is refused and left alone.
    std::memset(&presets, 0x5a, sizeof(presets));
    presets.this_size = 4;
    Require(sceUserServiceGetGamePresets(USER_SERVICE_INITIAL_USER_ID, &presets) == USER_SERVICE_ERROR_INVALID_ARGUMENT);
    Require(presets.difficulty == 0x5a5a5a5au && presets.priority == 0x5a5a5a5au);
}
