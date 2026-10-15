#include "SceTypes.hpp"
#include "prx/libSceSystemService/SystemService.hpp"
#include <cstdlib>
#include <string>

extern "C" int APS5_VABI sceSystemServiceGetHdrToneMapLuminance(SystemServiceHdrToneMapLuminance* luminance);

namespace {

void Require(bool value) { if (!value) std::abort(); }

}

// argv[1]: the peak luminance APS5_HDR selects for this run (100 nits for an SDR TV, the default; 1000 for HDR, D68).
int main(int argc, char** argv) {
    const float expected = argc > 1 ? std::stof(argv[1]) : 100.0f;
    SystemServiceHdrToneMapLuminance luminance{-1.0f, -1.0f, -1.0f};
    Require(sceSystemServiceGetHdrToneMapLuminance(&luminance) == SYSTEM_SERVICE_OK);
    Require(luminance.max_full_frame_tone_map_luminance == expected);
    Require(luminance.max_tone_map_luminance == expected);
    Require(luminance.min_tone_map_luminance == 0.0f);
}
