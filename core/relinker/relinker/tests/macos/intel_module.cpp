// A bundled guest module with an AMD-only instruction (SSE4a EXTRQ); --to-intel moves it into a stub
// in the module's own __AMDSTUB segment.
#include <ammintrin.h>

extern "C" unsigned long long amdExtract(unsigned long long value) {
    const __m128i extracted = _mm_extracti_si64(_mm_cvtsi64_si128(static_cast<long long>(value)), 8, 8);
    return static_cast<unsigned long long>(_mm_cvtsi128_si64(extracted));
}
