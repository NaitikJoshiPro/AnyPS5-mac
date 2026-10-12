#include <immintrin.h>

extern "C" [[noreturn]] void exit(int);

void (*volatile exitPointer)(int) = exit;

static int failures = 0;

static void Check(bool right, int bit) {
    if (!right) failures |= 1 << bit;
}

template <typename T>
static T Opaque(T value) {
    asm volatile("" : "+m"(value));
    return value;
}

static unsigned Bits(__m128 value) {
    const float first = _mm_cvtss_f32(value);
    unsigned bits;
    __builtin_memcpy(&bits, &first, sizeof(bits));
    return bits;
}

static __m128 Single(unsigned bits) {
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return Opaque(_mm_set1_ps(value));
}

constexpr unsigned One = 0x3F800000, QuietNan = 0x7FC00000, PositiveZero = 0, NegativeZero = 0x80000000;

static void MinimumMaximum() {
    Check(Bits(_mm_min_ss(Single(QuietNan), Single(One))) == One && Bits(_mm_min_ss(Single(One), Single(QuietNan))) == QuietNan, 0);
    Check(Bits(_mm_max_ss(Single(QuietNan), Single(One))) == One && Bits(_mm_max_ss(Single(One), Single(QuietNan))) == QuietNan, 0);
    Check(Bits(_mm_min_ps(Single(QuietNan), Single(One))) == One && Bits(_mm_max_ps(Single(One), Single(QuietNan))) == QuietNan, 0);
    Check(Bits(_mm_min_ps(Single(NegativeZero), Single(PositiveZero))) == PositiveZero &&
              Bits(_mm_min_ps(Single(PositiveZero), Single(NegativeZero))) == NegativeZero &&
              Bits(_mm_max_ps(Single(NegativeZero), Single(PositiveZero))) == PositiveZero,
          0);
}

static void Conversions() {
    Check(_mm_cvttss_si32(Single(QuietNan)) == static_cast<int>(0x80000000u) && _mm_cvttss_si32(Opaque(_mm_set_ss(3e9f))) == static_cast<int>(0x80000000u) &&
              _mm_cvttss_si32(Opaque(_mm_set_ss(-3e9f))) == static_cast<int>(0x80000000u),
          1);
    Check(_mm_cvttsd_si64(Opaque(_mm_set_sd(1e20))) == static_cast<long long>(0x8000000000000000ull), 1);
    alignas(16) int lanes[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), _mm_cvttps_epi32(Opaque(_mm_setr_ps(__builtin_nanf(""), 3e9f, -3e9f, 7.9f))));
    Check(lanes[0] == static_cast<int>(0x80000000u) && lanes[1] == static_cast<int>(0x80000000u) && lanes[2] == static_cast<int>(0x80000000u) && lanes[3] == 7, 1);
    Check(_mm_cvtss_si32(Opaque(_mm_set_ss(2.5f))) == 2 && _mm_cvtss_si32(Opaque(_mm_set_ss(3.5f))) == 4, 1);
}

static void Nans() {
    Check(Bits(_mm_mul_ss(Single(PositiveZero), Single(0x7F800000))) == 0xFFC00000 && Bits(_mm_sqrt_ss(Single(0xBF800000))) == 0xFFC00000, 2);
    Check(Bits(_mm_add_ss(Single(0x7FC12345), Single(One))) == 0x7FC12345 && Bits(_mm_add_ss(Single(One), Single(0x7F800001))) == 0x7FC00001, 2);
}

static void Rounding(unsigned control) {
    const float a = 2.7f, b = -2.5f, c = 2.2f, d = -2.7f;
    _mm_setcsr((control & ~0x6000u) | 0x2000u);
    Check(_mm_cvtss_si32(Opaque(_mm_set_ss(a))) == 2 && _mm_cvtss_si32(Opaque(_mm_set_ss(b))) == -3 &&
              Bits(_mm_add_ss(Single(One), Single(0x30800000))) == One &&
              _mm_cvtss_f32(_mm_round_ss(_mm_setzero_ps(), Opaque(_mm_set_ss(a)), _MM_FROUND_CUR_DIRECTION)) == 2.0f,
          3);
    _mm_setcsr((control & ~0x6000u) | 0x4000u);
    Check(_mm_cvtss_si32(Opaque(_mm_set_ss(c))) == 3 && Bits(_mm_add_ss(Single(One), Single(0x30800000))) == One + 1, 3);
    _mm_setcsr((control & ~0x6000u) | 0x6000u);
    Check(_mm_cvtss_si32(Opaque(_mm_set_ss(d))) == -2, 3);
    _mm_setcsr(control);
    Check(_mm_cvtss_si32(Opaque(_mm_set_ss(a))) == 3, 3);
}

static void Denormals(unsigned control) {
    const auto scaled = [] { return Bits(_mm_mul_ss(Single(0x00400000), Single(0x71800000))); };
    const auto halved = [] { return Bits(_mm_mul_ss(Single(0x00800000), Single(0x3F000000))); };
    Check(scaled() == 0x32000000 && halved() == 0x00400000, 4);
    _mm_setcsr(control | 0x8000u);
    Check(scaled() == 0x32000000 && halved() == 0, 4);
    _mm_setcsr(control | 0x0040u);
    Check(scaled() == 0 && halved() == 0x00400000, 4);
    _mm_setcsr(control | 0x8040u);
    Check(scaled() == 0 && halved() == 0 && _mm_getcsr() == (control | 0x8040u), 4);
    _mm_setcsr(control);
}

extern "C" [[noreturn]] void _start(void*) {
    const unsigned control = _mm_getcsr();
    MinimumMaximum();
    Conversions();
    Nans();
    Rounding(control);
    Denormals(control);
    exitPointer(failures == 0 ? 43 : 100 + failures);
    __builtin_unreachable();
}
