#include <cpuid.h>
#include <immintrin.h>

extern "C" void* mmap(void* address, unsigned long length, int protection, int flags, int descriptor, long offset);
extern "C" int munmap(void* address, unsigned long length);
extern "C" [[noreturn]] void exit(int);

static int failures = 0;

static void Check(bool right, int bit) {
    if (!right) failures |= 1 << bit;
}

template <typename T>
static T Opaque(T value) {
    asm volatile("" : "+m"(value));
    return value;
}

template <typename T, int N>
static bool Lanes(const T (&actual)[N], const T (&expected)[N]) {
    for (int i = 0; i < N; ++i)
        if (actual[i] != expected[i]) return false;
    return true;
}

static void Features() {
    unsigned a, b, c, d;
    __cpuid(1, a, b, c, d);
    const unsigned leaf1 = bit_SSE3 | bit_SSSE3 | bit_FMA | bit_SSE41 | bit_SSE42 | bit_POPCNT | bit_XSAVE | bit_OSXSAVE | bit_AVX | bit_F16C;
    Check((c & leaf1) == leaf1, 0);
    __cpuid_count(7, 0, a, b, c, d);
    const unsigned leaf7 = bit_BMI | bit_AVX2 | bit_BMI2;
    Check((b & leaf7) == leaf7, 0);
    __cpuid(0x80000001, a, b, c, d);
    constexpr unsigned Rdtscp = 1u << 27;
    Check((c & bit_LZCNT) != 0 && (d & Rdtscp) != 0, 0);
    Check((_xgetbv(0) & 6) == 6, 1);
}

static void Integers() {
    alignas(32) int lanes[8];
    const __m256i counting = Opaque(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),
                       _mm256_add_epi32(_mm256_mullo_epi32(counting, counting), Opaque(_mm256_set1_epi32(100000))));
    Check(Lanes(lanes, {100000, 100001, 100004, 100009, 100016, 100025, 100036, 100049}), 2);

    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),
                       _mm256_permutevar8x32_epi32(counting, Opaque(_mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0))));
    Check(Lanes(lanes, {7, 6, 5, 4, 3, 2, 1, 0}), 3);

    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),
                       _mm256_sllv_epi32(Opaque(_mm256_set1_epi32(1)), Opaque(_mm256_setr_epi32(0, 1, 2, 3, 31, 32, 33, 4))));
    Check(Lanes(lanes, {1, 2, 4, 8, static_cast<int>(0x80000000u), 0, 0, 16}), 3);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),
                       _mm256_srav_epi32(Opaque(_mm256_set1_epi32(-64)), Opaque(_mm256_setr_epi32(0, 1, 2, 3, 6, 7, 40, 31))));
    Check(Lanes(lanes, {-64, -32, -16, -8, -1, -1, -1, -1}), 3);

    alignas(32) unsigned char bytes[32];
    for (int i = 0; i < 32; ++i) bytes[i] = static_cast<unsigned char>(i);
    const __m256i shuffled = _mm256_shuffle_epi8(_mm256_load_si256(reinterpret_cast<const __m256i*>(bytes)), Opaque(_mm256_set1_epi8(15)));
    _mm256_store_si256(reinterpret_cast<__m256i*>(bytes), shuffled);
    Check(bytes[0] == 15 && bytes[15] == 15 && bytes[16] == 31 && bytes[31] == 31, 4);

    Check(_mm256_movemask_epi8(_mm256_cmpgt_epi8(Opaque(_mm256_set1_epi8(1)), _mm256_setzero_si256())) == -1, 4);
    Check(_mm256_testz_si256(counting, Opaque(_mm256_set1_epi32(8))) == 1, 4);
}

static void Gathers() {
    int table[16];
    for (int i = 0; i < 16; ++i) table[i] = i * i;
    alignas(32) int lanes[8];
    const __m256i index = Opaque(_mm256_setr_epi32(15, 0, 3, 7, 1, 1, 8, 12));
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), _mm256_i32gather_epi32(table, index, 4));
    Check(Lanes(lanes, {225, 0, 9, 49, 1, 1, 64, 144}), 5);
    const __m256i mask = Opaque(_mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0));
    const __m256i wild = Opaque(_mm256_setr_epi32(15, 1 << 28, 3, 1 << 28, 1, 1 << 28, 8, 1 << 28));
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), _mm256_mask_i32gather_epi32(_mm256_set1_epi32(-5), table, wild, mask, 4));
    Check(Lanes(lanes, {225, -5, 9, -5, 1, -5, 64, -5}), 5);
}

static void FusedMultiplyAdd() {
    alignas(32) float singles[8];
    const __m256 a = Opaque(_mm256_set1_ps(1.0f + 0x1p-12f));
    _mm256_store_ps(singles, _mm256_fmadd_ps(a, a, Opaque(_mm256_set1_ps(-(1.0f + 0x1p-11f)))));
    Check(singles[0] == 0x1p-24f && singles[7] == 0x1p-24f, 6);
    _mm256_store_ps(singles, _mm256_fnmadd_ps(a, a, Opaque(_mm256_set1_ps(1.0f + 0x1p-11f))));
    Check(singles[0] == -0x1p-24f && singles[7] == -0x1p-24f, 6);

    alignas(32) double doubles[4];
    const __m256d b = Opaque(_mm256_set1_pd(1.0 + 0x1p-27));
    _mm256_store_pd(doubles, _mm256_fmsub_pd(b, b, Opaque(_mm256_set1_pd(1.0 + 0x1p-26))));
    Check(doubles[0] == 0x1p-54 && doubles[3] == 0x1p-54, 6);
    Check(_mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(Opaque(1.0 + 0x1p-27)), _mm_set_sd(1.0 + 0x1p-27), _mm_set_sd(-(1.0 + 0x1p-26)))) == 0x1p-54, 6);
}

static void Bits() {
    Check(_pext_u64(Opaque(0x12345678ull), 0xFF00FF00ull) == 0x1256, 7);
    Check(_pdep_u64(Opaque(0x1256ull), 0xFF00FF00ull) == 0x12005600, 7);
    Check(_bzhi_u64(Opaque(~0ull), 12) == 0xFFF, 7);
    unsigned long long high = 0;
    Check(_mulx_u64(Opaque(1ull << 63), 6, &high) == 0 && high == 3, 7);
    Check(_andn_u64(Opaque(0xF0ull), 0xFFull) == 0x0F, 8);
    Check(_blsr_u64(Opaque(0b1100ull)) == 0b1000 && _blsi_u64(Opaque(0b1100ull)) == 0b0100, 8);
    Check(_tzcnt_u64(Opaque(0x80ull)) == 7 && _tzcnt_u64(Opaque(0ull)) == 64, 8);
    Check(_lzcnt_u64(Opaque(1ull)) == 63 && _lzcnt_u64(Opaque(0ull)) == 64, 8);
    Check(_mm_popcnt_u64(Opaque(0xF0F0F0F0F0F0F0F0ull)) == 32, 8);
    long long value = Opaque(-1024ll);
    unsigned count = Opaque(3u);
    Check((value >> count) == -128 && (static_cast<unsigned long long>(value) << count) == 0xFFFFFFFFFFFFE000ull, 8);
}

static void Conversions() {
    alignas(16) float singles[4];
    _mm_store_ps(singles, _mm_cvtph_ps(Opaque(_mm_setr_epi16(0x3C00, static_cast<short>(0xC000), 0x7BFF, 0x0001, 0, 0, 0, 0))));
    Check(Lanes(singles, {1.0f, -2.0f, 65504.0f, 0x1p-24f}), 9);
    Check(_mm_extract_epi16(_mm_cvtps_ph(Opaque(_mm_set1_ps(1.5f)), _MM_FROUND_TO_NEAREST_INT), 0) == 0x3E00, 9);

    alignas(32) float rounded[8];
    const __m256 halves = Opaque(_mm256_setr_ps(2.5f, 3.5f, -2.5f, -2.7f, 0.5f, 1.5f, -0.5f, 2.7f));
    _mm256_store_ps(rounded, _mm256_round_ps(halves, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    Check(Lanes(rounded, {2.0f, 4.0f, -2.0f, -3.0f, 0.0f, 2.0f, -0.0f, 3.0f}), 10);
    _mm256_store_ps(rounded, _mm256_round_ps(halves, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
    Check(Lanes(rounded, {2.0f, 3.0f, -2.0f, -2.0f, 0.0f, 1.0f, -0.0f, 2.0f}), 10);

    alignas(32) double roots[4];
    _mm256_store_pd(roots, _mm256_sqrt_pd(Opaque(_mm256_setr_pd(4.0, 9.0, 16.0, 2.0))));
    Check(roots[0] == 2.0 && roots[1] == 3.0 && roots[2] == 4.0 && roots[3] == 0x1.6a09e667f3bcdp+0, 10);

    alignas(32) float sums[8];
    const __m256 counting = Opaque(_mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8));
    _mm256_store_ps(sums, _mm256_hadd_ps(counting, counting));
    Check(Lanes(sums, {3.0f, 7.0f, 3.0f, 7.0f, 11.0f, 15.0f, 11.0f, 15.0f}), 10);
    _mm256_store_ps(sums, _mm256_blendv_ps(counting, _mm256_setzero_ps(), Opaque(_mm256_setr_ps(-1, 1, -1, 1, -1, 1, -1, 1))));
    Check(Lanes(sums, {0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f, 0.0f, 8.0f}), 10);
}

void* (*volatile mapPointer)(void*, unsigned long, int, int, int, long) = mmap;

static void DoublesToIntegers() {
    alignas(16) int lanes[4];
    __m128i converted;
    asm("cvtpd2dq %1, %0" : "=x"(converted) : "x"(Opaque(_mm_setr_pd(16777217.0, -2147483000.0))));
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), converted);
    Check(lanes[0] == 16777217 && lanes[1] == -2147483000, 13);
    asm("cvttpd2dq %1, %0" : "=x"(converted) : "x"(Opaque(_mm_setr_pd(33554435.75, -16777219.5))));
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), converted);
    Check(lanes[0] == 33554435 && lanes[1] == -16777219, 13);
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), _mm_cvttpd_epi32(Opaque(_mm_setr_pd(33554435.75, -16777219.5))));
    Check(lanes[0] == 33554435 && lanes[1] == -16777219, 13);
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), _mm256_cvtpd_epi32(Opaque(_mm256_setr_pd(16777217.0, 16777219.0, -16777221.0, 1e10))));
    Check(lanes[0] == 16777217 && lanes[1] == 16777219 && lanes[2] == -16777221 && lanes[3] == static_cast<int>(0x80000000u), 13);
}

static void MaskedLoads() {
    constexpr unsigned long page = 16384;
    auto* pages = static_cast<unsigned char*>(mapPointer(nullptr, 2 * page, 3, 0x1002, -1, 0));
    if (pages == reinterpret_cast<unsigned char*>(-1l) || munmap(pages + page, page) != 0) {
        failures |= 1 << 11;
        return;
    }
    auto* edge = reinterpret_cast<float*>(pages + page) - 4;
    for (int i = 0; i < 4; ++i) edge[i] = static_cast<float>(i + 1);
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, _mm256_maskload_ps(edge, Opaque(_mm256_setr_epi32(-1, -1, 0, -1, 0, 0, 0, 0))));
    Check(Lanes(lanes, {1.0f, 2.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f}), 11);
    _mm256_maskstore_ps(edge, Opaque(_mm256_setr_epi32(0, -1, 0, 0, 0, 0, 0, 0)), _mm256_set1_ps(9.0f));
    Check(edge[0] == 1.0f && edge[1] == 9.0f && edge[2] == 3.0f && edge[3] == 4.0f, 11);
    munmap(pages, page);
}

static void TimeStamps() {
    const unsigned long long first = __rdtsc();
    unsigned long long later = first;
    for (int i = 0; i < 100000000 && later == first; ++i) later = __rdtsc();
    Check(later > first, 14);
    unsigned low, high, processor;
    asm volatile("rdtscp" : "=a"(low), "=d"(high), "=c"(processor));
    Check((static_cast<unsigned long long>(high) << 32 | low) >= later && processor < 4096, 14);
    unsigned long long identifier = ~0ull;
    asm volatile("rdpid %0" : "=r"(identifier));
    Check(identifier < 4096, 14);
}

static void Vectorized() {
    static int values[1000];
    for (int i = 0; i < 1000; ++i) values[i] = Opaque(i) * 3;
    long long sum = 0;
    for (int i = 0; i < 1000; ++i) sum += values[i] ^ (i & 7);
    long long expected = 0;
    for (int i = 0; i < 1000; ++i) expected += Opaque((i * 3) ^ (i & 7));
    Check(sum == expected, 12);
}

extern "C" [[noreturn]] void _start(void*) {
    Features();
    Integers();
    Gathers();
    FusedMultiplyAdd();
    Bits();
    Conversions();
    DoublesToIntegers();
    MaskedLoads();
    TimeStamps();
    Vectorized();
    exit(failures == 0 ? 43 : 100 + __builtin_ctz(failures));
}
