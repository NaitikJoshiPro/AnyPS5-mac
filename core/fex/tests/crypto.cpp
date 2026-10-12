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

static bool Same(const void* a, const void* b, int size) {
    for (int i = 0; i < size; ++i)
        if (static_cast<const unsigned char*>(a)[i] != static_cast<const unsigned char*>(b)[i]) return false;
    return true;
}

static void Checksums() {
    const unsigned char* text = Opaque(reinterpret_cast<const unsigned char*>("123456789"));
    unsigned crc = ~0u;
    for (int i = 0; i < 9; ++i) crc = _mm_crc32_u8(crc, text[i]);
    Check(~crc == 0xE3069283u, 0);
    unsigned long long eight = 0;
    for (int i = 0; i < 8; ++i) eight |= static_cast<unsigned long long>(text[i]) << (8 * i);
    unsigned long long wide = _mm_crc32_u64(~0ull, Opaque(eight));
    Check(~_mm_crc32_u8(static_cast<unsigned>(wide), text[8]) == 0xE3069283u, 0);
    crc = _mm_crc32_u32(~0u, Opaque(0x34333231u));
    crc = _mm_crc32_u16(crc, Opaque(static_cast<unsigned short>(0x3635)));
    crc = _mm_crc32_u16(crc, Opaque(static_cast<unsigned short>(0x3837)));
    Check(~_mm_crc32_u8(crc, '9') == 0xE3069283u, 0);
}

template <int Round>
static __m128i ExpandKey(__m128i key) {
    __m128i assist = _mm_shuffle_epi32(_mm_aeskeygenassist_si128(key, Round), 0xff);
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, assist);
}

static void Aes() {
    alignas(16) unsigned char bytes[16];
    for (int i = 0; i < 16; ++i) bytes[i] = static_cast<unsigned char>(i);
    __m128i keys[11];
    keys[0] = Opaque(_mm_load_si128(reinterpret_cast<const __m128i*>(bytes)));
    keys[1] = ExpandKey<0x01>(keys[0]);
    keys[2] = ExpandKey<0x02>(keys[1]);
    keys[3] = ExpandKey<0x04>(keys[2]);
    keys[4] = ExpandKey<0x08>(keys[3]);
    keys[5] = ExpandKey<0x10>(keys[4]);
    keys[6] = ExpandKey<0x20>(keys[5]);
    keys[7] = ExpandKey<0x40>(keys[6]);
    keys[8] = ExpandKey<0x80>(keys[7]);
    keys[9] = ExpandKey<0x1b>(keys[8]);
    keys[10] = ExpandKey<0x36>(keys[9]);
    alignas(16) const unsigned char lastKey[16] = {0x13, 0x11, 0x1d, 0x7f, 0xe3, 0x94, 0x4a, 0x17,
                                                   0xf3, 0x07, 0xa7, 0x8b, 0x4d, 0x2b, 0x30, 0xc5};
    _mm_store_si128(reinterpret_cast<__m128i*>(bytes), keys[10]);
    Check(Same(bytes, lastKey, 16), 1);

    for (int i = 0; i < 16; ++i) bytes[i] = static_cast<unsigned char>(i * 0x11);
    const __m128i plain = Opaque(_mm_load_si128(reinterpret_cast<const __m128i*>(bytes)));
    __m128i block = _mm_xor_si128(plain, keys[0]);
    for (int round = 1; round < 10; ++round) block = _mm_aesenc_si128(block, keys[round]);
    block = _mm_aesenclast_si128(block, keys[10]);
    alignas(16) const unsigned char cipher[16] = {0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
                                                  0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a};
    alignas(16) unsigned char result[16];
    _mm_store_si128(reinterpret_cast<__m128i*>(result), block);
    Check(Same(result, cipher, 16), 1);

    block = _mm_xor_si128(block, keys[10]);
    for (int round = 9; round > 0; --round) block = _mm_aesdec_si128(block, _mm_aesimc_si128(keys[round]));
    block = _mm_aesdeclast_si128(block, keys[0]);
    _mm_store_si128(reinterpret_cast<__m128i*>(result), block);
    Check(Same(result, bytes, 16), 1);
}

static void CarrylessMultiply() {
    alignas(16) unsigned long long lanes[2];
    const __m128i a = Opaque(_mm_set_epi64x(0x0123456789ABCDEFll, static_cast<long long>(0x8000000000000001ull)));
    const __m128i b = Opaque(_mm_set_epi64x(static_cast<long long>(0xFEDCBA9876543210ull), static_cast<long long>(0xC000000000000003ull)));
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), _mm_clmulepi64_si128(a, b, 0x00));
    Check(lanes[0] == 0x4000000000000003ull && lanes[1] == 0x6000000000000001ull, 2);
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), _mm_clmulepi64_si128(a, b, 0x11));
    Check(lanes[0] == 0x40a0789828c810f0ull && lanes[1] == 0x00e038d8688850b0ull, 2);
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), _mm_clmulepi64_si128(a, Opaque(_mm_set_epi64x(0, 1)), 0x01));
    Check(lanes[0] == 0x0123456789ABCDEFull && lanes[1] == 0, 2);
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), _mm_clmulepi64_si128(Opaque(_mm_set_epi64x(0, 1)), b, 0x10));
    Check(lanes[0] == 0xFEDCBA9876543210ull && lanes[1] == 0, 2);
}

static __m128i Text(const char* text) {
    return Opaque(_mm_loadu_si128(reinterpret_cast<const __m128i*>(text)));
}

static void Strings() {
    const __m128i haystack = Text("abcdefgxyzabcdef");
    Check(_mm_cmpistri(Text("zyx\0............"), haystack, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY) == 7, 3);
    Check(_mm_cmpistri(Text("xyz\0............"), haystack, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED) == 7, 3);
    Check(_mm_cmpistri(Text("AZ\0............."), Text("abc1Defghijklmno"), _SIDD_UBYTE_OPS | _SIDD_CMP_RANGES) == 4, 3);
    Check(_mm_cmpistri(Text("z\0.............."), Text("ab\0z............"), _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY) == 16, 3);
    Check(_mm_cmpistrz(Text("z\0.............."), Text("ab\0z............"), _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY) == 1, 3);
    Check(_mm_cmpistri(Text("ab\0............."), haystack, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_MOST_SIGNIFICANT) == 11, 3);
    alignas(16) unsigned short mask[8];
    _mm_store_si128(reinterpret_cast<__m128i*>(mask), _mm_cmpistrm(Text("ab\0............."), haystack, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY));
    Check(mask[0] == 0x0C03, 3);
    Check(_mm_cmpestri(Text("gxy............."), 3, haystack, 9, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED) == 6, 3);
    Check(_mm_cmpestrc(Text("q..............."), 1, haystack, 16, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY) == 0, 3);
    const __m128i words = Opaque(_mm_setr_epi16(10, 20, 30, 40, 50, 60, 70, 80));
    Check(_mm_cmpestri(Opaque(_mm_setr_epi16(25, 45, 0, 0, 0, 0, 0, 0)), 2, words, 8, _SIDD_UWORD_OPS | _SIDD_CMP_RANGES) == 2, 3);
}

static void Pixels() {
    alignas(16) unsigned long long sums[2];
    _mm_store_si128(reinterpret_cast<__m128i*>(sums), _mm_sad_epu8(Opaque(_mm_set1_epi8(10)), Opaque(_mm_setr_epi8(0, 20, 10, 10, static_cast<char>(255), 0, 5, 15, 1, 2, 3, 4, 5, 6, 7, 8))));
    Check(sums[0] == 10 + 10 + 0 + 0 + 245 + 10 + 5 + 5 && sums[1] == 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2, 4);
    alignas(16) short products[8];
    _mm_store_si128(reinterpret_cast<__m128i*>(products), _mm_maddubs_epi16(Opaque(_mm_set1_epi8(static_cast<char>(200))), Opaque(_mm_set1_epi8(100))));
    Check(products[0] == 32767 && products[7] == 32767, 4);
    alignas(16) int dots[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(dots), _mm_madd_epi16(Opaque(_mm_setr_epi16(1, 2, 3, 4, -32768, -32768, 7, 8)), Opaque(_mm_setr_epi16(5, 6, 7, 8, -32768, -32768, 9, 10))));
    Check(dots[0] == 17 && dots[1] == 53 && dots[2] == static_cast<int>(0x80000000u) && dots[3] == 143, 4);
}

extern "C" [[noreturn]] void _start(void*) {
    Checksums();
    Aes();
    CarrylessMultiply();
    Strings();
    Pixels();
    exitPointer(failures == 0 ? 43 : 100 + failures);
    __builtin_unreachable();
}
