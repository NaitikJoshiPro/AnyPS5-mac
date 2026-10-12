#include "bench.h"

#if defined(__AVX2__)
#define BENCH_VECTOR_BYTES 32
#else
#define BENCH_VECTOR_BYTES 16
#endif

u64 benchExtra = 0;
u64 benchElapsed = 0;

namespace {

constexpr u64 WordCount = 32768;
constexpr u64 ByteCount = WordCount * 8;
constexpr u64 NumberCount = 65536;
constexpr u64 HashBytes = 65536;
constexpr u64 FloatCount = 16384;
constexpr u64 SortCount = 16384;
constexpr u64 BridgeSortCount = 1024;
constexpr u64 TextBytes = 65536;
constexpr u64 SmallCopy = 64;
constexpr u32 MaxThreads = 4;

typedef float VFloat __attribute__((vector_size(BENCH_VECTOR_BYTES)));
typedef u8 VByte __attribute__((vector_size(BENCH_VECTOR_BYTES)));
typedef float F4 __attribute__((vector_size(16)));
constexpr u64 FloatLanes = BENCH_VECTOR_BYTES / sizeof(float);

alignas(64) u64 source[WordCount];
alignas(64) u64 target[WordCount];
alignas(64) u32 numbers[NumberCount];
alignas(64) u32 sorted[SortCount];
alignas(64) float floatX[FloatCount];
alignas(64) float floatY[FloatCount];
alignas(64) float vectors[FloatCount];
alignas(64) float transformed[FloatCount];
alignas(64) i16 samples[FloatCount];
alignas(64) u8 text[TextBytes];
u32 crcTable[256];
u32 histogram[256];

inline void Clobber() {
    asm volatile("" ::: "memory");
}

template <typename T>
inline T Opaque(T value) {
    asm volatile("" : "+r"(value));
    return value;
}

inline u64 InRegister(u64 value) {
    asm("" : "+r"(value));
    return value;
}

inline void Pause() {
#if defined(__x86_64__)
    asm volatile("pause");
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

constexpr u64 Mix(u64 value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

u8* Bytes(u64* words) {
    return reinterpret_cast<u8*>(words);
}

void FillWords(u64* words, u64 count, u64 seed) {
    for (u64 i = 0; i < count; ++i) words[i] = Mix(seed + i);
}

u64 Sample(const u64* words, u64 count) {
    u64 hash = count;
    for (u64 i = 0; i < count; i += 61) hash = hash * 0x9e3779b97f4a7c15ULL + words[i];
    return hash;
}

void PrepareWords() {
    FillWords(source, WordCount, 1);
}

void PrepareNumbers() {
    for (u64 i = 0; i < NumberCount; ++i) numbers[i] = static_cast<u32>(Mix(i + 7));
}

u64 IntSum(u64 work) {
    u64 total = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
#pragma clang loop vectorize(disable)
        for (u64 i = 0; i < NumberCount; ++i) total += numbers[i];
    }
    return total;
}

u64 IntHash(u64 work) {
    const u8* bytes = Bytes(source);
    u64 hash = 0xcbf29ce484222325ULL;
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        for (u64 i = 0; i < HashBytes; ++i) {
            hash ^= bytes[i];
            hash *= 0x100000001b3ULL;
        }
    }
    return hash;
}

constexpr u64 Prime1 = 0x9E3779B185EBCA87ULL;
constexpr u64 Prime2 = 0xC2B2AE3D27D4EB4FULL;

inline u64 Rotate(u64 value, int count) {
    return (value << count) | (value >> (64 - count));
}

inline u64 Round(u64 accumulator, u64 input) {
    return Rotate(accumulator + input * Prime2, 31) * Prime1;
}

u64 IntMix(u64 work) {
    u64 a = Prime1 + Prime2;
    u64 b = Prime2;
    u64 c = 0;
    u64 d = 0 - Prime1;
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        for (u64 i = 0; i < WordCount; i += 4) {
            a = Round(a, source[i]);
            b = Round(b, source[i + 1]);
            c = Round(c, source[i + 2]);
            d = Round(d, source[i + 3]);
        }
    }
    return Rotate(a, 1) + Rotate(b, 7) + Rotate(c, 12) + Rotate(d, 18);
}

void PrepareCrc() {
    PrepareWords();
    for (u32 n = 0; n < 256; ++n) {
        u32 value = n;
        for (int bit = 0; bit < 8; ++bit) value = (value & 1) != 0 ? 0xEDB88320u ^ (value >> 1) : value >> 1;
        crcTable[n] = value;
    }
}

u64 IntCrc32(u64 work) {
    const u8* bytes = Bytes(source);
    u64 sum = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        u32 crc = 0xffffffffu ^ static_cast<u32>(pass);
        for (u64 i = 0; i < HashBytes; ++i) crc = crcTable[(crc ^ bytes[i]) & 0xff] ^ (crc >> 8);
        sum = sum * 31 + (crc ^ 0xffffffffu);
    }
    return sum;
}

template <typename Less>
void InsertionSort(u32* values, i64 count, Less less) {
    for (i64 i = 1; i < count; ++i) {
        const u32 value = values[i];
        i64 j = i - 1;
        while (j >= 0 && less(value, values[j])) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = value;
    }
}

template <typename Less>
void Order(u32& left, u32& right, Less less) {
    if (less(right, left)) {
        const u32 swap = left;
        left = right;
        right = swap;
    }
}

template <typename Less>
void QuickSort(u32* values, i64 count, Less less) {
    while (count > 16) {
        const i64 middle = count / 2;
        Order(values[0], values[middle], less);
        Order(values[middle], values[count - 1], less);
        Order(values[0], values[middle], less);
        const u32 pivot = values[middle];
        i64 i = -1;
        i64 j = count;
        for (;;) {
            do ++i;
            while (less(values[i], pivot));
            do --j;
            while (less(pivot, values[j]));
            if (i >= j) break;
            const u32 swap = values[i];
            values[i] = values[j];
            values[j] = swap;
        }
        const i64 left = j + 1;
        if (left < count - left) {
            QuickSort(values, left, less);
            values += left;
            count -= left;
        } else {
            QuickSort(values + left, count - left, less);
            count = left;
        }
    }
    InsertionSort(values, count, less);
}

void PrepareSort() {
    for (u64 i = 0; i < SortCount; ++i) numbers[i] = static_cast<u32>(Mix(i * 3 + 11));
}

u64 SortedDigest(const u32* values, u64 count) {
    u64 hash = 0;
    bool ordered = true;
    for (u64 i = 0; i < count; ++i) {
        if (i != 0 && values[i - 1] > values[i]) ordered = false;
        hash = hash * 31 + values[i];
    }
    return ordered ? hash : 0;
}

void Scramble(u64 count, u64 pass) {
    for (u64 i = 0; i < count; ++i) sorted[i] = numbers[i] ^ static_cast<u32>(pass * 0x9e3779b9u);
}

u64 IntSort(u64 work) {
    u64 sum = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Scramble(SortCount, pass);
        QuickSort(sorted, SortCount, [](u32 left, u32 right) { return left < right; });
        sum = sum * 31 + SortedDigest(sorted, SortCount);
    }
    return sum;
}

int CompareNumbers(const void* left, const void* right) {
    const u32 a = *static_cast<const u32*>(left);
    const u32 b = *static_cast<const u32*>(right);
    return a < b ? -1 : a > b;
}

int (*volatile comparePointer)(const void*, const void*) = CompareNumbers;

struct ByPointer {
    int (*compare)(const void*, const void*);

    bool operator()(u32 left, u32 right) const {
        return compare(&left, &right) < 0;
    }
};

u64 IntSortPointer(u64 work) {
    u64 sum = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Scramble(SortCount, pass);
        QuickSort(sorted, SortCount, ByPointer {comparePointer});
        sum = sum * 31 + SortedDigest(sorted, SortCount);
    }
    return sum;
}

enum Opcode : u8 {
    OpMultiply,
    OpAddImmediate,
    OpShiftRight,
    OpAndImmediate,
    OpAdd,
    OpXor,
    OpSubtractImmediate,
    OpJumpNotZero,
    OpHalt,
};

struct Instruction {
    u8 op;
    u8 target;
    u8 left;
    u8 right;
    i32 immediate;
};

Instruction program[] = {
    {OpMultiply, 2, 2, 3, 0},
    {OpAddImmediate, 2, 2, 0, 12345},
    {OpShiftRight, 4, 2, 0, 16},
    {OpAndImmediate, 4, 4, 0, 0x7fff},
    {OpAdd, 1, 1, 4, 0},
    {OpXor, 1, 1, 0, 0},
    {OpSubtractImmediate, 0, 0, 0, 1},
    {OpJumpNotZero, 0, 0, 0, 0},
    {OpHalt, 0, 0, 0, 0},
};

u64 Interpret(const Instruction* code, u64 iterations) {
    u64 registers[8] = {iterations, 0, 12345, 1103515245, 0, 0, 0, 0};
    u32 pc = 0;
    for (;;) {
        const Instruction instruction = code[pc++];
        switch (instruction.op) {
        case OpMultiply: registers[instruction.target] = registers[instruction.left] * registers[instruction.right]; break;
        case OpAddImmediate: registers[instruction.target] = registers[instruction.left] + instruction.immediate; break;
        case OpShiftRight: registers[instruction.target] = registers[instruction.left] >> instruction.immediate; break;
        case OpAndImmediate: registers[instruction.target] = registers[instruction.left] & instruction.immediate; break;
        case OpAdd: registers[instruction.target] = registers[instruction.left] + registers[instruction.right]; break;
        case OpXor: registers[instruction.target] = registers[instruction.left] ^ registers[instruction.right]; break;
        case OpSubtractImmediate: registers[instruction.target] = registers[instruction.left] - instruction.immediate; break;
        case OpJumpNotZero:
            if (registers[instruction.left] != 0) pc = static_cast<u32>(instruction.immediate);
            break;
        default: return registers[1];
        }
    }
}

u64 IntInterpreter(u64 work) {
    return Interpret(Opaque(&program[0]), work);
}

struct TextWriter {
    u64 at = 0;

    void Put(u64 c) {
        text[at++] = static_cast<u8>(InRegister(c));
    }
};

void PrepareText() {
    const char operators[] = "+-*=<>(){};,.&|!";
    for (u64 i = 0; i < TextBytes; ++i) text[i] = ' ';
    TextWriter out;
    u64 state = 99;
    while (out.at + 64 < TextBytes) {
        state = Mix(state + out.at);
        const u64 length = 1 + (state >> 8) % 10;
        switch (state % 8) {
        case 0:
        case 1:
            for (u64 i = 0; i < length; ++i) out.Put(i == 0 ? 'a' + (state >> (16 + i)) % 26 : static_cast<u8>("abcxyz_0123"[(state >> (12 + i)) % 11]));
            break;
        case 2:
            for (u64 i = 0; i < length % 6 + 1; ++i) out.Put('0' + (state >> (20 + i)) % 10);
            break;
        case 3: out.Put((state & 0x100) != 0 ? '\n' : ' '); break;
        case 4: out.Put(static_cast<u8>(operators[(state >> 24) % 16])); break;
        case 5:
            out.Put('"');
            for (u64 i = 0; i < length; ++i) out.Put('a' + (state >> (30 + i)) % 26);
            out.Put('"');
            break;
        case 6:
            out.Put('/');
            out.Put('/');
            for (u64 i = 0; i < length * 2; ++i) out.Put(i % 5 == 4 ? ' ' : 'k' + (state >> (8 + i)) % 8);
            out.Put('\n');
            break;
        default: out.Put(' '); break;
        }
    }
    text[TextBytes - 1] = '\n';
}

inline bool IsLetter(u8 c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

inline bool IsDigit(u8 c) {
    return c >= '0' && c <= '9';
}

enum class LexState : u8 { Start, Identifier, Number, String, Slash, Comment };

u64 Lex(const u8* input, u64 size) {
    u64 identifiers = 0, literals = 0, strings = 0, comments = 0, operators = 0, lines = 0;
    LexState state = LexState::Start;
    for (u64 i = 0; i < size; ++i) {
        const u8 c = input[i];
        switch (state) {
        case LexState::Identifier:
            if (IsLetter(c) || IsDigit(c)) continue;
            break;
        case LexState::Number:
            if (IsDigit(c)) continue;
            break;
        case LexState::String:
            if (c == '"') state = LexState::Start;
            continue;
        case LexState::Slash:
            if (c == '/') {
                state = LexState::Comment;
                ++comments;
                continue;
            }
            ++operators;
            break;
        case LexState::Comment:
            if (c == '\n') {
                state = LexState::Start;
                ++lines;
            }
            continue;
        case LexState::Start: break;
        }
        state = LexState::Start;
        if (IsLetter(c)) {
            state = LexState::Identifier;
            ++identifiers;
        } else if (IsDigit(c)) {
            state = LexState::Number;
            ++literals;
        } else if (c == '"') {
            state = LexState::String;
            ++strings;
        } else if (c == '/') {
            state = LexState::Slash;
        } else if (c == '\n') {
            ++lines;
        } else if (c != ' ') {
            ++operators;
        }
    }
    return Mix(identifiers + (literals << 12) + (strings << 24) + (comments << 36) + (operators << 48) + (lines << 56));
}

u64 IntLexer(u64 work) {
    u64 sum = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        sum += Lex(text, TextBytes);
    }
    return sum;
}

__attribute__((noinline)) u64 Fibonacci(u32 n) {
    return n < 2 ? n : Fibonacci(n - 1) + Fibonacci(n - 2);
}

u64 IntFibonacci(u64 work) {
    u64 sum = 0;
    for (u64 pass = 0; pass < work; ++pass) sum += Fibonacci(Opaque(24u));
    return sum;
}

u64 IntHistogram(u64 work) {
    const u8* bytes = Bytes(source);
    for (u32 i = 0; i < 256; ++i) histogram[i] = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        for (u64 i = 0; i < ByteCount; ++i) ++histogram[bytes[i]];
    }
    u64 hash = 0;
    for (u32 i = 0; i < 256; ++i) hash = hash * 31 + histogram[i];
    return hash;
}

u64 MemcpyLarge(u64 work) {
    for (u64 pass = 0; pass < work; ++pass) BenchCopy(target, source, ByteCount);
    return Sample(target, WordCount);
}

u64 MemcpySmall(u64 work) {
    u8* to = Bytes(target);
    const u8* from = Bytes(source);
    for (u64 pass = 0; pass < work; ++pass)
        for (u64 offset = 0; offset < ByteCount; offset += SmallCopy) BenchCopy(to + offset, from + offset, SmallCopy);
    return Sample(target, WordCount);
}

u64 MemsetLarge(u64 work) {
    for (u64 pass = 0; pass < work; ++pass) BenchFill(target, static_cast<int>(pass & 0xff), ByteCount);
    return Sample(target, WordCount);
}

u64 CopyScalar(u64 work) {
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
#pragma clang loop vectorize(disable)
        for (u64 i = 0; i < WordCount; ++i) target[i] = InRegister(source[i]);
    }
    return Sample(target, WordCount);
}

u64 CopyVector(u64 work) {
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        for (u64 i = 0; i < WordCount; ++i) target[i] = source[i];
    }
    return Sample(target, WordCount);
}

void RepMovsb(void* destination, const void* from, u64 size) {
#if defined(__x86_64__)
    asm volatile("rep movsb" : "+D"(destination), "+S"(from), "+c"(size) : : "memory");
#else
    BenchCopy(destination, from, size);
#endif
}

u64 RepMovsbCopy(u64 work) {
    for (u64 pass = 0; pass < work; ++pass) RepMovsb(target, source, ByteCount);
    return Sample(target, WordCount);
}

void PrepareFloats() {
    for (u64 i = 0; i < FloatCount; ++i) {
        floatX[i] = static_cast<float>(static_cast<i32>(i % 17) - 8);
        floatY[i] = static_cast<float>(static_cast<i32>(i % 13));
    }
}

template <typename T>
inline T Load(const void* address) {
    T value;
    __builtin_memcpy(&value, address, sizeof(value));
    return value;
}

template <typename T>
inline void Store(void* address, T value) {
    __builtin_memcpy(address, &value, sizeof(value));
}

void Saxpy(float a, const float* __restrict x, float* __restrict y, u64 count) {
    for (u64 i = 0; i < count; ++i) y[i] = a * x[i] + y[i];
}

u64 FloatDigest(const float* values, u64 count) {
    u64 hash = 0;
    for (u64 i = 0; i < count; ++i) hash = hash * 31 + static_cast<u64>(static_cast<i64>(values[i]));
    return hash;
}

u64 SimdSaxpy(u64 work) {
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        Saxpy(3.0f, floatX, floatY, FloatCount);
    }
    return FloatDigest(floatY, FloatCount);
}

float Dot(const float* x, const float* y, u64 count) {
    VFloat a0 = {}, a1 = {}, a2 = {}, a3 = {};
    for (u64 i = 0; i < count; i += 4 * FloatLanes) {
        a0 += Load<VFloat>(x + i) * Load<VFloat>(y + i);
        a1 += Load<VFloat>(x + i + FloatLanes) * Load<VFloat>(y + i + FloatLanes);
        a2 += Load<VFloat>(x + i + 2 * FloatLanes) * Load<VFloat>(y + i + 2 * FloatLanes);
        a3 += Load<VFloat>(x + i + 3 * FloatLanes) * Load<VFloat>(y + i + 3 * FloatLanes);
    }
    const VFloat all = (a0 + a1) + (a2 + a3);
    float sum = 0;
    for (u64 lane = 0; lane < FloatLanes; ++lane) sum += all[lane];
    return sum;
}

u64 SimdDot(u64 work) {
    i64 sum = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        sum += static_cast<i64>(Dot(floatX, floatY, FloatCount));
    }
    return static_cast<u64>(sum);
}

F4 matrix[4] = {{1, -2, 3, 0}, {0, 1, -1, 2}, {2, 0, 1, -3}, {-1, 3, 0, 1}};

void PrepareVectors() {
    for (u64 i = 0; i < FloatCount; ++i) vectors[i] = static_cast<float>(static_cast<i32>((i * 7) % 9) - 4);
}

u64 SimdMat4(u64 work) {
    const F4 c0 = matrix[0], c1 = matrix[1], c2 = matrix[2], c3 = matrix[3];
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        for (u64 i = 0; i < FloatCount; i += 4) {
            const F4 v = Load<F4>(vectors + i);
            Store(transformed + i, c0 * v[0] + c1 * v[1] + c2 * v[2] + c3 * v[3]);
        }
    }
    return FloatDigest(transformed, FloatCount);
}

constexpr u64 NeedleCount = 4;
constexpr u8 FirstNeedle = 200;

void PrepareHaystack() {
    u8* bytes = Bytes(source);
    for (u64 i = 0; i < ByteCount; i += 8) {
        const u64 random = Mix(i + 3);
        for (u64 b = 0; b < 8; ++b) bytes[i + b] = static_cast<u8>(((random >> (8 * b)) & 0xff) % FirstNeedle);
    }
    for (u64 k = 0; k < NeedleCount; ++k) bytes[ByteCount - 1 - 4099 * k - 977 * (k & 1)] = static_cast<u8>(FirstNeedle + k);
}

u64 FindByte(const u8* bytes, u64 size, u8 needle) {
    const VByte match = VByte {} + needle;
    for (u64 i = 0; i < size; i += sizeof(VByte)) {
        if (__builtin_reduce_or(Load<VByte>(bytes + i) == match) != 0) {
            for (u64 j = i; j < size; ++j)
                if (bytes[j] == needle) return j;
        }
    }
    return size;
}

u64 SimdMemchr(u64 work) {
    u64 sum = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        for (u64 k = 0; k < NeedleCount; ++k) sum += FindByte(Bytes(source), ByteCount, static_cast<u8>(FirstNeedle + k));
    }
    return sum;
}

void PrepareSamples() {
    for (u64 i = 0; i < FloatCount; ++i) floatX[i] = static_cast<float>(static_cast<i32>(i % 193) - 96) / 64.0f;
}

void ToPcm(const float* __restrict input, i16* __restrict output, u64 count) {
    for (u64 i = 0; i < count; ++i) {
        i32 value = static_cast<i32>(input[i] * 32767.0f);
        value = value < -32768 ? -32768 : value;
        value = value > 32767 ? 32767 : value;
        output[i] = static_cast<i16>(value);
    }
}

u64 SimdPcm16(u64 work) {
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        ToPcm(floatX, samples, FloatCount);
    }
    u64 hash = 0;
    for (u64 i = 0; i < FloatCount; ++i) hash = hash * 31 + static_cast<u16>(samples[i]);
    return hash;
}

void Blur(const u8* __restrict input, u8* __restrict output, u64 count) {
    for (u64 i = 1; i + 1 < count; ++i) output[i] = static_cast<u8>((input[i - 1] + 2 * input[i] + input[i + 1] + 2) >> 2);
}

u64 SimdBlur(u64 work) {
    for (u64 pass = 0; pass < work; ++pass) {
        Clobber();
        Blur(Bytes(source), Bytes(target), ByteCount);
    }
    return Sample(target, WordCount);
}

struct alignas(64) Counter {
    u64 value;
};

Counter shared;
Counter privateCounters[MaxThreads];
Counter lockWord;
Counter guardedCount;
Counter guardedSum;
Counter startFlag;
Counter readyCount;
Counter doneCount;

using ThreadBody = u64 (*)(u32 index, u64 iterations);

struct ThreadWork {
    ThreadBody body;
    u64 iterations;
    u32 index;
    u64 result;
};

void PrepareAtomics() {
    shared.value = 0;
    for (u32 i = 0; i < MaxThreads; ++i) privateCounters[i].value = 0;
    lockWord.value = 0;
    guardedCount.value = 0;
    guardedSum.value = 0;
}

void* ThreadEntry(void* argument) {
    auto* work = static_cast<ThreadWork*>(argument);
    __atomic_fetch_add(&readyCount.value, 1, __ATOMIC_ACQ_REL);
    while (__atomic_load_n(&startFlag.value, __ATOMIC_ACQUIRE) == 0) Pause();
    work->result = work->body(work->index, work->iterations);
    __atomic_fetch_add(&doneCount.value, 1, __ATOMIC_ACQ_REL);
    return nullptr;
}

u64 RunThreads(u32 threads, u64 total, ThreadBody body) {
    ThreadWork work[MaxThreads];
    void* handles[MaxThreads];
    __atomic_store_n(&startFlag.value, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&readyCount.value, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&doneCount.value, 0, __ATOMIC_RELEASE);
    const u64 each = total / threads;
    for (u32 i = 0; i < threads; ++i) work[i] = {body, each, i, 0};
    u32 started = 1;
    for (u32 i = 1; i < threads; ++i, ++started)
        if (!BenchSpawn(&handles[i], ThreadEntry, &work[i])) break;
    while (__atomic_load_n(&readyCount.value, __ATOMIC_ACQUIRE) != started - 1) Pause();
    const u64 start = BenchNow();
    __atomic_store_n(&startFlag.value, 1, __ATOMIC_RELEASE);
    work[0].result = body(0, each);
    while (__atomic_load_n(&doneCount.value, __ATOMIC_ACQUIRE) != started - 1) Pause();
    benchElapsed = BenchNow() - start;
    for (u32 i = 1; i < started; ++i) BenchJoin(handles[i]);
    if (started != threads) return 0;
    u64 sum = 0;
    for (u32 i = 0; i < threads; ++i) sum += work[i].result;
    return sum;
}

u64 XaddBody(u32, u64 iterations) {
    u64 seen = 0;
    for (u64 i = 0; i < iterations; ++i) seen += __atomic_fetch_add(&shared.value, 1, __ATOMIC_SEQ_CST);
    return seen;
}

u64 CasBody(u32, u64 iterations) {
    u64 seen = 0;
    for (u64 i = 0; i < iterations; ++i) {
        u64 old = __atomic_load_n(&shared.value, __ATOMIC_RELAXED);
        while (!__atomic_compare_exchange_n(&shared.value, &old, old + 1, false, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
        }
        seen += old;
    }
    return seen;
}

u64 LockBody(u32 index, u64 iterations) {
    for (u64 i = 0; i < iterations; ++i) {
        while (__atomic_exchange_n(&lockWord.value, 1, __ATOMIC_ACQUIRE) != 0)
            while (__atomic_load_n(&lockWord.value, __ATOMIC_RELAXED) != 0) Pause();
        guardedCount.value += 1;
        guardedSum.value += i ^ index;
        __atomic_store_n(&lockWord.value, 0, __ATOMIC_RELEASE);
    }
    return 0;
}

u64 PrivateBody(u32 index, u64 iterations) {
    u64 seen = 0;
    for (u64 i = 0; i < iterations; ++i) seen += __atomic_fetch_add(&privateCounters[index].value, 1, __ATOMIC_SEQ_CST);
    return seen;
}

template <u32 Threads, ThreadBody Body>
u64 Atomic(u64 work) {
    const u64 seen = RunThreads(Threads, work, Body);
    u64 counters = 0;
    for (u32 i = 0; i < MaxThreads; ++i) counters = counters * 31 + privateCounters[i].value;
    return Mix(seen + Mix(shared.value) + Mix(counters) + Mix(guardedCount.value) + Mix(guardedSum.value + 1));
}

u64 BridgeCall(u64 work) {
    u64 sum = 0;
    for (u64 i = 0; i < work; ++i) sum += BenchTrivial();
    benchExtra = work;
    return sum;
}

u64 compareCalls = 0;

int CountedCompare(const void* left, const void* right) {
    ++compareCalls;
    return CompareNumbers(left, right);
}

u64 BridgeQsort(u64 work) {
    compareCalls = 0;
    u64 sum = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        Scramble(BridgeSortCount, pass);
        BenchSort(sorted, BridgeSortCount, sizeof(sorted[0]), CountedCompare);
        sum = sum * 31 + SortedDigest(sorted, BridgeSortCount);
    }
    benchExtra = compareCalls;
    return sum;
}

void* EmptyThread(void* argument) {
    return argument;
}

u64 BridgeThread(u64 work) {
    u64 joined = 0;
    for (u64 pass = 0; pass < work; ++pass) {
        void* handle = nullptr;
        if (!BenchSpawn(&handle, EmptyThread, nullptr)) break;
        BenchJoin(handle);
        ++joined;
    }
    benchExtra = joined;
    return joined;
}

}

const BenchKernel BenchKernels[] = {
    {"int_sum", PrepareNumbers, IntSum, 2048, 1},
    {"int_hash", PrepareWords, IntHash, 256, 1},
    {"int_mix", PrepareWords, IntMix, 1024, 1},
    {"int_crc32", PrepareCrc, IntCrc32, 128, 1},
    {"int_sort", PrepareSort, IntSort, 32, 1},
    {"int_sort_fnptr", PrepareSort, IntSortPointer, 16, 1},
    {"int_interp", nullptr, IntInterpreter, 1 << 20, 1},
    {"int_lexer", PrepareText, IntLexer, 160, 1},
    {"int_fib", nullptr, IntFibonacci, 128, 1},
    {"int_histogram", PrepareWords, IntHistogram, 128, 1},
    {"mem_memcpy_large", PrepareWords, MemcpyLarge, 4096, 1},
    {"mem_memcpy_small", PrepareWords, MemcpySmall, 128, 1},
    {"mem_memset_large", nullptr, MemsetLarge, 4096, 1},
    {"mem_copy_scalar", PrepareWords, CopyScalar, 4096, 1},
    {"mem_copy_vector", PrepareWords, CopyVector, 4096, 1},
    {"mem_rep_movsb", PrepareWords, RepMovsbCopy, 4096, 1},
    {"simd_saxpy", PrepareFloats, SimdSaxpy, 16384, 1},
    {"simd_dot", PrepareFloats, SimdDot, 16384, 1},
    {"simd_mat4", PrepareVectors, SimdMat4, 8192, 1},
    {"simd_memchr", PrepareHaystack, SimdMemchr, 512, 1},
    {"simd_pcm16", PrepareSamples, SimdPcm16, 8192, 1},
    {"simd_blur", PrepareWords, SimdBlur, 1024, 1},
    {"atomic_xadd_t1", PrepareAtomics, Atomic<1, XaddBody>, 1 << 22, 1},
    {"atomic_xadd_t2", PrepareAtomics, Atomic<2, XaddBody>, 1 << 20, 2},
    {"atomic_xadd_t4", PrepareAtomics, Atomic<4, XaddBody>, 1 << 20, 4},
    {"atomic_cas_t1", PrepareAtomics, Atomic<1, CasBody>, 1 << 22, 1},
    {"atomic_cas_t2", PrepareAtomics, Atomic<2, CasBody>, 1 << 20, 2},
    {"atomic_cas_t4", PrepareAtomics, Atomic<4, CasBody>, 1 << 20, 4},
    {"atomic_spinlock_t1", PrepareAtomics, Atomic<1, LockBody>, 1 << 21, 1},
    {"atomic_spinlock_t2", PrepareAtomics, Atomic<2, LockBody>, 1 << 20, 2},
    {"atomic_spinlock_t4", PrepareAtomics, Atomic<4, LockBody>, 1 << 20, 4},
    {"atomic_private_t1", PrepareAtomics, Atomic<1, PrivateBody>, 1 << 22, 1},
    {"atomic_private_t2", PrepareAtomics, Atomic<2, PrivateBody>, 1 << 22, 2},
    {"atomic_private_t4", PrepareAtomics, Atomic<4, PrivateBody>, 1 << 22, 4},
    {"bridge_call", nullptr, BridgeCall, 1 << 18, 1},
    {"bridge_qsort", PrepareSort, BridgeQsort, 16, 1},
    {"bridge_thread", nullptr, BridgeThread, 32, 2},
};

const u32 BenchKernelCount = sizeof(BenchKernels) / sizeof(BenchKernels[0]);
