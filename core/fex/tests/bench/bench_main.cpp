#include "bench.h"

namespace {

constexpr u64 MaxRepetitions = 64;

class Line {
public:
    Line() {
        buffer[0] = 0;
    }

    void Append(const char* text) {
        while (*text != 0 && size + 1 < sizeof(buffer)) buffer[size++] = *text++;
        buffer[size] = 0;
    }

    void Append(u64 value) {
        char digits[24];
        u32 count = 0;
        do {
            digits[count++] = static_cast<char>('0' + value % 10);
            value /= 10;
        } while (value != 0);
        while (count != 0 && size + 1 < sizeof(buffer)) buffer[size++] = digits[--count];
        buffer[size] = 0;
    }

    void AppendHex(u64 value) {
        for (int shift = 60; shift >= 0 && size + 1 < sizeof(buffer); shift -= 4) buffer[size++] = "0123456789abcdef"[(value >> shift) & 15];
        buffer[size] = 0;
    }

    const char* Text() const {
        return buffer;
    }

private:
    char buffer[2048];
    u32 size = 0;
};

bool Same(const char* left, const char* right) {
    while (*left != 0 && *left == *right) {
        ++left;
        ++right;
    }
    return *left == *right;
}

bool Parse(const char* text, u64* value) {
    if (*text == 0) return false;
    u64 result = 0;
    for (; *text != 0; ++text) {
        if (*text < '0' || *text > '9') return false;
        result = result * 10 + static_cast<u64>(*text - '0');
    }
    *value = result;
    return true;
}

bool Selected(const char* name, const char* selection) {
    const char* token = selection;
    for (;;) {
        const char* end = token;
        while (*end != 0 && *end != ',') ++end;
        const u64 length = static_cast<u64>(end - token);
        bool matches = length == 3 && token[0] == 'a' && token[1] == 'l' && token[2] == 'l';
        if (!matches && length != 0 && token[length - 1] == '*') {
            matches = true;
            for (u64 i = 0; i + 1 < length && matches; ++i) matches = name[i] == token[i];
        } else if (!matches) {
            u64 i = 0;
            while (i < length && name[i] == token[i]) ++i;
            matches = i == length && name[i] == 0;
        }
        if (matches) return true;
        if (*end == 0) return false;
        token = end + 1;
    }
}

void List() {
    for (u32 index = 0; index < BenchKernelCount; ++index) {
        const BenchKernel& kernel = BenchKernels[index];
        Line line;
        line.Append("KERNEL ");
        line.Append(kernel.name);
        line.Append(" work=");
        line.Append(kernel.work);
        line.Append(" threads=");
        line.Append(static_cast<u64>(kernel.threads));
        BenchWrite(line.Text());
    }
}

bool Measure(const BenchKernel& kernel, u64 permille, u64 repetitions) {
    u64 work = kernel.work * permille / 1000;
    if (work == 0) work = 1;
    u64 times[MaxRepetitions];
    u64 cpuTimes[MaxRepetitions];
    u64 first = 0;
    u64 firstExtra = 0;
    bool stable = true;
    for (u64 repetition = 0; repetition < repetitions; ++repetition) {
        if (kernel.prepare != nullptr) kernel.prepare();
        benchExtra = 0;
        benchElapsed = 0;
        const u64 cpuStart = BenchCpuNow();
        const u64 start = BenchNow();
        const u64 sum = kernel.run(work);
        const u64 end = BenchNow();
        cpuTimes[repetition] = BenchCpuNow() - cpuStart;
        times[repetition] = benchElapsed != 0 ? benchElapsed : end - start;
        if (repetition == 0) {
            first = sum;
            firstExtra = benchExtra;
        } else if (sum != first || benchExtra != firstExtra) {
            stable = false;
        }
    }
    Line line;
    line.Append("BENCH ");
    line.Append(kernel.name);
    line.Append(" threads=");
    line.Append(static_cast<u64>(kernel.threads));
    line.Append(" work=");
    line.Append(work);
    line.Append(" sum=");
    line.AppendHex(first);
    line.Append(" extra=");
    line.Append(firstExtra);
    line.Append(" stable=");
    line.Append(stable ? "1" : "0");
    line.Append(" ns=");
    for (u64 repetition = 0; repetition < repetitions; ++repetition) {
        if (repetition != 0) line.Append(",");
        line.Append(times[repetition]);
    }
    line.Append(" cpu=");
    for (u64 repetition = 0; repetition < repetitions; ++repetition) {
        if (repetition != 0) line.Append(",");
        line.Append(cpuTimes[repetition]);
    }
    BenchWrite(line.Text());
    return stable;
}

}

int BenchMain(int count, const char* const* arguments) {
    if (count < 2) {
        BenchWrite("usage: bench <list | all | name[,name...] | prefix*> [permille of the full work] [repetitions]");
        return 2;
    }
    if (Same(arguments[1], "list")) {
        List();
        return 0;
    }
    u64 permille = 1000;
    u64 repetitions = 8;
    if ((count > 2 && !Parse(arguments[2], &permille)) ||
        (count > 3 && (!Parse(arguments[3], &repetitions) || repetitions == 0 || repetitions > MaxRepetitions))) {
        BenchWrite("bench: permille and repetitions (1 to 64) are decimal numbers");
        return 2;
    }
    u32 matched = 0;
    int status = 0;
    for (u32 index = 0; index < BenchKernelCount; ++index) {
        if (!Selected(BenchKernels[index].name, arguments[1])) continue;
        ++matched;
        if (!Measure(BenchKernels[index], permille, repetitions)) status = 3;
    }
    if (matched == 0) {
        BenchWrite("bench: no kernel matches");
        return 2;
    }
    return status;
}
