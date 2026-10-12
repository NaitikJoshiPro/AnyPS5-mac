#include "prx/libc/include/X87Extended.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

namespace {

int failures = 0;

void Fail(const std::string& message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    if (++failures > 20) std::exit(1);
}

std::string Bits(X87Extended value) {
    char text[40];
    std::snprintf(text, sizeof(text), "%04x:%016llx", value.signExponent, static_cast<unsigned long long>(value.significand));
    return text;
}

std::string Host(const std::string& format, double value) {
    char text[512];
    std::snprintf(text, sizeof(text), format.c_str(), value);
    return text;
}

bool KeepsTrailingZeros(const std::string& spec, char conversion, const std::string& text) {
    if ((conversion | 0x20) != 'g' || spec.find('#') != std::string::npos) return false;
    const std::string mantissa = text.substr(0, text.find_first_of("eE"));
    const std::size_t last = mantissa.find_last_not_of(' ');
    return mantissa.find('.') != std::string::npos && last != std::string::npos && mantissa[last] == '0';
}

void CheckAgainstHostDoubles() {
    std::mt19937_64 random(12345);
    std::vector<double> values = {0.0, -0.0, 1.0, -1.0, 0.5, 2.5, 0.125, 0.375, 1e-300, 4.9e-324, 2.2250738585072014e-308,
        1.7976931348623157e308, 123456.789, 999999.5, 9.9999995, 0.000123456, 1e15, 1e16, 1e22, 1e23, 9.5, 0.05, 1e-5};
    for (int i = 0; i < 64; ++i) values.push_back(static_cast<double>(i) / 16.0);
    for (int i = 0; i < 64; ++i) values.push_back(-static_cast<double>(i) * 1000.5);
    for (int i = 0; i < 600; ++i) {
        std::uint64_t bits = random();
        double value;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) continue;
        values.push_back(value);
    }
    const char* specs[] = {"%", "%.0", "%.1", "%.3", "%.10", "%.17", "%.25", "%#.0", "%#", "%+", "% ", "%012.4", "%-12.4",
        "%+#015.6", "%.40", "%05"};
    for (const double value : values) {
        for (const char* spec : specs) {
            for (const char conversion : {'f', 'F', 'e', 'E', 'g', 'G'}) {
                const std::string expected = Host(std::string(spec) + conversion, value);
                if (KeepsTrailingZeros(spec, conversion, expected)) continue;
                const std::string actual = LibcDetail::FormatX87(LibcDetail::X87FromDouble(value), spec, conversion);
                if (actual != expected)
                    Fail("format " + std::string(spec) + conversion + " of " + Host("%a", value) + ": expected " + expected + ", got " + actual);
            }
        }
    }
}

X87Extended Parse(const char* text, int* error = nullptr, std::size_t* used = nullptr) {
    errno = 0;
    char* end = nullptr;
    const X87Extended value = LibcDetail::ParseX87(text, &end);
    if (error != nullptr) *error = errno;
    if (used != nullptr) *used = static_cast<std::size_t>(end - text);
    return value;
}

void CheckRoundTrips() {
    std::mt19937_64 random(54321);
    for (int i = 0; i < 1500; ++i) {
        X87Extended value;
        if (i % 10 == 0) {
            value = {random() >> 1, 0};
        } else {
            value = {random() | (std::uint64_t(1) << 63), static_cast<std::uint16_t>(1 + random() % 0x7ffe)};
        }
        if (random() & 1) value.signExponent |= 0x8000;
        const std::string decimal = LibcDetail::FormatX87(value, "%.20", 'e');
        if (Parse(decimal.c_str()) != value) Fail("decimal round trip of " + Bits(value) + " through " + decimal + " gave " + Bits(Parse(decimal.c_str())));
        if ((value.signExponent & 0x7fff) != 0) {
            const std::string hexadecimal = LibcDetail::FormatX87(value, "%", 'a');
            if (Parse(hexadecimal.c_str()) != value) Fail("hexadecimal round trip of " + Bits(value) + " through " + hexadecimal);
        }
    }
}

struct ParseCase {
    const char* text;
    std::uint64_t significand;
    std::uint16_t signExponent;
    std::size_t used;
    bool range;
};

void CheckParsing() {
    const std::string tieDown = "1.0000000000000000000542101086242752217003726400434970855712890625";
    const std::string tieUp = "1.0000000000000000001626303258728256651011179201304912567138671875";
    const std::string aboveTie = tieDown + "000000000000000000001";
    const std::string longZeros = "0." + std::string(29999, '0') + "1e30000";
    const ParseCase cases[] = {
        {"1.0000000000000000001!", 0x8000000000000001, 0x3fff, 21, false},
        {tieDown.c_str(), 0x8000000000000000, 0x3fff, tieDown.size(), false},
        {tieUp.c_str(), 0x8000000000000002, 0x3fff, tieUp.size(), false},
        {aboveTie.c_str(), 0x8000000000000001, 0x3fff, aboveTie.size(), false},
        {"0x1.0000000000000001p0", 0x8000000000000000, 0x3fff, 22, false},
        {"0x1.0000000000000003p0", 0x8000000000000002, 0x3fff, 22, false},
        {"0x1.00000000000000010000001p0", 0x8000000000000001, 0x3fff, 29, false},
        {"0.1", 0xcccccccccccccccd, 0x3ffb, 3, false},
        {"3.14159265358979323846264338327950288", 0xc90fdaa22168c235, 0x4000, 37, false},
        {"123456789012345678901234567890", 0xc77487fb61b9f077, 0x405f, 30, false},
        {"1e4000", 0xd1ba8323fe558c61, 0x73e6, 6, false},
        {"1e-4000", 0x9c3d73864f3805c0, 0x0c17, 7, false},
        {"1.18973149535723176502e+4932", 0xffffffffffffffff, 0x7ffe, 28, false},
        {"1.2e4932", 0x8000000000000000, 0x7fff, 8, true},
        {"-1e99999", 0x8000000000000000, 0xffff, 8, true},
        {"3.64519953188247460253e-4951", 1, 0, 28, true},
        {"0x1p-16446", 0, 0, 10, true},
        {"0x1.0000000000000001p-16446", 1, 0, 27, true},
        {"0x1p-16382", 0x8000000000000000, 1, 10, false},
        {"0x0.fffffffffffffffep-16382", 0x7fffffffffffffff, 0, 27, false},
        {"1e-5000", 0, 0, 7, true},
        {"0e99999", 0, 0, 7, false},
        {"  -0", 0, 0x8000, 4, false},
        {"+.5", 0x8000000000000000, 0x3ffe, 3, false},
        {"5.", 0xa000000000000000, 0x4001, 2, false},
        {"1e", 0x8000000000000000, 0x3fff, 1, false},
        {"1e+", 0x8000000000000000, 0x3fff, 1, false},
        {"1E+5x", 0xc350000000000000, 0x400f, 4, false},
        {"0x", 0, 0, 1, false},
        {"0xg", 0, 0, 1, false},
        {"0X.8P1", 0x8000000000000000, 0x3fff, 6, false},
        {"inf", 0x8000000000000000, 0x7fff, 3, false},
        {"-Infinityx", 0x8000000000000000, 0xffff, 9, false},
        {"infinit", 0x8000000000000000, 0x7fff, 3, false},
        {"nan", 0xc000000000000000, 0x7fff, 3, false},
        {"-NaN()", 0xc000000000000000, 0xffff, 6, false},
        {"nan(", 0xc000000000000000, 0x7fff, 3, false},
        {longZeros.c_str(), 0x8000000000000000, 0x3fff, longZeros.size(), false},
    };
    for (const auto& item : cases) {
        int error = 0;
        std::size_t used = 0;
        const X87Extended value = Parse(item.text, &error, &used);
        const std::string name = std::string(item.text).substr(0, 60);
        if (value != X87Extended{item.significand, item.signExponent}) Fail("parse " + name + ": got " + Bits(value));
        if (used != item.used) Fail("parse " + name + ": used " + std::to_string(used));
        if ((error == ERANGE) != item.range) Fail("parse " + name + ": errno " + std::to_string(error));
    }
    for (const char* text : {"", ".", "e5", "-", " +x", "-.e1"}) {
        int error = 0;
        std::size_t used = 1;
        const X87Extended value = Parse(text, &error, &used);
        if (value != X87Extended{0, 0} || used != 0 || error != 0) Fail(std::string("no conversion for \"") + text + "\"");
    }
}

struct FormatCase {
    std::uint64_t significand;
    std::uint16_t signExponent;
    const char* spec;
    char conversion;
    const char* expected;
};

void CheckFormatting() {
    const FormatCase cases[] = {
        {0x8000000000000001, 0x3fff, "%.25", 'f', "1.0000000000000000001084202"},
        {0x8000000000000001, 0x3fff, "%.21", 'g', "1.00000000000000000011"},
        {0x8000000000000001, 0x3fff, "%.20", 'e', "1.00000000000000000011e+00"},
        {0x8000000000000001, 0x3fff, "%-10.1", 'e', "1.0e+00   "},
        {0x9c54000000000000, 0x400c, "%.4", 'g', "1e+04"},
        {0x9c54000000000000, 0xc00c, "%012.4", 'g', "-0000001e+04"},
        {0xffffffffffffffff, 0x7ffe, "%.20", 'e', "1.18973149535723176502e+4932"},
        {0xffffffffffffffff, 0x7ffe, "%.20", 'G', "1.189731495357231765E+4932"},
        {1, 0, "%.20", 'e', "3.64519953188247460253e-4951"},
        {1, 0, "%", 'g', "3.6452e-4951"},
        {0x8000000000000000, 0x0001, "%.20", 'e', "3.36210314311209350626e-4932"},
        {0xc000000000000000, 0xbfff, "%+012.3", 'f', "-0000001.500"},
        {0x8000000000000000, 0x3fff, "%", 'a', "0x8p-3"},
        {0x8000000000000000, 0x3fff, "%#", 'a', "0x8.p-3"},
        {0x8000000000000000, 0x3fff, "%.1", 'a', "0x8.0p-3"},
        {0x8000000000000000, 0x3fff, "%", 'A', "0X8P-3"},
        {0x8000000000000001, 0x3fff, "%", 'a', "0x8.000000000000001p-3"},
        {0xffffffffffffffff, 0x7ffe, "%", 'a', "0xf.fffffffffffffffp+16380"},
        {0xf800000000000000, 0x3fff, "%.0", 'a', "0x1p+1"},
        {0xe800000000000000, 0x3fff, "%.0", 'a', "0xep-3"},
        {0x9800000000000000, 0x3fff, "%.0", 'a', "0xap-3"},
        {0x8800000000000000, 0x3fff, "%.0", 'a', "0x8p-3"},
        {0x8080000000000000, 0x3fff, "%.1", 'a', "0x8.0p-3"},
        {0x8180000000000000, 0x3fff, "%.1", 'a', "0x8.2p-3"},
        {0x8000000000000000, 0x3fff, "%.20", 'a', "0x8.00000000000000000000p-3"},
        {0x8000000000000000, 0x3fff, "%+012", 'a', "+0x000008p-3"},
        {0, 0, "%", 'a', "0x0p+0"},
        {0, 0x8000, "%.2", 'a', "-0x0.00p+0"},
        {0, 0x8000, "%", 'f', "-0.000000"},
        {0, 0, "%", 'g', "0"},
        {0, 0, "%#", 'g', "0.00000"},
        {0x8000000000000000, 0x7fff, "%", 'f', "inf"},
        {0x8000000000000000, 0xffff, "%6", 'F', "  -INF"},
    };
    for (const auto& item : cases) {
        const std::string actual = LibcDetail::FormatX87({item.significand, item.signExponent}, item.spec, item.conversion);
        if (actual != item.expected)
            Fail("format " + std::string(item.spec) + item.conversion + " of " + Bits({item.significand, item.signExponent}) + ": expected " + item.expected + ", got " + actual);
    }
}

#if defined(__x86_64__) && defined(__linux__)
void CheckAgainstHostLongDoubles() {
    std::mt19937_64 random(777);
    for (int i = 0; i < 500; ++i) {
        const X87Extended value = {random() | (std::uint64_t(1) << 63), static_cast<std::uint16_t>((1 + random() % 0x7ffe) | (random() & 0x8000))};
        long double host;
        std::memcpy(&host, &value, 10);
        for (const char* spec : {"%", "%.0", "%.3", "%.25", "%#.0", "%+012.4"}) {
            for (const char conversion : {'f', 'e', 'g', 'a'}) {
                if (conversion == 'f' && (value.signExponent & 0x7fff) > 16383 + 400) continue;
                char expected[8192];
                std::snprintf(expected, sizeof(expected), (std::string(spec) + 'L' + conversion).c_str(), host);
                const std::string actual = LibcDetail::FormatX87(value, spec, conversion);
                if (actual != expected) Fail("format " + std::string(spec) + 'L' + conversion + " of " + Bits(value) + ": host " + expected + ", got " + actual);
                X87Extended parsed{};
                const long double reparsed = std::strtold(expected, nullptr);
                std::memcpy(&parsed, &reparsed, 10);
                if (Parse(expected) != parsed) Fail(std::string("parse ") + expected + ": host " + Bits(parsed) + ", got " + Bits(Parse(expected)));
            }
        }
    }
}
#endif

}

int main() {
    CheckParsing();
    CheckFormatting();
    CheckRoundTrips();
    CheckAgainstHostDoubles();
#if defined(__x86_64__) && defined(__linux__)
    CheckAgainstHostLongDoubles();
#endif
    if (failures != 0) return 1;
    std::puts("x87 extended precision checks passed");
    return 0;
}
