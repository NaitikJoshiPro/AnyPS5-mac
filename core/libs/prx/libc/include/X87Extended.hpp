#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_X87EXTENDED_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_X87EXTENDED_HPP

#include "General.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct alignas(16) X87Extended {
    std::uint64_t significand;
    std::uint16_t signExponent;

    bool operator==(const X87Extended&) const = default;
};
static_assert(sizeof(X87Extended) == 16);

namespace LibcDetail {

inline X87Extended X87FromDouble(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = (bits >> 63) != 0 ? 0x8000 : 0;
    const unsigned exponent = static_cast<unsigned>(bits >> 52) & 0x7ffu;
    const std::uint64_t fraction = bits & ((std::uint64_t(1) << 52) - 1);
    if (exponent == 0x7ff) {
        const std::uint64_t significand = fraction != 0 ? (std::uint64_t(3) << 62) | (fraction << 11) : std::uint64_t(1) << 63;
        return {significand, static_cast<std::uint16_t>(sign | 0x7fff)};
    }
    if (exponent == 0) {
        if (fraction == 0) return {0, sign};
        const int shift = __builtin_clzll(fraction);
        return {fraction << shift, static_cast<std::uint16_t>(sign | (16383 - 1074 + 63 - shift))};
    }
    return {(std::uint64_t(1) << 63) | (fraction << 11), static_cast<std::uint16_t>(sign | (exponent - 1023 + 16383))};
}

}

#if defined(__x86_64__)
using GuestLongDouble = long double;
inline GuestLongDouble GuestLongDoubleFromDouble(double value) { return value; }
#else
using GuestLongDouble = X87Extended;
inline GuestLongDouble GuestLongDoubleFromDouble(double value) { return LibcDetail::X87FromDouble(value); }
#endif

namespace LibcDetail {

class X87BigUnsigned {
    std::vector<std::uint32_t> words;

    void Trim() {
        while (!words.empty() && words.back() == 0) words.pop_back();
    }

public:
    X87BigUnsigned() = default;
    explicit X87BigUnsigned(std::uint64_t value) {
        for (; value != 0; value >>= 32) words.push_back(static_cast<std::uint32_t>(value));
    }

    bool IsZero() const { return words.empty(); }

    std::size_t BitLength() const {
        return words.empty() ? 0 : (words.size() - 1) * 32 + (32 - static_cast<std::size_t>(__builtin_clz(words.back())));
    }

    bool Bit(std::size_t index) const {
        const std::size_t word = index / 32;
        return word < words.size() && ((words[word] >> (index % 32)) & 1u) != 0;
    }

    bool AnyBitBelow(std::size_t index) const {
        const std::size_t whole = std::min(index / 32, words.size());
        for (std::size_t i = 0; i < whole; ++i)
            if (words[i] != 0) return true;
        return whole < words.size() && index % 32 != 0 && (words[whole] & ((1u << (index % 32)) - 1)) != 0;
    }

    std::uint64_t Bits64(std::size_t low) const {
        std::uint64_t value = 0;
        for (std::size_t i = 64; i-- > 0;) value = (value << 1) | (Bit(low + i) ? 1u : 0u);
        return value;
    }

    void MultiplyAdd(std::uint32_t factor, std::uint32_t addend) {
        std::uint64_t carry = addend;
        for (auto& word : words) {
            const std::uint64_t product = static_cast<std::uint64_t>(word) * factor + carry;
            word = static_cast<std::uint32_t>(product);
            carry = product >> 32;
        }
        if (carry != 0) words.push_back(static_cast<std::uint32_t>(carry));
        Trim();
    }

    void MultiplyPow5(long long count) {
        for (; count >= 13; count -= 13) MultiplyAdd(1220703125u, 0);
        std::uint32_t factor = 1;
        for (; count > 0; --count) factor *= 5;
        MultiplyAdd(factor, 0);
    }

    void ShiftLeft(std::size_t bits) {
        if (words.empty()) return;
        const std::size_t part = bits % 32;
        if (part != 0) {
            std::uint32_t carry = 0;
            for (auto& word : words) {
                const std::uint32_t next = word >> (32 - part);
                word = (word << part) | carry;
                carry = next;
            }
            if (carry != 0) words.push_back(carry);
        }
        words.insert(words.begin(), bits / 32, 0u);
    }

    void ShiftRightOne() {
        for (std::size_t i = 0; i < words.size(); ++i)
            words[i] = (words[i] >> 1) | (i + 1 < words.size() ? words[i + 1] << 31 : 0u);
        Trim();
    }

    std::uint32_t DivideSmall(std::uint32_t divisor) {
        std::uint64_t remainder = 0;
        for (std::size_t i = words.size(); i-- > 0;) {
            const std::uint64_t current = (remainder << 32) | words[i];
            words[i] = static_cast<std::uint32_t>(current / divisor);
            remainder = current % divisor;
        }
        Trim();
        return static_cast<std::uint32_t>(remainder);
    }

    int Compare(const X87BigUnsigned& other) const {
        if (words.size() != other.words.size()) return words.size() < other.words.size() ? -1 : 1;
        for (std::size_t i = words.size(); i-- > 0;)
            if (words[i] != other.words[i]) return words[i] < other.words[i] ? -1 : 1;
        return 0;
    }

    void Subtract(const X87BigUnsigned& other) {
        std::uint64_t borrow = 0;
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint64_t subtrahend = (i < other.words.size() ? other.words[i] : 0u) + borrow;
            borrow = words[i] < subtrahend ? 1u : 0u;
            words[i] = static_cast<std::uint32_t>((static_cast<std::uint64_t>(words[i]) + (borrow << 32)) - subtrahend);
        }
        Trim();
    }

    std::string Decimal() const {
        if (words.empty()) return "0";
        X87BigUnsigned value = *this;
        std::vector<std::uint32_t> chunks;
        while (!value.IsZero()) chunks.push_back(value.DivideSmall(1000000000u));
        std::string text = std::to_string(chunks.back());
        for (std::size_t i = chunks.size() - 1; i-- > 0;) {
            char chunk[10];
            std::snprintf(chunk, sizeof(chunk), "%09u", chunks[i]);
            text += chunk;
        }
        return text;
    }
};

inline X87Extended RoundX87(const X87BigUnsigned& n, long long exponent, bool sticky) {
    constexpr long long LowestUnit = -16445;
    const long long length = static_cast<long long>(n.BitLength());
    long long unit = std::max(length - 64 + exponent, LowestUnit);
    const long long drop = unit - exponent;
    std::uint64_t significand;
    bool inexact = sticky;
    if (drop <= 0) {
        significand = n.Bits64(0) << -drop;
    } else {
        significand = n.Bits64(static_cast<std::size_t>(drop));
        const bool half = n.Bit(static_cast<std::size_t>(drop - 1));
        const bool below = sticky || n.AnyBitBelow(static_cast<std::size_t>(drop - 1));
        inexact = half || below;
        if (half && (below || (significand & 1u) != 0) && ++significand == 0) {
            significand = std::uint64_t(1) << 63;
            ++unit;
        }
    }
    if ((significand >> 63) == 0) {
        if (inexact) errno = ERANGE;
        return {significand, 0};
    }
    const long long field = unit + 63 + 16383;
    if (field >= 0x7fff) {
        errno = ERANGE;
        return {std::uint64_t(1) << 63, 0x7fff};
    }
    return {significand, static_cast<std::uint16_t>(field)};
}

inline bool X87StartsWith(const char* text, const char* word) {
    for (; *word != 0; ++text, ++word)
        if ((*text | 0x20) != *word) return false;
    return true;
}

inline int X87HexValue(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if ((character | 0x20) >= 'a' && (character | 0x20) <= 'f') return (character | 0x20) - 'a' + 10;
    return -1;
}

inline const char* X87Exponent(const char* cursor, char marker, long long& exponent) {
    if ((*cursor | 0x20) != marker) return cursor;
    const char* digits = cursor + 1;
    const bool negative = *digits == '-';
    if (*digits == '+' || *digits == '-') ++digits;
    if (*digits < '0' || *digits > '9') return cursor;
    long long value = 0;
    for (; *digits >= '0' && *digits <= '9'; ++digits) value = std::min(value * 10 + (*digits - '0'), 1000000000000LL);
    exponent += negative ? -value : value;
    return digits;
}

inline X87Extended ParseX87(const char* text, char** end) {
    const char* cursor = text;
    while (*cursor == ' ' || (*cursor >= '\t' && *cursor <= '\r')) ++cursor;
    const bool negative = *cursor == '-';
    if (*cursor == '+' || *cursor == '-') ++cursor;
    const auto finish = [&](X87Extended value, const char* stop) {
        if (end != nullptr) *end = const_cast<char*>(stop);
        if (negative) value.signExponent |= 0x8000;
        return value;
    };
    if (X87StartsWith(cursor, "inf")) {
        cursor += X87StartsWith(cursor, "infinity") ? 8 : 3;
        return finish({std::uint64_t(1) << 63, 0x7fff}, cursor);
    }
    if (X87StartsWith(cursor, "nan")) {
        cursor += 3;
        if (*cursor == '(') {
            const char* close = cursor + 1;
            while ((*close >= '0' && *close <= '9') || ((*close | 0x20) >= 'a' && (*close | 0x20) <= 'z') || *close == '_') ++close;
            if (*close == ')') {
                if (close != cursor + 1) NotImplemented_nid_no_patch("strtold NaN payload");
                cursor = close + 1;
            }
        }
        return finish({std::uint64_t(3) << 62, 0x7fff}, cursor);
    }
    X87BigUnsigned value;
    bool sticky = false;
    bool point = false;
    if (cursor[0] == '0' && (cursor[1] | 0x20) == 'x' &&
        (X87HexValue(cursor[2]) >= 0 || (cursor[2] == '.' && X87HexValue(cursor[3]) >= 0))) {
        long long exponent = 0;
        for (cursor += 2;; ++cursor) {
            if (*cursor == '.' && !point) { point = true; continue; }
            const int digit = X87HexValue(*cursor);
            if (digit < 0) break;
            if (value.BitLength() < 72) {
                value.MultiplyAdd(16, static_cast<std::uint32_t>(digit));
                if (point) exponent -= 4;
            } else {
                sticky |= digit != 0;
                if (!point) exponent += 4;
            }
        }
        cursor = X87Exponent(cursor, 'p', exponent);
        return finish(value.IsZero() ? X87Extended{0, 0} : RoundX87(value, exponent, sticky), cursor);
    }
    constexpr std::size_t MaxDigits = 16384;
    std::string digits;
    long long exponent = 0;
    bool any = false;
    for (;; ++cursor) {
        if (*cursor == '.' && !point) { point = true; continue; }
        if (*cursor < '0' || *cursor > '9') break;
        any = true;
        if (digits.empty() && *cursor == '0') {
            if (point) --exponent;
        } else if (digits.size() < MaxDigits) {
            digits += *cursor;
            if (point) --exponent;
        } else {
            sticky |= *cursor != '0';
            if (!point) ++exponent;
        }
    }
    if (!any) {
        if (end != nullptr) *end = const_cast<char*>(text);
        return {0, 0};
    }
    cursor = X87Exponent(cursor, 'e', exponent);
    if (digits.empty()) return finish({0, 0}, cursor);
    if (sticky) {
        digits += '1';
        --exponent;
    }
    const long long magnitude = exponent + static_cast<long long>(digits.size());
    if (magnitude - 1 >= 4933) {
        errno = ERANGE;
        return finish({std::uint64_t(1) << 63, 0x7fff}, cursor);
    }
    if (magnitude <= -4951) {
        errno = ERANGE;
        return finish({0, 0}, cursor);
    }
    for (const char digit : digits) value.MultiplyAdd(10, static_cast<std::uint32_t>(digit - '0'));
    if (exponent >= 0) {
        value.MultiplyPow5(exponent);
        value.ShiftLeft(static_cast<std::size_t>(exponent));
        return finish(RoundX87(value, 0, false), cursor);
    }
    X87BigUnsigned divisor(1);
    divisor.MultiplyPow5(-exponent);
    const long long shift = static_cast<long long>(value.BitLength()) - static_cast<long long>(divisor.BitLength()) - 68;
    if (shift >= 0) divisor.ShiftLeft(static_cast<std::size_t>(shift));
    else value.ShiftLeft(static_cast<std::size_t>(-shift));
    X87BigUnsigned step = divisor;
    step.ShiftLeft(68);
    unsigned __int128 quotient = 0;
    for (int bit = 68; bit >= 0; --bit) {
        quotient <<= 1;
        if (value.Compare(step) >= 0) {
            value.Subtract(step);
            quotient |= 1;
        }
        step.ShiftRightOne();
    }
    X87BigUnsigned whole(static_cast<std::uint64_t>(quotient >> 64));
    whole.ShiftLeft(32);
    whole.MultiplyAdd(1, static_cast<std::uint32_t>(static_cast<std::uint64_t>(quotient) >> 32));
    whole.ShiftLeft(32);
    whole.MultiplyAdd(1, static_cast<std::uint32_t>(quotient));
    return finish(RoundX87(whole, exponent + shift, !value.IsZero()), cursor);
}

inline std::string X87RoundDigits(const std::string& digits, long long count) {
    if (count >= static_cast<long long>(digits.size())) return digits + std::string(static_cast<std::size_t>(count) - digits.size(), '0');
    std::string kept = count > 0 ? digits.substr(0, static_cast<std::size_t>(count)) : std::string();
    if (count >= 0) {
        const char first = digits[static_cast<std::size_t>(count)];
        const bool rest = digits.find_first_not_of('0', static_cast<std::size_t>(count) + 1) != std::string::npos;
        const bool odd = !kept.empty() && ((kept.back() - '0') & 1) != 0;
        if (first > '5' || (first == '5' && (rest || odd))) {
            std::size_t i = kept.size();
            for (; i > 0 && kept[i - 1] == '9'; --i) kept[i - 1] = '0';
            if (i > 0) ++kept[i - 1];
            else kept.insert(kept.begin(), '1');
        }
    }
    return kept.empty() ? "0" : kept;
}

inline std::string X87Fixed(const std::string& digits, long long exponent, long long precision, bool point) {
    std::string rounded = X87RoundDigits(digits, static_cast<long long>(digits.size()) + exponent + precision);
    rounded.erase(0, std::min(rounded.find_first_not_of('0'), rounded.size() - 1));
    if (static_cast<long long>(rounded.size()) <= precision)
        rounded.insert(0, static_cast<std::size_t>(precision) + 1 - rounded.size(), '0');
    const std::size_t integer = rounded.size() - static_cast<std::size_t>(precision);
    std::string text = rounded.substr(0, integer);
    if (precision > 0 || point) text += '.';
    return text + rounded.substr(integer);
}

inline std::string X87Scientific(const std::string& digits, long long exponent, long long precision, bool point, char marker) {
    std::string rounded;
    long long power = 0;
    if (digits == "0") {
        rounded.assign(static_cast<std::size_t>(precision) + 1, '0');
    } else {
        rounded = X87RoundDigits(digits, precision + 1);
        power = static_cast<long long>(digits.size()) + exponent - 1;
        if (static_cast<long long>(rounded.size()) > precision + 1) {
            rounded.pop_back();
            ++power;
        }
    }
    std::string text(1, rounded[0]);
    if (precision > 0 || point) text += '.';
    text += rounded.substr(1);
    char suffix[24];
    std::snprintf(suffix, sizeof(suffix), "%c%c%02lld", marker, power < 0 ? '-' : '+', power < 0 ? -power : power);
    return text + suffix;
}

inline std::string X87Hexadecimal(std::uint64_t significand, unsigned field, long long precision, bool point, bool upper) {
    const char* const symbols = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    std::uint64_t leading = 0;
    std::uint64_t fraction = 0;
    long long power = 0;
    if (significand != 0) {
        if (field == 0) NotImplemented_nid_no_patch("x87 subnormal hexadecimal formatting");
        leading = significand >> 60;
        fraction = significand & ((std::uint64_t(1) << 60) - 1);
        power = static_cast<long long>(field) - 16383 - 3;
    }
    int count = 15;
    if (precision < 0) {
        while (count > 0 && ((fraction >> (60 - 4 * count)) & 0xf) == 0) --count;
    } else if (precision < 15) {
        count = static_cast<int>(precision);
        const int drop = 4 * (15 - count);
        const std::uint64_t half = std::uint64_t(1) << (drop - 1);
        const std::uint64_t rest = fraction & ((std::uint64_t(1) << drop) - 1);
        fraction >>= drop;
        const bool odd = ((count > 0 ? fraction : leading) & 1u) != 0;
        if (rest > half || (rest == half && odd)) {
            if (++fraction == std::uint64_t(1) << (4 * count)) {
                fraction = 0;
                if (++leading == 16) {
                    leading = 1;
                    power += 4;
                }
            }
        }
        fraction <<= drop;
    }
    std::string text = upper ? "0X" : "0x";
    text += symbols[leading];
    if (count > 0 || point || precision > 0) text += '.';
    for (int i = 0; i < count; ++i) text += symbols[(fraction >> (56 - 4 * i)) & 0xf];
    if (precision > 15) text.append(static_cast<std::size_t>(precision) - 15, '0');
    char suffix[24];
    std::snprintf(suffix, sizeof(suffix), "%c%+lld", upper ? 'P' : 'p', power);
    return text + suffix;
}

inline std::string FormatX87(X87Extended value, const std::string& spec, char conversion) {
    bool left = false;
    bool plus = false;
    bool space = false;
    bool alternate = false;
    bool zero = false;
    std::size_t position = 1;
    for (; position < spec.size() && std::string("-+ #0").find(spec[position]) != std::string::npos; ++position) {
        left |= spec[position] == '-';
        plus |= spec[position] == '+';
        space |= spec[position] == ' ';
        alternate |= spec[position] == '#';
        zero |= spec[position] == '0';
    }
    std::size_t width = 0;
    for (; position < spec.size() && spec[position] >= '0' && spec[position] <= '9'; ++position)
        width = width * 10 + static_cast<std::size_t>(spec[position] - '0');
    long long precision = -1;
    if (position < spec.size() && spec[position] == '.') {
        precision = 0;
        for (++position; position < spec.size() && spec[position] >= '0' && spec[position] <= '9'; ++position)
            precision = std::min(precision * 10 + (spec[position] - '0'), 1000000000LL);
    }

    const bool negative = (value.signExponent & 0x8000) != 0;
    const unsigned field = value.signExponent & 0x7fffu;
    const bool integerBit = (value.significand >> 63) != 0;
    if (field == 0x7fff) {
        if (!integerBit) NotImplemented_nid_no_patch("x87 pseudo-infinity or pseudo-NaN formatting");
        const double special = (value.significand << 1) == 0 ? HUGE_VAL : std::nan("");
        const std::string format = spec + conversion;
        const int size = std::snprintf(nullptr, 0, format.c_str(), negative ? -special : special);
        if (size < 0) throw std::runtime_error("Formatting conversion failed");
        std::string text(static_cast<std::size_t>(size) + 1, '\0');
        std::snprintf(text.data(), text.size(), format.c_str(), negative ? -special : special);
        text.pop_back();
        return text;
    }
    if (field != 0 && !integerBit) NotImplemented_nid_no_patch("x87 unnormal formatting");

    const bool upper = conversion >= 'A' && conversion <= 'Z';
    const char lower = static_cast<char>(conversion | 0x20);
    std::string prefix = negative ? "-" : plus ? "+" : space ? " " : "";
    std::string body;
    if (lower == 'a') {
        body = X87Hexadecimal(value.significand, field, precision, alternate, upper);
        prefix += body.substr(0, 2);
        body.erase(0, 2);
    } else {
        const long long power = static_cast<long long>(field == 0 ? 1 : field) - 16383 - 63;
        X87BigUnsigned exact(value.significand);
        long long exponent = 0;
        if (power >= 0) {
            exact.ShiftLeft(static_cast<std::size_t>(power));
        } else {
            exact.MultiplyPow5(-power);
            exponent = power;
        }
        const std::string digits = exact.Decimal();
        if (precision < 0) precision = 6;
        const char marker = upper ? 'E' : 'e';
        if (lower == 'f') {
            body = X87Fixed(digits, exponent, precision, alternate);
        } else if (lower == 'e') {
            body = X87Scientific(digits, exponent, precision, alternate, marker);
        } else {
            const long long significant = precision == 0 ? 1 : precision;
            long long power10 = 0;
            if (digits != "0") {
                power10 = static_cast<long long>(digits.size()) + exponent - 1;
                if (static_cast<long long>(X87RoundDigits(digits, significant).size()) > significant) ++power10;
            }
            body = significant > power10 && power10 >= -4
                ? X87Fixed(digits, exponent, significant - 1 - power10, alternate)
                : X87Scientific(digits, exponent, significant - 1, alternate, marker);
            if (!alternate) {
                const std::size_t mark = body.find(marker);
                std::string mantissa = body.substr(0, mark);
                const std::string suffix = mark == std::string::npos ? std::string() : body.substr(mark);
                if (mantissa.find('.') != std::string::npos) {
                    mantissa.erase(mantissa.find_last_not_of('0') + 1);
                    if (mantissa.back() == '.') mantissa.pop_back();
                }
                body = mantissa + suffix;
            }
        }
    }
    const std::size_t length = prefix.size() + body.size();
    if (width <= length) return prefix + body;
    const std::size_t pad = width - length;
    if (left) return prefix + body + std::string(pad, ' ');
    if (zero) return prefix + std::string(pad, '0') + body;
    return std::string(pad, ' ') + prefix + body;
}

}

#endif
