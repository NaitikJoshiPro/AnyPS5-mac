#ifndef CORE_FEX_SRC_BRIDGESIGNATURES_HPP
#define CORE_FEX_SRC_BRIDGESIGNATURES_HPP

#include <cstdint>
#include <span>
#include <string_view>

namespace Aps5Fex {

enum class BridgeKind : std::uint8_t {
    Void,
    Signed,
    Unsigned,
    Int128,
    Pointer,
    Function,
    Float,
    Double,
    LongDouble,
    X87,
    Vector,
    Record,
    Empty,
    Unsupported,
};

enum class BridgeLeafKind : std::uint8_t {
    Integer,
    Float,
    Double,
    LongDouble,
    X87,
    Vector,
};

namespace BridgeReason {

inline constexpr std::uint16_t Variadic = 1u << 0;
inline constexpr std::uint16_t X87Result = 1u << 1;
inline constexpr std::uint16_t LongDouble = 1u << 2;
inline constexpr std::uint16_t X87Argument = 1u << 3;
inline constexpr std::uint16_t StackLayout = 1u << 4;
inline constexpr std::uint16_t RegisterShift = 1u << 5;
inline constexpr std::uint16_t StructRegisters = 1u << 6;
inline constexpr std::uint16_t StructMemory = 1u << 7;
inline constexpr std::uint16_t HiddenResult = 1u << 8;
inline constexpr std::uint16_t StructResult = 1u << 9;
inline constexpr std::uint16_t Int128 = 1u << 10;
inline constexpr std::uint16_t Unclassified = 1u << 11;

}

inline constexpr std::uint32_t NoBridgeCallback = 0xffffffffu;

struct BridgeLeaf {
    std::uint16_t offset;
    std::uint8_t size;
    std::uint8_t align;
    BridgeLeafKind kind;
};

struct BridgeValue {
    BridgeKind kind;
    bool indirect;
    std::uint16_t align;
    std::uint32_t size;
    std::uint32_t firstLeaf;
    std::uint32_t leafCount;
    std::uint32_t callback;
};

struct BridgeSignature {
    std::string_view name;
    std::uint32_t firstValue;
    std::uint8_t parameterCount;
    bool variadic;
    std::uint16_t reasons;
};

bool HasBridgeSignatures();
std::span<const BridgeSignature> BridgeSignatures();
const BridgeSignature* FindBridgeSignature(std::string_view name);
const BridgeValue& BridgeResult(const BridgeSignature& signature);
std::span<const BridgeValue> BridgeParameters(const BridgeSignature& signature);
std::span<const BridgeLeaf> BridgeLeaves(const BridgeValue& value);
const BridgeSignature* BridgeCallback(const BridgeValue& value);

}

#endif
