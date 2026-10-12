#ifndef CORE_FEX_SRC_BRIDGESIGNATURES_HPP
#define CORE_FEX_SRC_BRIDGESIGNATURES_HPP

#include <cstddef>
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

// Where one part of a value lives: a general register (guest: rdi..r9 for arguments, rax/rdx for a result;
// host: x0..x7), a vector register and byte, a byte offset into the stack arguments, st(0), or the memory a
// hidden result pointer names (guest: rdi in, rax out; host: x8).
enum class BridgePlace : std::uint8_t {
    Gpr,
    Vector,
    Stack,
    X87,
    Sret,
};

struct BridgePiece {
    BridgePlace place;
    std::uint8_t reg;
    std::uint16_t at;
    std::uint16_t offset;
    std::uint16_t size;
    bool reference;
};

// Per value (the result, then the parameters): its pieces under SysV (guest) and Apple arm64 (host).
struct BridgeLayout {
    std::uint32_t guest;
    std::uint32_t guestCount;
    std::uint32_t host;
    std::uint32_t hostCount;
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
// value 0 is the result, value i + 1 parameter i.
std::span<const BridgePiece> BridgeGuestPieces(const BridgeSignature& signature, std::size_t value);
std::span<const BridgePiece> BridgeHostPieces(const BridgeSignature& signature, std::size_t value);

}

#endif
