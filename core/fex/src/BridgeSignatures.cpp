#include "BridgeSignatures.hpp"

#include <algorithm>
#include <array>
#include <functional>

namespace Aps5Fex {

namespace {

#if defined(APS5_FEX_BRIDGE_SIGNATURES)
#include "BridgeSignatures.inc"
#else
constexpr std::array<BridgeLeaf, 0> Leaves {};
constexpr std::array<BridgeValue, 0> Values {};
constexpr std::array<BridgeSignature, 0> Callbacks {};
constexpr std::array<BridgeSignature, 0> Signatures {};
#endif

static_assert(std::ranges::adjacent_find(Signatures, std::ranges::greater_equal {}, &BridgeSignature::name) == Signatures.end());

constexpr bool InRange(std::span<const BridgeSignature> table) {
    return std::ranges::all_of(table, [](const BridgeSignature& signature) {
        return signature.firstValue + 1u + signature.parameterCount <= Values.size();
    });
}

static_assert(InRange(Signatures) && InRange(Callbacks));
static_assert(std::ranges::all_of(Values, [](const BridgeValue& value) {
    return value.firstLeaf + value.leafCount <= Leaves.size() && (value.callback == NoBridgeCallback || value.callback < Callbacks.size());
}));

}

bool HasBridgeSignatures() {
    return !Signatures.empty();
}

std::span<const BridgeSignature> BridgeSignatures() {
    return Signatures;
}

const BridgeSignature* FindBridgeSignature(std::string_view name) {
    const auto found = std::ranges::lower_bound(Signatures, name, std::ranges::less {}, &BridgeSignature::name);
    return found != Signatures.end() && found->name == name ? &*found : nullptr;
}

const BridgeValue& BridgeResult(const BridgeSignature& signature) {
    return std::span<const BridgeValue>(Values)[signature.firstValue];
}

std::span<const BridgeValue> BridgeParameters(const BridgeSignature& signature) {
    return std::span<const BridgeValue>(Values).subspan(signature.firstValue + 1u, signature.parameterCount);
}

std::span<const BridgeLeaf> BridgeLeaves(const BridgeValue& value) {
    return std::span<const BridgeLeaf>(Leaves).subspan(value.firstLeaf, value.leafCount);
}

const BridgeSignature* BridgeCallback(const BridgeValue& value) {
    return value.callback == NoBridgeCallback ? nullptr : &std::span<const BridgeSignature>(Callbacks)[value.callback];
}

}
