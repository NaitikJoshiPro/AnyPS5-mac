#include "GuestCall.hpp"

#include "BridgeSignatures.hpp"
#include "GuestCpu.hpp"
#include "X87Convert.hpp"

#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

extern "C" char Aps5GuestThunks[];
extern "C" char Aps5GuestThunksEnd[];

// What Aps5GuestThunkEntry saves: the arm64 argument registers, x8 and the caller's stack arguments.
struct Aps5GuestThunkFrame {
    std::uint64_t gpr[8];
    std::uint8_t vector[8][16];
    std::uint64_t x8;
    const std::uint8_t* stack;
};
static_assert(offsetof(Aps5GuestThunkFrame, vector) == 64 && offsetof(Aps5GuestThunkFrame, x8) == 192 &&
              offsetof(Aps5GuestThunkFrame, stack) == 200 && sizeof(Aps5GuestThunkFrame) == 208);

// GuestCpu.cpp's positional callback entry; Aps5GuestCallFrame is the prefix of Aps5GuestThunkFrame.
struct Aps5GuestCallFrame;
extern "C" void Aps5RunGuestCall(std::uint64_t target, Aps5GuestCallFrame* frame);

namespace Aps5Fex {

namespace {

constexpr std::size_t ThunkBytes = 8;
constexpr std::size_t ThunkCount = 4096;

struct Thunk {
    std::uint64_t guest;
    const BridgeSignature* callback;
    bool shaped;
};

std::array<Thunk, ThunkCount> thunks {};
std::atomic<std::size_t> thunkCount {0};
std::mutex thunkMutex;
std::map<std::pair<std::uint64_t, const BridgeSignature*>, std::size_t> thunkIndex;

bool Shaped(const BridgeSignature* signature) {
    return signature != nullptr && !signature->variadic && (signature->reasons & ~BridgeReason::Variadic) != 0;
}

constexpr std::uint32_t GuestArgumentRegisters[6] = {FEXCore::X86State::REG_RDI, FEXCore::X86State::REG_RSI, FEXCore::X86State::REG_RDX,
                                                     FEXCore::X86State::REG_RCX, FEXCore::X86State::REG_R8,  FEXCore::X86State::REG_R9};
constexpr std::uint32_t GuestResultRegisters[2] = {FEXCore::X86State::REG_RAX, FEXCore::X86State::REG_RDX};

// One shaped native->guest call: the guest stack area holds the stack arguments, then room for a result
// the guest returns through a hidden pointer while arm64 returns it in registers.
struct ShapedCallback {
    const BridgeSignature* signature;
    Aps5GuestThunkFrame* frame;
    std::uint8_t* arguments = nullptr;
    std::uint8_t* hiddenResult = nullptr;
};

std::uint8_t* HostPlace(const BridgePiece& piece, Aps5GuestThunkFrame& frame, bool result) {
    switch (piece.place) {
    case BridgePlace::Gpr: return reinterpret_cast<std::uint8_t*>(&frame.gpr[piece.reg]);
    case BridgePlace::Vector: return frame.vector[piece.reg] + piece.at;
    case BridgePlace::Stack:
        if (!result) return const_cast<std::uint8_t*>(frame.stack) + piece.at;
        break;
    default: break;
    }
    throw std::runtime_error("a host callback value in no place");
}

std::uint8_t* GuestPlace(const BridgePiece& piece, FEXCore::Core::CPUState& state, std::uint8_t* arguments, bool result) {
    switch (piece.place) {
    case BridgePlace::Gpr:
        return reinterpret_cast<std::uint8_t*>(&state.gregs[result ? GuestResultRegisters[piece.reg] : GuestArgumentRegisters[piece.reg]]);
    case BridgePlace::Vector: return reinterpret_cast<std::uint8_t*>(state.xmm.sse.data[piece.reg]) + piece.at;
    case BridgePlace::Stack: return arguments + piece.at;
    default: break;
    }
    throw std::runtime_error("a guest callback value in no place");
}

void Extend(const BridgeValue& value, const BridgePiece& piece, std::uint8_t* place) {
    if (value.kind != BridgeKind::Signed || piece.size >= 8 || piece.place != BridgePlace::Gpr) return;
    std::uint64_t word = 0;
    std::memcpy(&word, place, 8);
    const unsigned shift = 64 - 8 * piece.size;
    word = static_cast<std::uint64_t>(static_cast<std::int64_t>(word << shift) >> shift);
    std::memcpy(place, &word, 8);
}

std::size_t StackBytes(std::span<const BridgePiece> pieces) {
    std::size_t bytes = 0;
    for (const BridgePiece& piece : pieces)
        if (piece.place == BridgePlace::Stack) bytes = std::max<std::size_t>(bytes, piece.at + piece.size);
    return bytes;
}

void SetupCallback(FEXCore::Core::CPUState& state, void* context) {
    auto& call = *static_cast<ShapedCallback*>(context);
    const BridgeSignature& signature = *call.signature;
    Aps5GuestThunkFrame& frame = *call.frame;
    const auto parameters = BridgeParameters(signature);
    std::size_t argumentBytes = 0;
    for (std::size_t index = 0; index < parameters.size(); ++index)
        argumentBytes = std::max(argumentBytes, StackBytes(BridgeGuestPieces(signature, index + 1)));
    const auto guestResult = BridgeGuestPieces(signature, 0);
    const auto hostResult = BridgeHostPieces(signature, 0);
    const bool guestHidden = !guestResult.empty() && guestResult[0].place == BridgePlace::Sret;
    const bool hostHidden = !hostResult.empty() && hostResult[0].place == BridgePlace::Sret;
    const std::size_t resultBytes = guestHidden && !hostHidden ? BridgeResult(signature).size : 0;
    const std::size_t arguments16 = (argumentBytes + 15) & ~std::size_t {15};
    const std::size_t total = arguments16 + ((resultBytes + 15) & ~std::size_t {15});
    // Below the red zone of the guest code that called into native code. The callback entry pushes the
    // return address 16 bytes below rsp, so the guest finds its stack arguments at rsp - 8.
    const std::uint64_t area = (state.gregs[FEXCore::X86State::REG_RSP] - 256 - total) & ~std::uint64_t {15};
    call.arguments = reinterpret_cast<std::uint8_t*>(area);
    call.hiddenResult = resultBytes != 0 ? call.arguments + arguments16 : nullptr;
    state.gregs[FEXCore::X86State::REG_RSP] = area + 8;

    for (std::size_t index = 0; index < parameters.size(); ++index) {
        const BridgeValue& value = parameters[index];
        const auto guest = BridgeGuestPieces(signature, index + 1);
        const auto host = BridgeHostPieces(signature, index + 1);
        if (guest.empty()) continue;
        if (host[0].reference) {
            std::uint64_t pointer = 0;
            std::memcpy(&pointer, HostPlace(host[0], frame, false), 8);
            if (guest[0].reference)
                std::memcpy(GuestPlace(guest[0], state, call.arguments, false), &pointer, 8);
            else
                std::memcpy(GuestPlace(guest[0], state, call.arguments, false), reinterpret_cast<const void*>(pointer), value.size);
            continue;
        }
        alignas(16) std::uint8_t bytes[64] {};
        if (value.size > sizeof(bytes)) throw std::runtime_error("a callback argument is too large");
        for (const BridgePiece& piece : host) std::memcpy(bytes + piece.offset, HostPlace(piece, frame, false), piece.size);
        if (value.kind == BridgeKind::LongDouble) {
            double converted = 0;
            std::memcpy(&converted, bytes, sizeof(converted));
            DoubleToX87(converted, bytes);
        }
        for (const BridgePiece& piece : guest) {
            std::uint8_t* place = GuestPlace(piece, state, call.arguments, false);
            if (piece.place == BridgePlace::Gpr) std::memset(place, 0, 8);
            std::memcpy(place, bytes + piece.offset, piece.size);
            Extend(value, piece, place);
        }
    }
    if (guestHidden)
        state.gregs[FEXCore::X86State::REG_RDI] = hostHidden ? frame.x8 : reinterpret_cast<std::uint64_t>(call.hiddenResult);
}

void FinishCallback(FEXCore::Core::CPUState& state, void* context) {
    auto& call = *static_cast<ShapedCallback*>(context);
    const BridgeSignature& signature = *call.signature;
    Aps5GuestThunkFrame& frame = *call.frame;
    const auto guest = BridgeGuestPieces(signature, 0);
    const auto host = BridgeHostPieces(signature, 0);
    if (guest.empty()) return;
    const BridgeValue& value = BridgeResult(signature);
    const bool hostHidden = !host.empty() && host[0].place == BridgePlace::Sret;
    alignas(16) std::uint8_t bytes[64] {};
    if (guest[0].place == BridgePlace::Sret) {
        if (hostHidden) return;
        if (value.size > sizeof(bytes)) throw std::runtime_error("a callback result is too large");
        std::memcpy(bytes, call.hiddenResult, value.size);
    } else if (guest[0].place == BridgePlace::X87) {
        // The guest returned st(0): read it and pop the x87 stack, as the caller would.
        const unsigned top = state.flags[FEXCore::X86State::X87FLAG_TOP_LOC] & 7u;
        std::memcpy(bytes, state.mm[top], 10);
        state.AbridgedFTW &= static_cast<std::uint8_t>(~(1u << top));
        state.flags[FEXCore::X86State::X87FLAG_TOP_LOC] = static_cast<std::uint8_t>((top + 1) & 7u);
        if (value.kind == BridgeKind::LongDouble) {
            const double converted = X87ToDouble(bytes);
            std::memset(bytes, 0, sizeof(bytes));
            std::memcpy(bytes, &converted, sizeof(converted));
        }
    } else {
        for (const BridgePiece& piece : guest) std::memcpy(bytes + piece.offset, GuestPlace(piece, state, nullptr, true), piece.size);
    }
    if (hostHidden) {
        std::memcpy(reinterpret_cast<void*>(frame.x8), bytes, value.size);
        return;
    }
    for (const BridgePiece& piece : host) {
        std::uint8_t* place = HostPlace(piece, frame, true);
        if (piece.place == BridgePlace::Gpr) std::memset(place, 0, 8);
        std::memcpy(place, bytes + piece.offset, piece.size);
        Extend(value, piece, place);
    }
}

}

std::uint64_t GuestThunkFor(std::uint64_t guest, const BridgeSignature* callback) {
    std::lock_guard lock(thunkMutex);
    const auto key = std::pair {guest, callback};
    std::size_t index = 0;
    if (const auto found = thunkIndex.find(key); found != thunkIndex.end()) {
        index = found->second;
    } else {
        index = thunkCount.load(std::memory_order_relaxed);
        if (index == ThunkCount || reinterpret_cast<std::uintptr_t>(Aps5GuestThunks) + ThunkCount * ThunkBytes !=
                                       reinterpret_cast<std::uintptr_t>(Aps5GuestThunksEnd))
            return guest;
        thunks[index] = {guest, callback, Shaped(callback)};
        thunkCount.store(index + 1, std::memory_order_release);
        thunkIndex.emplace(key, index);
    }
    return reinterpret_cast<std::uint64_t>(Aps5GuestThunks) + index * ThunkBytes;
}

std::uint64_t GuestBehindThunk(std::uint64_t address) {
    const auto first = reinterpret_cast<std::uint64_t>(Aps5GuestThunks);
    if (address < first || address >= first + ThunkCount * ThunkBytes || (address - first) % ThunkBytes != 0) return 0;
    const std::size_t index = (address - first) / ThunkBytes;
    if (index >= thunkCount.load(std::memory_order_acquire)) return 0;
    return thunks[index].guest;
}

}

extern "C" void Aps5RunGuestThunk(std::uint32_t index, Aps5GuestThunkFrame* frame) {
    using namespace Aps5Fex;
    if (index >= thunkCount.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "[aps5-fex] native code called guest thunk %u, which is not in use\n", index);
        std::abort();
    }
    const Thunk& thunk = thunks[index];
    if (!thunk.shaped) {
        Aps5RunGuestCall(thunk.guest, reinterpret_cast<Aps5GuestCallFrame*>(frame));
        return;
    }
    ShapedCallback call {thunk.callback, frame};
    try {
        CallGuestWith(thunk.guest, SetupCallback, FinishCallback, &call);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[aps5-fex] a callback to guest function %#llx: %s\n", static_cast<unsigned long long>(thunk.guest), error.what());
        std::abort();
    }
}
