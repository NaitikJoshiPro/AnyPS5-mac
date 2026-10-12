#include "BridgeSignatures.hpp"

#include <cstdio>
#include <cstdlib>
#include <source_location>
#include <string_view>

namespace {

using namespace Aps5Fex;

void Require(bool value, std::source_location where = std::source_location::current()) {
    if (!value) {
        std::fprintf(stderr, "bridge signature check failed at line %u\n", static_cast<unsigned>(where.line()));
        std::abort();
    }
}

const BridgeSignature& Find(std::string_view name) {
    const BridgeSignature* signature = FindBridgeSignature(name);
    if (signature == nullptr) {
        std::fprintf(stderr, "%.*s has no bridge signature\n", static_cast<int>(name.size()), name.data());
        std::abort();
    }
    Require(signature->name == name);
    return *signature;
}

bool Has(const BridgeSignature& signature, std::uint16_t reason) {
    return (signature.reasons & reason) != 0;
}

void CheckTable() {
    Require(HasBridgeSignatures());
    Require(BridgeSignatures().size() > 1000);
    Require(FindBridgeSignature("") == nullptr);
    Require(FindBridgeSignature("Aps5NoSuchExport") == nullptr);
    for (const BridgeSignature& signature : BridgeSignatures()) {
        Require(FindBridgeSignature(signature.name) == &signature);
        Require(!Has(signature, BridgeReason::Unclassified));
        Require(BridgeResult(signature).kind != BridgeKind::Unsupported);
        Require(Has(signature, BridgeReason::Variadic) == signature.variadic);
    }
}

void CheckPositional() {
    const BridgeSignature& strlen = Find("strlen_nid_postfix");
    Require(strlen.reasons == 0 && strlen.parameterCount == 1);
    Require(BridgeResult(strlen).kind == BridgeKind::Unsigned && BridgeResult(strlen).size == 8);
    Require(BridgeParameters(strlen)[0].kind == BridgeKind::Pointer);

    const BridgeSignature& lldiv = Find("lldiv_nid_postfix");
    Require(lldiv.reasons == 0);
    const BridgeValue& quotient = BridgeResult(lldiv);
    Require(quotient.kind == BridgeKind::Record && !quotient.indirect && quotient.size == 16);
    Require(BridgeLeaves(quotient).size() == 2 && BridgeLeaves(quotient)[1].offset == 8);
    Require(BridgeLeaves(quotient)[0].kind == BridgeLeafKind::Integer);
}

void CheckCallbacks() {
    const BridgeSignature& qsort = Find("qsort_nid_postfix");
    Require(qsort.reasons == 0 && qsort.parameterCount == 4);
    const BridgeValue& compare = BridgeParameters(qsort)[3];
    Require(compare.kind == BridgeKind::Function);
    const BridgeSignature* callback = BridgeCallback(compare);
    Require(callback != nullptr && callback->name.empty() && callback->parameterCount == 2 && callback->reasons == 0);
    Require(BridgeResult(*callback).kind == BridgeKind::Signed && BridgeResult(*callback).size == 4);
    Require(BridgeParameters(*callback)[0].kind == BridgeKind::Pointer);
    Require(BridgeCallback(BridgeParameters(qsort)[0]) == nullptr);
}

void CheckMarshalling() {
    const BridgeSignature& releaseMem = Find("sceAgcCbReleaseMem");
    Require(releaseMem.parameterCount == 12 && Has(releaseMem, BridgeReason::StackLayout));
    Require(BridgeParameters(releaseMem)[9].kind == BridgeKind::Unsigned && BridgeParameters(releaseMem)[9].size == 2);

    const BridgeSignature& coefficients = Find("sceAudioOut2GetSpeakerArrayCoefficients");
    Require(Has(coefficients, BridgeReason::StructRegisters));
    const BridgeValue& position = BridgeParameters(coefficients)[1];
    Require(position.kind == BridgeKind::Record && position.size == 12 && BridgeLeaves(position).size() == 3);
    for (const BridgeLeaf& leaf : BridgeLeaves(position)) Require(leaf.kind == BridgeLeafKind::Float && leaf.size == 4);

    const BridgeSignature& currentException = Find("_ZSt17current_exceptionv_nid_postfix");
    Require(Has(currentException, BridgeReason::HiddenResult) && BridgeResult(currentException).indirect);

    const BridgeSignature& printf = Find("printf_nid_postfix");
    Require(printf.variadic && printf.parameterCount == 1 && printf.reasons == BridgeReason::Variadic);

    const BridgeSignature& strtold = Find("strtold_nid_postfix");
    Require(strtold.reasons == BridgeReason::X87Result && BridgeResult(strtold).kind == BridgeKind::X87);

    const BridgeSignature& acosl = Find("acosl_nid_postfix");
    Require(Has(acosl, BridgeReason::LongDouble) && BridgeResult(acosl).kind == BridgeKind::LongDouble);
    Require(BridgeParameters(acosl)[0].kind == BridgeKind::LongDouble);

    const BridgeSignature& signbitl = Find("__signbitl_nid_postfix");
    Require(Has(signbitl, BridgeReason::X87Argument) && BridgeParameters(signbitl)[0].kind == BridgeKind::X87);
}

bool IsPiece(const BridgePiece& piece, BridgePlace place, unsigned reg, unsigned at, unsigned offset, unsigned size) {
    return piece.place == place && piece.reg == reg && piece.at == at && piece.offset == offset && piece.size == size && !piece.reference;
}

void CheckLayouts() {
    for (const BridgeSignature& signature : BridgeSignatures())
        for (std::size_t value = 0; value <= signature.parameterCount; ++value) {
            for (const BridgePiece& piece : BridgeGuestPieces(signature, value)) Require(piece.size != 0);
            for (const BridgePiece& piece : BridgeHostPieces(signature, value)) Require(piece.size != 0);
        }

    // long double: 16 bytes on the guest stack and st(0), a double in d0 on arm64.
    const BridgeSignature& acosl = Find("acosl_nid_postfix");
    Require(BridgeGuestPieces(acosl, 0).size() == 1 && BridgeGuestPieces(acosl, 0)[0].place == BridgePlace::X87);
    Require(IsPiece(BridgeGuestPieces(acosl, 1)[0], BridgePlace::Stack, 0, 0, 0, 16));
    Require(IsPiece(BridgeHostPieces(acosl, 1)[0], BridgePlace::Vector, 0, 0, 0, 8));
    Require(IsPiece(BridgeHostPieces(acosl, 0)[0], BridgePlace::Vector, 0, 0, 0, 8));

    // The ninth and tenth arguments of sceAgcCbReleaseMem: 8-byte guest slots, packed u16s on arm64.
    const BridgeSignature& releaseMem = Find("sceAgcCbReleaseMem");
    Require(IsPiece(BridgeGuestPieces(releaseMem, 9)[0], BridgePlace::Stack, 0, 16, 0, 2));
    Require(IsPiece(BridgeHostPieces(releaseMem, 9)[0], BridgePlace::Stack, 0, 0, 0, 2));
    Require(IsPiece(BridgeHostPieces(releaseMem, 10)[0], BridgePlace::Stack, 0, 2, 0, 2));
    Require(IsPiece(BridgeHostPieces(releaseMem, 12)[0], BridgePlace::Stack, 0, 8, 0, 4));

    // A float HFA: two SSE eightbytes on the guest, three s registers on arm64.
    const BridgeSignature& coefficients = Find("sceAudioOut2GetSpeakerArrayCoefficients");
    Require(BridgeGuestPieces(coefficients, 2).size() == 2 && BridgeHostPieces(coefficients, 2).size() == 3);
    Require(IsPiece(BridgeGuestPieces(coefficients, 2)[1], BridgePlace::Vector, 1, 0, 8, 4));
    Require(IsPiece(BridgeHostPieces(coefficients, 2)[2], BridgePlace::Vector, 2, 0, 8, 4));

    // Hidden results: rdi on the guest, x8 on arm64.
    const BridgeSignature& currentException = Find("_ZSt17current_exceptionv_nid_postfix");
    Require(BridgeGuestPieces(currentException, 0)[0].place == BridgePlace::Sret && BridgeGuestPieces(currentException, 0)[0].reference);
    Require(BridgeHostPieces(currentException, 0)[0].place == BridgePlace::Sret);

    // GuestLongDouble results: st(0) on the guest, x0:x1 on arm64.
    const BridgeSignature& strtold = Find("strtold_nid_postfix");
    Require(BridgeGuestPieces(strtold, 0)[0].place == BridgePlace::X87 && BridgeHostPieces(strtold, 0).size() == 2);
    Require(IsPiece(BridgeHostPieces(strtold, 0)[1], BridgePlace::Gpr, 1, 0, 8, 8));
}

}

int main() {
    CheckTable();
    CheckPositional();
    CheckCallbacks();
    CheckMarshalling();
    CheckLayouts();
    std::printf("%zu bridge signatures checked\n", BridgeSignatures().size());
    return 0;
}
