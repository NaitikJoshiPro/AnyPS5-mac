#ifndef CORE_LIBS_PRX_LIBSCEAGC_SHADER_INCLUDE_SHADERUTILS_HPP
#define CORE_LIBS_PRX_LIBSCEAGC_SHADER_INCLUDE_SHADERUTILS_HPP

#include <cstdint>
#include <cstring>
#include "SceShaders.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"

// A shader header stores its tables as offsets relative to each field. `field` may be read through the host view of
// a displaced header (D31, D33), so the offset is applied to the field's guest address: the resolved pointer is the
// guest's, whichever view resolved it.
template <typename T>
void ResolveRelativePtr(T*& field) {
    if (field == nullptr) {
        return;
    }
    field = reinterpret_cast<T*>(
        reinterpret_cast<std::uintptr_t>(field) + GuestArena::GuestAddress(reinterpret_cast<std::uintptr_t>(&field))
    );
}

// The host view of a guest shader header, for native code that reads it. Titles may keep headers, code and tables
// in displaced guest memory (D31, D33: ASTRO BOT's are at 0x5xxxxxxxx in the shared region tail), where every native
// load through the guest address takes the fault path (two signals). HostShader copies the fixed fields once from
// the header's host view and points each table (code, registers, specials, semantics, user data) at its host view,
// so the reads after it never fault. ShaderUserData's own tables stay guest pointers: translate them with
// GuestArena::HostPointer where they are read. The view is for reading only: anything handed to the driver
// (registration keys on the guest header and code addresses) or stored back into guest memory uses the guest
// pointers. Costs a compare per pointer for memory outside the window.
inline Shader HostShader(const Shader* guest) {
    Shader view;
    std::memcpy(&view, GuestArena::HostRange(guest, sizeof(Shader)), sizeof(Shader));
    view.user_data = GuestArena::HostPointer(view.user_data);
    view.code = GuestArena::HostPointer(view.code);
    view.cx_registers = GuestArena::HostPointer(view.cx_registers);
    view.sh_registers = GuestArena::HostPointer(view.sh_registers);
    view.specials = GuestArena::HostPointer(view.specials);
    view.input_semantics = GuestArena::HostPointer(view.input_semantics);
    view.output_semantics = GuestArena::HostPointer(view.output_semantics);
    return view;
}

bool GetProgramAddressRegisterOffset(std::uint8_t type, std::uint32_t& loOffset);
int PatchProgramAddressRegister(ShaderRegister* regs, std::uint32_t numRegs, std::uint8_t type, std::uint64_t base);
std::uint32_t GraphicsPrimTypeToGsOut(std::uint32_t primType);
std::uint32_t ShaderSemanticWord(const ShaderSemantic& s);
std::uint32_t ApplyInterpolantDefaultValue(std::uint32_t value, std::uint32_t psWord);
std::uint32_t ApplyInterpolantDefaultValueHi(std::uint32_t value, std::uint32_t psWord);
std::uint32_t CreateInterpolantF16Value(std::uint32_t psWord, const ShaderSemantic* gsSemantic);
std::uint32_t CreateInterpolantNonF16Value(std::uint32_t psWord, const ShaderSemantic* gsSemantic);
std::uint32_t CreateInterpolantMappingValue(std::uint32_t value, std::uint32_t psWord, std::uint32_t gsWord);
std::uint32_t CreateInterpolantDefaultValue(std::uint32_t value, std::uint32_t psWord);
const ShaderSemantic* FindOutputSemantic(const Shader* gs, std::uint32_t semantic);
void SetInterpolantRegister(ShaderRegister* regs, std::uint32_t index, std::uint32_t value);
void FillIdentityInterpolants(ShaderRegister* regs, std::uint32_t firstIndex);

#endif
