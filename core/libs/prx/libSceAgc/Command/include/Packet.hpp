#ifndef CORE_LIBS_PRX_LIBSCEAGC_COMMAND_INCLUDE_PACKET_HPP
#define CORE_LIBS_PRX_LIBSCEAGC_COMMAND_INCLUDE_PACKET_HPP

#include <cstdint>
#include <initializer_list>
#include "SceTypes.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"

namespace Agc::Command {

// The host view of guest command memory, command buffer structs and register arrays. Guest memory in the displaced
// window (D31, D33; ASTRO BOT keeps its command buffers there) is written through its alias, not through the fault
// handler (one or two signals per native access). An object that starts in the window lies wholly inside it (the
// arena never maps across its edges), so the start address decides. Guest pointers handed back stay guest pointers.
template <typename T>
T* Host(T* guest) {
    return GuestArena::HostRange(guest, sizeof(T));
}

void Require(bool condition, const char* function, const char* reason);
void CheckBits(std::uint64_t value, std::uint64_t mask, const char* function);
void CheckAddress(std::uint64_t address, std::uint32_t alignment, const char* function);
std::uint32_t Header(std::uint32_t opcode, std::uint32_t count, std::uint32_t flags = 0);
void Reserve(CommandBuffer* buffer, std::uint32_t count, const char* function);
std::uint32_t* Allocate(CommandBuffer* buffer, std::uint32_t count, const char* function);
std::uint32_t* Emit(CommandBuffer* buffer, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload, const char* function);
void ValidatePacket(const std::uint32_t* packet, std::uint32_t opcode, std::uint32_t count, const char* function);
std::uint32_t* WriteNop(CommandBuffer* buffer, std::uint32_t count, const char* function);
std::uint32_t* WriteRegisterRange(CommandBuffer* buffer, std::uint32_t opcode, std::uint32_t offset, const std::uint32_t* values, std::uint32_t count, const char* function);
std::uint32_t* WriteRegisters(CommandBuffer* buffer, std::uint32_t opcode, const volatile ShaderRegister* registers, std::uint32_t count, bool snapshotAll, const char* function);
std::uint32_t* WriteIndirectRegisters(CommandBuffer* buffer, std::uint32_t opcode, const volatile ShaderRegister* registers, std::uint32_t count, const char* function);
void PatchIndirectAddress(std::uint32_t* packet, std::uint32_t opcode, const volatile ShaderRegister* registers, const char* function);
void PatchIndirectCount(std::uint32_t* packet, std::uint32_t opcode, std::uint32_t count, const char* function);
void SetIndirectCount(std::uint32_t* packet, std::uint32_t opcode, std::uint32_t count, const char* function);

}

#endif
