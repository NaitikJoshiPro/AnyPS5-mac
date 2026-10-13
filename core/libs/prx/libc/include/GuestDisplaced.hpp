#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTDISPLACED_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTDISPLACED_HPP

#include <cstddef>
#include <cstdint>

// Guest addresses the host cannot map at their own value (macOS on arm64: the kernel's permanent "GPU Carveout"
// guard, 64-448 GiB, which xnu's vm_map_exec puts into every 64-bit process) live at a fixed alias instead:
// host = guest | DisplacedFlip. Guest JIT code and native code that touch the guest address fault on the guard; a
// fault handler runs the faulting load/store against the alias and backpatches JIT sites (GuestDisplaced.cpp).
// Native code that maps, protects, wraps or bulk-reads guest memory translates with HostAddress().
namespace GuestArena {

#if defined(__APPLE__) && defined(__aarch64__)
inline constexpr std::uintptr_t DisplacedStart = 0x0000001000000000ull; // MACH_VM_MIN_GPU_CARVEOUT_ADDRESS
inline constexpr std::uintptr_t DisplacedEnd = 0x0000007000000000ull;   // MACH_VM_MAX_GPU_CARVEOUT_ADDRESS
inline constexpr std::uintptr_t DisplacedFlip = 0x0000010000000000ull;  // bit 40: one EOR immediate
#else
inline constexpr std::uintptr_t DisplacedStart = 0;
inline constexpr std::uintptr_t DisplacedEnd = 0;
inline constexpr std::uintptr_t DisplacedFlip = 0;
#endif

inline constexpr bool IsDisplaced(std::uintptr_t guest) {
    return guest - DisplacedStart < DisplacedEnd - DisplacedStart;
}

// True when [guest, guest + bytes) touches the displaced window at all.
inline constexpr bool OverlapsDisplaced(std::uintptr_t guest, std::size_t bytes) {
    return bytes != 0 && guest < DisplacedEnd && guest + bytes > DisplacedStart;
}

inline constexpr std::uintptr_t HostAddress(std::uintptr_t guest) {
    return IsDisplaced(guest) ? guest | DisplacedFlip : guest;
}

inline constexpr std::uintptr_t GuestAddress(std::uintptr_t host) {
    return IsDisplaced(host & ~DisplacedFlip) && (host & DisplacedFlip) != 0 ? host & ~DisplacedFlip : host;
}

template <typename T>
inline T* HostPointer(T* guest) {
    return reinterpret_cast<T*>(HostAddress(reinterpret_cast<std::uintptr_t>(guest)));
}

extern "C" {
// Reserves the alias window and installs the fault handler (once). False when the host has no displaced window
// or the alias could not be reserved; fixed guest mappings in the window then fail as host regions.
bool GuestDisplacedEnable_nid_no_patch();
bool GuestDisplacedActive_nid_no_patch();
// Set by the CPU backend. writer replaces one instruction word of JIT code at its executable address, so displaced
// accesses in translated code are backpatched to a stub instead of faulting every time; allocator returns executable
// memory within branch range of a JIT pc (and its writable view), or 0 when pc is not JIT code.
void GuestDisplacedSetCodeWriter_nid_no_patch(bool (*writer)(std::uintptr_t pc, std::uint32_t instruction),
    std::uintptr_t (*allocator)(std::uintptr_t pc, std::size_t bytes, std::uintptr_t* writable));
// Faults redirected to the alias, and the JIT sites patched so far.
void GuestDisplacedStats_nid_no_patch(std::uint64_t* faults, std::uint64_t* patches);
}

}

#endif
