#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTDISPLACED_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTDISPLACED_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Guest addresses the host cannot map at their own value live at a fixed alias instead: host = guest | DisplacedFlip
// (D31, D33). On macOS arm64 these are a table of host-reserved ranges, found at startup (GuestDisplaced.cpp):
//   - the unused tail of the dyld shared region, from the end of the shared cache to 0xFC0000000 (its start moves
//     with the cache slide, so it is queried at runtime);
//   - the commpage nesting guard, [0xFC0000000, 0x1000000000);
//   - the "GPU Carveout" guard, [64 GiB, 448 GiB), which xnu's vm_map_exec puts into every 64-bit process.
// The ranges are contiguous, so they form one window [DisplacedStart(), DisplacedEnd) with one flip. Guest JIT code
// and native code that touch a guest address there fault (reads and writes fault in all three); a fault handler runs
// the faulting load/store against the alias and backpatches JIT sites. Native code that maps, protects, wraps or
// bulk-reads guest memory translates with HostAddress().
namespace GuestArena {

#if defined(__APPLE__) && defined(__aarch64__)
inline constexpr std::uintptr_t DisplacedLowest = 0x0000000180000000ull;  // SHARED_REGION_BASE_ARM64: no range starts lower
inline constexpr std::uintptr_t CarveoutStart = 0x0000001000000000ull;    // MACH_VM_MIN_GPU_CARVEOUT_ADDRESS
inline constexpr std::uintptr_t DisplacedEnd = 0x0000007000000000ull;     // MACH_VM_MAX_GPU_CARVEOUT_ADDRESS
inline constexpr std::uintptr_t DisplacedFlip = 0x0000010000000000ull;    // bit 40: one EOR immediate
#else
inline constexpr std::uintptr_t DisplacedLowest = 0;
inline constexpr std::uintptr_t CarveoutStart = 0;
inline constexpr std::uintptr_t DisplacedEnd = 0;
inline constexpr std::uintptr_t DisplacedFlip = 0;
#endif

struct DisplacedRange {
    std::uintptr_t guestStart;
    std::uintptr_t guestEnd;
    std::uintptr_t hostBase; // alias of guestStart; always guestStart | DisplacedFlip (the stubs flip one bit)
    const char* name;
};

extern "C" {
// Lowest displaced guest address: CarveoutStart until GuestDisplacedEnable_nid_no_patch() finds a lower range.
// Exported by libc (a function: the patched libraries export functions only); code that cannot link libc uses
// MayBeDisplaced().
std::uintptr_t GuestDisplacedStart_nid_no_patch();
}

inline std::uintptr_t DisplacedStart() {
    return GuestDisplacedStart_nid_no_patch();
}

// Coarse, link-free test: false for every address that can never be displaced. Titles whose memory lies above
// DisplacedEnd (the arena's first-fit placement) pay one compare.
inline constexpr bool MayBeDisplaced(std::uintptr_t guest) {
    return guest - DisplacedLowest < DisplacedEnd - DisplacedLowest;
}

inline bool IsDisplaced(std::uintptr_t guest) {
    return MayBeDisplaced(guest) && guest >= DisplacedStart();
}

// True when [guest, guest + bytes) touches the displaced window at all.
inline bool OverlapsDisplaced(std::uintptr_t guest, std::size_t bytes) {
    return bytes != 0 && guest < DisplacedEnd && guest + bytes > DisplacedStart();
}

inline std::uintptr_t HostAddress(std::uintptr_t guest) {
    return IsDisplaced(guest) ? guest | DisplacedFlip : guest;
}

inline std::uintptr_t GuestAddress(std::uintptr_t host) {
    return (host & DisplacedFlip) != 0 && IsDisplaced(host & ~DisplacedFlip) ? host & ~DisplacedFlip : host;
}

template <typename T>
inline T* HostPointer(T* guest) {
    return reinterpret_cast<T*>(HostAddress(reinterpret_cast<std::uintptr_t>(guest)));
}

// The host address of guest [guest, guest + bytes) for native bulk access (memcpy, memcmp, scans, host system calls),
// translated once up front instead of faulting per load/store; a host system call cannot be rescued by the fault
// handler at all (the kernel returns EFAULT). A range crossing the window's edge has no single backing and throws
// (the arena never maps one: the shared cache lies below the window, malloc's regions above it). Host addresses,
// aliases included, pass through unchanged. Costs one compare for memory above the window.
inline std::uintptr_t HostRange(std::uintptr_t guest, std::size_t bytes) {
    if (!MayBeDisplaced(guest)) return guest;
    const auto start = DisplacedStart();
    if (guest >= start) {
        if (bytes > DisplacedEnd - guest) throw std::runtime_error("native access to guest memory straddles the upper edge of the displaced window");
        return guest | DisplacedFlip;
    }
    if (bytes > start - guest) throw std::runtime_error("native access to guest memory straddles the lower edge of the displaced window");
    return guest;
}

template <typename T>
inline T* HostRange(T* guest, std::size_t bytes) {
    return reinterpret_cast<T*>(HostRange(reinterpret_cast<std::uintptr_t>(guest), bytes));
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
// Faults at sites that cannot be patched (native code, signed or out of branch range), and how many of their stubs
// returned through BRK (a second signal: a far stub whose return could not be made a branch), and the far stubs
// built with a branch return instead.
void GuestDisplacedNativeStats_nid_no_patch(std::uint64_t* unpatched, std::uint64_t* trapReturns, std::uint64_t* branchSites);
// Tests: native code in [start, end) gets far stubs and is treated as immutable like the dyld shared cache (0, 0 ends it).
void GuestDisplacedTestFarCode_nid_no_patch(std::uintptr_t start, std::uintptr_t end);
// The active displaced ranges in ascending order (empty when inactive).
void GuestDisplacedRanges_nid_no_patch(const DisplacedRange** ranges, std::size_t* count);
// Code in [start, end) is about to be freed (the CPU backend's code buffer, a module's code): forget every patched
// site in it and free their stubs (and the sites inside those stubs) for reuse. The caller guarantees that no thread
// runs the code any more (FEX frees a code buffer only once every thread has left it), so none runs their stubs
// either: a stub is entered only from its site and returns into its site's code. Safe against concurrent faults on
// other threads (one lock, signals blocked while it is held). Returns the number of sites forgotten.
std::size_t GuestDisplacedCodeFreed_nid_no_patch(std::uintptr_t start, std::uintptr_t end);
// Site table and stub memory occupancy: live sites, sites forgotten so far, stub slots in use, stub slots on free
// lists, stub chunks reserved, and sites whose pc lies inside a stub (a stub's own access that faulted).
struct DisplacedSiteStats {
    std::uint64_t liveSites = 0, forgottenSites = 0, slotsInUse = 0, slotsFree = 0, chunks = 0, stubSites = 0;
};
void GuestDisplacedSiteStats_nid_no_patch(DisplacedSiteStats* stats);
// Tests: cap the live sites and the stub chunks below their built-in limits (0 restores the default), so churn
// beyond the cap shows whether sites and stubs are reclaimed.
void GuestDisplacedTestLimits_nid_no_patch(std::size_t maxSites, std::size_t maxChunks);
}

}

#endif
