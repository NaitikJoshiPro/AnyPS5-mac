#ifndef CORE_FEX_SRC_GUESTCALL_HPP
#define CORE_FEX_SRC_GUESTCALL_HPP

#include <cstdint>

namespace Aps5Fex {

struct BridgeSignature;

// A native function pointer that calls the guest function at guest the way callback describes (null:
// positionally): an entry of the static thunk table in Trampolines.S. The same guest function and shape
// always get the same thunk. Returns guest when the table is full; native code calling a guest address
// still works then, through the fault path (GuestCpu.cpp), with positional arguments.
std::uint64_t GuestThunkFor(std::uint64_t guest, const BridgeSignature* callback);

// The guest function a thunk calls, or 0 when address is not a thunk.
std::uint64_t GuestBehindThunk(std::uint64_t address);

}

#endif
