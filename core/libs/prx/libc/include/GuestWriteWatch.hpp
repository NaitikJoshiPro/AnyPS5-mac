#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTWRITEWATCH_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTWRITEWATCH_HPP

#include <cstddef>
#include <cstdint>

namespace GuestWriteWatch {

extern "C" {

bool GuestWriteWatchAvailable_nid_postfix();
void GuestWriteWatchRegister_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestWriteWatchUnregister_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestWriteWatchCovers_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestWriteWatchOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestWriteWatchCollect_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);
void GuestWriteWatchBeginHostWrite_nid_postfix(const void* pointer, std::size_t bytes);
void GuestWriteWatchEndHostWrite_nid_postfix(const void* pointer, std::size_t bytes);
void GuestWriteWatchProtectionChanged_nid_postfix(const void* pointer, std::size_t bytes, int protection);
int GuestWriteWatchProtection_nid_postfix(std::uintptr_t address, std::uintptr_t limit, std::uintptr_t* runEnd);
bool GuestWriteWatchCollectArmed_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);
bool GuestWriteWatchCollectFresh_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);
// 4 KiB store precision where host pages are larger (macOS: 16 KiB). While a range is tracked (the
// calls nest per host page), a collection reports, for each host page holding an edge of the range,
// only the 4 KiB parts the CPU stored to: the parts of the first store after the page was armed (the
// fault address, so a same-value store counts) and the parts whose bytes differ from the page as that
// store found it (copied in the fault handler before the page is opened). A host write reports the
// parts it opens. A later same-value store to another part of an opened page is not seen. Without a
// copy (a page written before it was tracked, registered again, a protection change, a host write
// still open) the whole page is reported. No-op where host pages are 4 KiB.
void GuestWriteWatchTrackSubpages_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestWriteWatchUntrackSubpages_nid_postfix(std::uintptr_t address, std::size_t bytes);

}

}

#endif
