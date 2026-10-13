// macOS threading model of a guest run through its entry point (GuestCpu::Run, no StartFunction): the guest's
// main thread is not the process main thread (bit 0), the process main thread serves Aps5RunOnMainThread
// (bit 1), and AppKit creates a window there (bit 2). Before the fix the guest kept the process main thread, the
// main queue was never served, and a windowed sceVideoOutOpen failed with "NSWindow should only be instantiated
// on the main thread!" (DOOM PPSA99666, 2026-10-13).
#include "abi.h"

static void Run() {
    const int probe = Aps5AbiMainThreadProbe();
    Check((probe & 1) != 0, 0);
    Check((probe & 2) != 0, 1);
    Check((probe & 4) != 0, 2);
}

ABI_MAIN(Run())
