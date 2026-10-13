// libkernel's _sigaction (NID UDCI-WazohQ): the FreeBSD libc name of sigaction. SuperTuxKart's oops-sdk calls it
// while its renderer comes up. It forwards to sigaction (Signals.cpp) unchanged.
#include "prx/libc/include/general/VabiMacros.hpp"

extern "C" int APS5_VABI sigaction_nid_postfix(int guest, const void* action, void* previous);

extern "C" int APS5_VABI _sigaction_nid_postfix(int guest, const void* action, void* previous) {
    return sigaction_nid_postfix(guest, action, previous);
}
