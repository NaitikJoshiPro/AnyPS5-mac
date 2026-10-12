#ifndef CORE_FEX_SRC_OPTIONS_HPP
#define CORE_FEX_SRC_OPTIONS_HPP

namespace Aps5Fex {

void LoadOptions();
void ApplyThreadQos();
// APS5_FEX_CLEAR_CODE=1: drop the code cache each time a thread enters guest code from the host at depth 0
// (a stress mode for code-buffer reclamation).
bool ClearCodeOnGuestEntry();

}

#endif
