#pragma once

// FEXCore's per-thread translation statistics (FEXCore::SHMStats::ThreadStats) for aps5-fex: time spent compiling
// guest blocks, blocks compiled, lookup-cache misses (a dispatcher lookup that found no linked block), the lookup
// cache's lock time and disk-cache hits. APS5_FEX_STATS=<seconds> turns them on and prints one `[fex-stats]`
// line per window and at exit (tools/mac/profile_title.py reads them). Off, FEXCore's counters stay null and
// cost nothing.

#include <cstdio>

namespace FEXCore::Core {
struct InternalThreadState;
}

namespace Aps5Fex {

class GuestStats {
public:
    static bool Enabled();
    static void Start();
    static void Register(FEXCore::Core::InternalThreadState* thread);
    static void Unregister(FEXCore::Core::InternalThreadState* thread);
    static void Report(std::FILE* out);
};

}
