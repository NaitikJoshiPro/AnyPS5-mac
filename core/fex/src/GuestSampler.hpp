#ifndef CORE_FEX_SRC_GUESTSAMPLER_HPP
#define CORE_FEX_SRC_GUESTSAMPLER_HPP

#include <FEXCore/Core/SignalDelegator.h>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace FEXCore::Context {
class Context;
}
namespace FEXCore::Core {
struct InternalThreadState;
}

namespace Aps5Fex {

class GuestModules;

// Wall-clock sampler of guest threads: maps the host PC of every thread that runs guest code back to a guest RIP
// (FEX's per-block RIP entries) and reports where each thread spends its time, with the guest registers of the
// most recent sample at each RIP. A spinning guest thread shows as one or two hot RIPs plus the address it polls.
//
//   APS5_FEX_SAMPLE=<ms>             sample every <ms> milliseconds (off when unset or 0; then nothing runs)
//   APS5_FEX_SAMPLE_REPORT=<s>       print a report every <s> seconds (default 10) and at exit, on stderr
//   APS5_FEX_SAMPLE_TOP=<n>          RIPs per thread in a report (default 8)
//
// Report lines start with "[fex-sample]". A guest RIP is printed absolute and as <module>+0x<offset>. A thread
// stopped in host code (an HLE call, a wait) is reported by the host symbol (dladdr) and the guest RIP that
// called out. Threads are suspended with thread_suspend for a few microseconds per sample; no allocation happens
// while a thread is suspended.
class GuestSampler {
public:
    // True when APS5_FEX_SAMPLE asks for sampling. Cheap; read once.
    static bool Enabled();

    // Starts the sampler thread (once). modules may be null.
    // importName names a RIP outside the modules (the bridge's import stubs); it may return "".
    static void Start(FEXCore::Context::Context* context, const GuestModules* modules,
                      const FEXCore::SignalDelegatorConfig::SRAIndexMapping& gprMapping,
                      std::function<std::string(std::uint64_t)> importName);
    // Detaches the context and modules (GuestCpu's destructor): sampling pauses until the next Start.
    static void Stop(FEXCore::Context::Context* context);

    // Called on the host thread that will run guest code through `thread`, and before that thread is destroyed.
    static void Register(FEXCore::Core::InternalThreadState* thread);
    static void Unregister(FEXCore::Core::InternalThreadState* thread);

    // Prints the report now (also used at exit).
    static void Report(std::FILE* out);

    // Formatting helpers, exposed for tests.
    struct Location {
        std::string module;
        std::uint64_t offset = 0;
    };
    static Location Locate(const GuestModules* modules, std::uint64_t rip);
};

}

#endif
