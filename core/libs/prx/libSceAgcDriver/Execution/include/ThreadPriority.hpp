#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADPRIORITY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADPRIORITY_HPP

#include <cstdint>
#include <optional>
#include <string>

namespace AgcDriver {

enum class ThreadPriorityMode { Off, High, Realtime };

struct RttimeLimit {
    std::uint64_t soft = 0;
    std::uint64_t hard = 0;
    bool operator==(const RttimeLimit&) const = default;
};

struct RtkitLimits {
    std::int64_t maxPriority = 0;
    std::int64_t maxRttimeUs = 0;
};

struct RealtimePlan {
    std::string refusal;
    std::optional<RttimeLimit> limit;
};

// macOS quality of service of the driver's frame-path threads (queue workers, presenter): APS5_WORKER_QOS =
// user-interactive (the default), user-initiated, default, utility or off (leave the inherited class).
enum class HostThreadQos { Unchanged, UserInteractive, UserInitiated, Default, Utility };

ThreadPriorityMode ParseThreadPriorityMode(const char* text);
HostThreadQos ParseHostThreadQos(const char* text);
RealtimePlan PlanRealtime(std::uint32_t priority, const std::optional<RtkitLimits>& rtkit, RttimeLimit current, std::uint64_t requestedUs);
void RaiseWorkerThreadPriority(const char* role);

}

#endif
