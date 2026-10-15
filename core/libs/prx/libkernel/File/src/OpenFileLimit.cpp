#if defined(__linux__) || defined(__APPLE__)

#include <cerrno>
#include <system_error>
#include <sys/resource.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace {

struct OpenFileLimit {
    OpenFileLimit() {
        rlimit limit{};
        if (getrlimit(RLIMIT_NOFILE, &limit) != 0) throw std::system_error(errno, std::generic_category(), "read open file limit");
#if defined(__APPLE__)
        // macOS reports an unlimited hard limit but refuses a soft limit above kern.maxfilesperproc, so the real
        // ceiling is that sysctl. Pin soft and hard to it so the guest sees one limit, as on Linux.
        int perProcess = 0;
        size_t size = sizeof(perProcess);
        if (sysctlbyname("kern.maxfilesperproc", &perProcess, &size, nullptr, 0) != 0)
            throw std::system_error(errno, std::generic_category(), "read kern.maxfilesperproc");
        rlim_t ceiling = limit.rlim_max;
        if (perProcess > 0 && static_cast<rlim_t>(perProcess) < ceiling) ceiling = static_cast<rlim_t>(perProcess);
        if (limit.rlim_cur == ceiling && limit.rlim_max == ceiling) return;
        limit.rlim_cur = ceiling;
        limit.rlim_max = ceiling;
#else
        if (limit.rlim_cur >= limit.rlim_max) return;
        limit.rlim_cur = limit.rlim_max;
#endif
        if (setrlimit(RLIMIT_NOFILE, &limit) != 0) throw std::system_error(errno, std::generic_category(), "raise open file limit");
    }
} openFileLimit;

}

#endif
