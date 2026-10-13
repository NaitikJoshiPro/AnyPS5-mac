// link(2) between two guest paths. The console's libkernel exports it (SuperTuxKart imports it from
// libkernel); both paths go through the guest path layer (/app0, /download0, ...) like the other file calls.
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/FilesystemError.hpp"
#include <filesystem>
#include <system_error>

extern "C" int* APS5_VABI __error_nid_postfix();

namespace {

constexpr int GUEST_ENOENT = 2;
constexpr int GUEST_EFAULT = 14;

int Fail(int error) {
    *__error_nid_postfix() = error;
    return -1;
}

}

extern "C" int APS5_VABI link_nid_postfix(const char* existing, const char* created) {
    if (existing == nullptr || created == nullptr) return Fail(GUEST_EFAULT);
    if (*existing == '\0' || *created == '\0') return Fail(GUEST_ENOENT);
    std::filesystem::path source;
    if (const int error = ResolvePathError_nid_no_patch(existing, source)) return Fail(error);
    std::filesystem::path target;
    if (const int error = ResolvePathError_nid_no_patch(created, target)) return Fail(error);
    std::error_code error;
    std::filesystem::create_hard_link(source, target, error);
    if (error) return Fail(FilesystemError(error));
    RecordWrittenPath_nid_no_patch(target);
    return 0;
}
