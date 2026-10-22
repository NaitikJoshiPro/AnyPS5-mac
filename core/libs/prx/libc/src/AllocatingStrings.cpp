#include "prx/libc/include/ApplicationHeap.hpp"
#include "prx/libc/include/General.hpp"
#include "SceTypes.hpp"
#include "prx/libc/include/VarArgsAbi.hpp"
#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <new>

extern "C" {

int APS5_VABI vasprintf_nid_postfix(char** destination, const char* format, VaList* args);

char* APS5_VABI strndup_nid_postfix(const char* source, std::size_t limit) {
    if (!source) { errno = 22; return nullptr; }
    std::size_t length = 0;
    while (length < limit && source[length] != '\0') ++length;
    try {
        auto* result = static_cast<char*>(ApplicationHeapAllocate_nid_no_patch(length + 1));
        if (!result) { errno = 12; return nullptr; }
        std::memcpy(result, source, length);
        result[length] = '\0';
        return result;
    } catch (const std::bad_alloc&) {
        errno = 12;
        return nullptr;
    }
}

char* APS5_VABI strdup_nid_postfix(const char* source) {
    return strndup_nid_postfix(source, static_cast<std::size_t>(-1));
}

int APS5_VABI asprintf_nid_postfix(char** destination, const char* format, ...) {
    APS5_VA_BEGIN(format);
    const int result = vasprintf_nid_postfix(destination, format, reinterpret_cast<VaList*>(args));
    APS5_VA_END();
    return result;
}

}
