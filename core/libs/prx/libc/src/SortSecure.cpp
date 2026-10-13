// qsort_s from C11 Annex K (bounds-checking interfaces), which the console's Dinkumware libc exports
// (NID G7yOZJObV+4). The argument order is Annex K's: the comparator takes the context as its third
// argument and the context comes last, unlike Microsoft's qsort_s (context first). That order is
// inferred from Annex K, not checked against a console caller (TechnicalDebt). Runtime-constraint
// violations (sizes above RSIZE_MAX, a null base or comparator with elements to sort) return EINVAL
// without calling a constraint handler.
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/ProgramCall.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

constexpr int GUEST_EINVAL = 22;
constexpr std::size_t RSIZE_MAX_GUEST = std::numeric_limits<std::size_t>::max() >> 1;

}

extern "C" int APS5_VABI qsort_s_nid_postfix(void* base, std::size_t count, std::size_t size,
    int (APS5_VABI *compare)(const void*, const void*, void*), void* context) {
    if (count > RSIZE_MAX_GUEST || size > RSIZE_MAX_GUEST) return GUEST_EINVAL;
    if (count != 0 && (base == nullptr || compare == nullptr)) return GUEST_EINVAL;
    if (count < 2 || size == 0) return 0;
    if (count > RSIZE_MAX_GUEST / size) return GUEST_EINVAL;

    auto* bytes = static_cast<unsigned char*>(base);
    const auto swapElements = [bytes, size](std::size_t left, std::size_t right) {
        auto* leftElement = bytes + left * size;
        auto* rightElement = bytes + right * size;
        for (std::size_t index = 0; index < size; ++index) std::swap(leftElement[index], rightElement[index]);
    };
    const auto less = [bytes, size, compare, context](std::size_t left, std::size_t right) {
        return CallProgram(compare, static_cast<const void*>(bytes + left * size), static_cast<const void*>(bytes + right * size), context) < 0;
    };
    const auto siftDown = [&](std::size_t root, std::size_t heapSize) {
        while (root < heapSize / 2) {
            std::size_t child = root * 2 + 1;
            if (child + 1 < heapSize && less(child, child + 1)) ++child;
            if (!less(root, child)) return;
            swapElements(root, child);
            root = child;
        }
    };
    for (std::size_t parent = count / 2; parent != 0; --parent) siftDown(parent - 1, count);
    for (std::size_t heapSize = count; heapSize > 1;) {
        swapElements(0, --heapSize);
        siftDown(0, heapSize);
    }
    return 0;
}
