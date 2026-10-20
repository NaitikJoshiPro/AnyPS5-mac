// qsort_s (C11 Annex K argument order): sorts with the context passed to the comparator, and rejects
// runtime-constraint violations with EINVAL after calling the installed constraint handler (the
// default handler aborts, as Sony's does; upstream's qsort_s, see also GuestQsort.cpp).
#include "SceTypes.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

extern "C" int APS5_VABI qsort_s_nid_postfix(void* base, std::size_t count, std::size_t size,
    int (APS5_VABI *compare)(const void*, const void*, void*), void* context);
extern "C" void* APS5_VABI set_constraint_handler_s_nid_postfix(void* handler);

static void Require(bool value) { if (!value) std::abort(); }

struct Context {
    int direction;
    int calls = 0;
};

static int constraintCalls = 0;

static void APS5_VABI CountConstraint(const char*, void*, int error) {
    Require(error == 22);
    ++constraintCalls;
}

static int APS5_VABI Compare(const void* left, const void* right, void* opaque) {
    auto& context = *static_cast<Context*>(opaque);
    ++context.calls;
    const int a = *static_cast<const int*>(left);
    const int b = *static_cast<const int*>(right);
    return context.direction * ((a > b) - (a < b));
}

int main() {
    int values[] = {5, -3, 9, 0, 9, 1, -7, 4, 2, 8, 3};
    constexpr std::size_t count = sizeof(values) / sizeof(values[0]);
    Context ascending{1};
    Require(qsort_s_nid_postfix(values, count, sizeof(int), Compare, &ascending) == 0 && ascending.calls > 0);
    for (std::size_t i = 1; i < count; ++i) Require(values[i - 1] <= values[i]);
    Context descending{-1};
    Require(qsort_s_nid_postfix(values, count, sizeof(int), Compare, &descending) == 0);
    for (std::size_t i = 1; i < count; ++i) Require(values[i - 1] >= values[i]);

    Context unused{1};
    Require(qsort_s_nid_postfix(nullptr, 0, sizeof(int), nullptr, &unused) == 0);
    Require(qsort_s_nid_postfix(values, 1, sizeof(int), Compare, &unused) == 0 && unused.calls == 0);
    void* const previous = set_constraint_handler_s_nid_postfix(reinterpret_cast<void*>(CountConstraint));
    Require(qsort_s_nid_postfix(nullptr, 2, sizeof(int), Compare, &unused) == 22);
    Require(qsort_s_nid_postfix(values, 2, sizeof(int), nullptr, &unused) == 22);
    Require(qsort_s_nid_postfix(values, std::numeric_limits<std::size_t>::max(), sizeof(int), Compare, &unused) == 22);
    Require(qsort_s_nid_postfix(values, 2, std::numeric_limits<std::size_t>::max(), Compare, &unused) == 22);
    Require(set_constraint_handler_s_nid_postfix(previous) == reinterpret_cast<void*>(CountConstraint));
    Require(constraintCalls == 4 && unused.calls == 0);
}
