#include "bench.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <pthread/qos.h>
#include <utility>

namespace {

qos_class_t requestedQos = QOS_CLASS_UNSPECIFIED;

bool ReadQos() {
    const char* name = std::getenv("APS5_FEX_QOS");
    if (name == nullptr || *name == 0) return true;
    constexpr std::pair<const char*, qos_class_t> Classes[] = {
        {"user-interactive", QOS_CLASS_USER_INTERACTIVE}, {"user-initiated", QOS_CLASS_USER_INITIATED},
        {"default", QOS_CLASS_DEFAULT}, {"utility", QOS_CLASS_UTILITY}, {"background", QOS_CLASS_BACKGROUND}};
    for (const auto& [text, qos] : Classes) {
        if (std::strcmp(name, text) == 0) {
            requestedQos = qos;
            return pthread_set_qos_class_self_np(qos, 0) == 0;
        }
    }
    return false;
}

}

extern "C" {

u64 BenchNow() {
    timespec now {};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<u64>(now.tv_sec) * 1000000000ULL + static_cast<u64>(now.tv_nsec);
}

u64 BenchCpuNow() {
    timespec now {};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
    return static_cast<u64>(now.tv_sec) * 1000000000ULL + static_cast<u64>(now.tv_nsec);
}

void BenchWrite(const char* line) {
    std::puts(line);
}

void* BenchCopy(void* destination, const void* source, unsigned long size) {
    return std::memcpy(destination, source, size);
}

void* BenchFill(void* destination, int value, unsigned long size) {
    return std::memset(destination, value, size);
}

void BenchSort(void* base, unsigned long count, unsigned long size, int (*compare)(const void*, const void*)) {
    if (count < 2) return;
    auto* bytes = static_cast<unsigned char*>(base);
    const auto swapElements = [bytes, size](std::size_t left, std::size_t right) {
        auto* leftElement = bytes + left * size;
        auto* rightElement = bytes + right * size;
        for (std::size_t index = 0; index < size; ++index) std::swap(leftElement[index], rightElement[index]);
    };
    const auto siftDown = [bytes, size, compare, &swapElements](std::size_t root, std::size_t heapSize) {
        while (root < heapSize / 2) {
            std::size_t child = root * 2 + 1;
            if (child + 1 < heapSize && compare(bytes + child * size, bytes + (child + 1) * size) < 0) ++child;
            if (compare(bytes + root * size, bytes + child * size) >= 0) return;
            swapElements(root, child);
            root = child;
        }
    };
    for (std::size_t parent = count / 2; parent != 0; --parent) siftDown(parent - 1, count);
    for (std::size_t heapSize = count; heapSize > 1;) {
        swapElements(0, --heapSize);
        siftDown(0, heapSize);
    }
}

__attribute__((noinline)) u64 BenchTrivial() {
    asm volatile("");
    return 1000000000ULL;
}

bool BenchSpawn(void** thread, void* (*entry)(void*), void* argument) {
    pthread_attr_t attributes;
    if (pthread_attr_init(&attributes) != 0) return false;
    if (requestedQos != QOS_CLASS_UNSPECIFIED) pthread_attr_set_qos_class_np(&attributes, requestedQos, 0);
    pthread_t handle {};
    const bool created = pthread_create(&handle, &attributes, entry, argument) == 0;
    pthread_attr_destroy(&attributes);
    if (created) *thread = handle;
    return created;
}

void BenchJoin(void* thread) {
    pthread_join(static_cast<pthread_t>(thread), nullptr);
}

void BenchExit(int status) {
    std::fflush(stdout);
    std::exit(status);
}

}

int main(int count, char** arguments) {
    if (!ReadQos()) {
        std::fprintf(stderr, "bench: APS5_FEX_QOS must be user-interactive, user-initiated, default, utility or background\n");
        return 2;
    }
    BenchExit(BenchMain(count, arguments));
}
