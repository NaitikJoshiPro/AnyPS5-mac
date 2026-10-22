// Metal twin of tests/HostImportRetireHold.cpp (agc_host_import_retire, upstream #4291), swapped in by
// twins/T1-tests.cmake. Upstream drives the recorder and the host imports over MockVulkan: its fences say when a
// batch is complete and its vkDestroyBuffer says when a retired import is gone. Here both run on the Metal test
// device with the image cache attached (as the device makes its Context): the imports are no-copy MTLBuffers,
// the batches are real MTL4 command buffers, and
// - "destroyed" is the import's buffer handle reaching Metal::SetImportReleaseProbe (MetalNoCopy.h), which
//   GuestBufferMemory.mm's import destroy reports before it releases the buffer;
// - a batch upstream keeps in flight by leaving its mock fence unsignaled is held on the GPU queue: the test
//   queues a wait on a shared event of its own before the batch is committed (MTL4CommandQueue waitForEvent)
//   and signals it from the CPU when upstream signals the fence (a scope guard signals it on every exit, so no
//   batch is left waiting); a batch upstream completes by signaling its fence is waited for with
//   Recorder::WaitSerial, which does not reap, as upstream's mock signal does not.
// The checks and their messages are upstream's.
// Upstream 89bd3f19b6 adds RunHostImportFailureTests (tests/HostImportFailure.cpp): vkCreateBuffer, vkAllocateMemory
// and vkBindBufferMemory failures injected through the device resolver must publish no handle and release nothing
// undefined, and a retry must import. A Metal import has one creation step (newBufferWithBytesNoCopy over the
// guest pages, after GuestBufferMemory.mm's unwrappable() page check), so FailedImportPublishesNothing makes it
// fail the way it can on Metal: the registered range's pages are unreadable (PROT_NONE). The failure must leave no
// import, no live bytes and no release-probe call (no buffer was published), and once the pages are readable a
// retry must import a buffer over them (TechnicalDebt metal-host-import-failure-seams: no seam for the other steps).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalTestDevice.h"
#include "../src/MetalImages.h"
#include "../src/MetalNoCopy.h"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <sys/mman.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

using namespace AgcDriver::Graphics;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr std::size_t Alignment = 65536;
constexpr std::size_t RangeBytes = std::size_t{1} << 20u;

std::mutex destroyedMutex;
std::set<const void*> destroyed;

void NoteDestroyed(const void* buffer) {
    std::lock_guard lock(destroyedMutex);
    destroyed.insert(buffer);
}

// A handle can be reused by a later import once its buffer is gone: a scenario forgets it when it takes the import.
void Track(VkBuffer buffer) {
    std::lock_guard lock(destroyedMutex);
    destroyed.erase(buffer);
}

bool Live(VkBuffer buffer) {
    std::lock_guard lock(destroyedMutex);
    return buffer != VK_NULL_HANDLE && !destroyed.contains(buffer);
}

// The Metal device of the test, with the image cache attached as the device's setup attaches it.
class Device {
public:
    Device() : cache(test.device, test.queue) {
        auto& limits = test.context.limits;
        limits.maxStorageBufferRange = static_cast<std::uint32_t>(std::min<std::uint64_t>(test.device.maxBufferLength, 0xffffffffu));
        limits.maxUniformBufferRange = limits.maxStorageBufferRange;
        limits.minStorageBufferOffsetAlignment = 4u;
        limits.nonCoherentAtomSize = 1u;
        test.context.hostImportAlignment = Alignment;
        cache.Attach(test.context);
    }
    ~Device() {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        ClearHostImports(GetContext().device);
        cache.Teardown();
    }
    const Context& GetContext() const { return *cache.ImageContext(); }
    id<MTLDevice> MetalDevice() const { return test.device; }

private:
    MetalTest::Device test;
    mutable Metal::ImageCache cache;
};

// A GPU-queue gate: Hold() makes every batch committed after it wait on the GPU until Open().
class QueueGate {
public:
    explicit QueueGate(const Device& device) : queue(Metal::QueueOf(device.GetContext())), event([device.MetalDevice() newSharedEvent]) {
        if (queue == nil || event == nil) throw std::runtime_error("cannot make the queue gate");
    }
    ~QueueGate() { Open(); }
    QueueGate(const QueueGate&) = delete;
    QueueGate& operator=(const QueueGate&) = delete;
    void Hold() {
        ++value;
        [queue waitForEvent:event value:value];
    }
    void Open() {
        if (event.signaledValue < value) event.signaledValue = value;
    }

private:
    id<MTL4CommandQueue> queue;
    id<MTLSharedEvent> event;
    std::uint64_t value = 0;
};

struct Heap {
    std::byte* base = nullptr;
    bool first = false;
    bool second = false;

    Heap() {
        base = static_cast<std::byte*>(std::aligned_alloc(Alignment, 2 * RangeBytes));
        if (base == nullptr) throw std::runtime_error("cannot allocate the test heap");
        GuestAllocations::Mutation mutation;
        mutation.Add(base, RangeBytes, true, true, true);
        first = true;
    }

    void Grow() {
        GuestAllocations::Mutation mutation;
        mutation.Add(base + RangeBytes, RangeBytes, true, true, true);
        second = true;
    }

    std::uint64_t Address(std::size_t offset = 0) const {
        return reinterpret_cast<std::uint64_t>(base) + offset;
    }

    ~Heap() {
        try {
            GuestAllocations::Mutation mutation;
            if (second) mutation.Remove(base + RangeBytes);
            if (first) mutation.Remove(base);
        } catch (...) {
        }
        std::free(base);
    }
};

struct ActiveRecorder {
    std::unique_ptr<Recorder> recorder;

    explicit ActiveRecorder(const Context& context) {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder = std::make_unique<Recorder>(context);
        recorder->Activate();
    }

    ~ActiveRecorder() {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder->Sync();
        recorder.reset();
    }
};

void WindowPromotionOutlivesTheHold(const Device& device) {
    struct RestoreWindow {
        ~RestoreWindow() { SetHostImportRangeWindow(std::uint64_t{64} << 20u); }
    } restore;
    SetHostImportRangeWindow(Alignment);
    Heap heap;
    ActiveRecorder active(device.GetContext());
    auto& recorder = *active.recorder;
    const auto& context = device.GetContext();
    QueueGate gate(device);
    VkBuffer retired = VK_NULL_HANDLE;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto* taken = HostImportFor(context, heap.Address(), 4096);
        Expect(taken != nullptr && taken->bytes == Alignment, "the first access did not import one window");
        if (taken == nullptr) return;
        retired = taken->buffer;
        Track(retired);
        const auto* whole = HostImportFor(context, heap.Address(), RangeBytes);
        Expect(whole != nullptr && whole->base == heap.Address() && whole->bytes == RangeBytes && whole->buffer != retired, "a held import window could not be promoted to its whole registered range");
        if (whole == nullptr) return;
        Expect(Live(retired) && taken->buffer == retired && taken->bytes == Alignment, "promotion invalidated the held window or its entry");
        recorder.Commands();
        recorder.Submit();
        recorder.Sync();
        Expect(Live(retired), "a sync inside the promoting hold destroyed its old window");
        recorder.Commands();
        // Upstream leaves this batch's fence unsignaled: it stays in flight until the gate opens.
        gate.Hold();
        recorder.Submit();
    }
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        Expect(Live(retired), "the old window died while its hold's later batch was in flight");
    }
    gate.Open();
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder.Sync();
    }
    Expect(!Live(retired), "the old window survived every batch of the promoting hold");
}

void IdleRetireOutlivesTheHold(const Device& device) {
    Heap heap;
    ActiveRecorder active(device.GetContext());
    auto& recorder = *active.recorder;
    const auto& context = device.GetContext();
    QueueGate gate(device);
    VkBuffer retired = VK_NULL_HANDLE;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto* taken = HostImportFor(context, heap.Address(), 4096);
        Expect(taken != nullptr, "the first range was not imported");
        if (taken == nullptr) return;
        retired = taken->buffer;
        Track(retired);
        Expect(recorder.Idle(), "the recorder is not idle before the drop");
        heap.Grow();
        const auto* grown = HostImportFor(context, heap.Address(RangeBytes), 4096);
        Expect(grown != nullptr && grown->buffer != retired && grown->base == heap.Address() && grown->bytes == 2 * RangeBytes, "the grown run was not imported as one import replacing the first");
        Expect(Live(retired), "an import dropped with the recorder idle was destroyed at once, while the hold that took it can still record with it");
        if (!Live(retired)) return;
        Expect(taken->buffer == retired && taken->base == heap.Address(), "the dropped import's entry changed under the hold that took it");
        recorder.Commands();
        recorder.Submit();
        recorder.Sync();
        Expect(Live(retired), "an import dropped in this hold was destroyed by a Sync inside the same hold");
        recorder.Commands();
        // Upstream leaves this batch's fence unsignaled: it stays in flight until the gate opens.
        gate.Hold();
        recorder.Submit();
        Expect(Live(retired), "an import dropped in this hold was destroyed while a batch the hold recorded afterwards is in flight");
    }
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        Expect(Live(retired), "an import dropped in a hold was destroyed at the hold's unlock while a batch of that hold is in flight");
    }
    gate.Open();
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder.Sync();
    }
    Expect(!Live(retired), "the dropped import outlived every batch of its hold");
}

void BusyRetireOutlivesALaterBatch(const Device& device) {
    Heap heap;
    ActiveRecorder active(device.GetContext());
    auto& recorder = *active.recorder;
    const auto& context = device.GetContext();
    QueueGate gate(device);
    VkBuffer retired = VK_NULL_HANDLE;
    std::uint64_t first = 0;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder.Keep(std::make_shared<int>(0));
        const auto* taken = HostImportFor(context, heap.Address(), 4096);
        Expect(taken != nullptr, "the first range was not imported");
        if (taken == nullptr) return;
        retired = taken->buffer;
        Track(retired);
        heap.Grow();
        Expect(HostImportFor(context, heap.Address(RangeBytes), 4096) != nullptr, "the grown run was not imported");
        first = recorder.Submissions();
        recorder.Submit();
        recorder.Commands();
        // Upstream signals only the first batch's fence: the later batch stays in flight until the gate opens.
        gate.Hold();
        recorder.Submit();
        Expect(recorder.Submissions() == first + 2, "the hold did not submit the keeping batch and the later one apart");
        if (recorder.Submissions() != first + 2) return;
        recorder.WaitSerial(first + 1);
        recorder.Reap();
        Expect(Live(retired), "the dropped import was destroyed when the batch open at the drop completed, while a later batch of the hold that may use it is in flight");
    }
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        Expect(Live(retired), "the dropped import was destroyed at the hold's unlock, while a later batch of the hold that may use it is in flight");
        gate.Open();
        recorder.WaitSerial(first + 2);
        recorder.Reap();
    }
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder.Commands();
        recorder.Sync();
    }
    Expect(!Live(retired), "the dropped import outlived every batch of its hold");
}

void FailedImportPublishesNothing(const Device& device) {
    const auto& context = device.GetContext();
    void* mapped = mmap(nullptr, Alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapped == MAP_FAILED) throw std::runtime_error("cannot map the host import failure heap");
    auto* data = static_cast<std::byte*>(mapped);
    std::memset(data, 0x37, Alignment);
    struct Release {
        std::byte* data;
        ~Release() {
            try {
                GuestAllocations::Mutation().Remove(data);
            } catch (...) {
            }
            munmap(data, Alignment);
        }
    } release{data};
    GuestAllocations::Mutation().Add(data, Alignment, true, true, true);
    const auto address = reinterpret_cast<std::uint64_t>(data);
    constexpr std::size_t Bytes = 4096;
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    ClearHostImports(context.device);
    SetImportWatch(context, ImportWatch::Unwatch);
    std::size_t releasedBefore = 0;
    {
        std::lock_guard lock(destroyedMutex);
        releasedBefore = destroyed.size();
    }
    if (mprotect(data, Alignment, PROT_NONE) != 0) throw std::runtime_error("cannot make the host import failure heap unreadable");
    const HostImport* failed = nullptr;
    bool threw = false;
    try {
        failed = HostImportFor(context, address, Bytes);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    const bool rejected = failed == nullptr && ExistingHostImportFor(context, address, Bytes) == nullptr && HostImportLiveBytes() == 0;
    std::size_t releasedAfter = 0;
    {
        std::lock_guard lock(destroyedMutex);
        releasedAfter = destroyed.size();
    }
    if (mprotect(data, Alignment, PROT_READ | PROT_WRITE) != 0) throw std::runtime_error("cannot make the host import failure heap readable again");
    ClearHostImports(context.device);
    Expect(rejected, "a failed host import creation published an import or counted its bytes");
    Expect(releasedAfter == releasedBefore, "a failed host import creation released a buffer it never published");
    const auto* recovered = HostImportFor(context, address, Bytes);
    Expect(recovered != nullptr && recovered->buffer != VK_NULL_HANDLE && recovered->address != 0, "host import retry failed");
    if (recovered != nullptr && recovered->buffer != VK_NULL_HANDLE) {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(recovered->buffer);
        const auto offset = address - recovered->base;
        Expect(buffer.length >= offset + Bytes && std::memcmp(static_cast<const std::byte*>(buffer.contents) + offset, data, Bytes) == 0, "host import retry returned incorrect data");
        Track(recovered->buffer);
    }
    ClearHostImports(context.device);
    Expect(HostImportLiveBytes() == 0, "the retried host import was not cleaned up");
    std::printf("host import failure: threw=%u rejected=%u\n", threw ? 1u : 0u, rejected ? 1u : 0u);
}

}

int main() {
    setenv("APS5_RELEASE_ON_UNLOCK", "1", 1);
    setenv("APS5_WRITE_WATCH_IMPORTS", "unwatch", 1);
    AgcDriver::Metal::SetImportReleaseProbe(NoteDestroyed);
    try {
        @autoreleasepool {
            Device device;
            WindowPromotionOutlivesTheHold(device);
            IdleRetireOutlivesTheHold(device);
            BusyRetireOutlivesALaterBatch(device);
            FailedImportPublishesNothing(device);
        }
    } catch (const std::exception& error) {
        AgcDriver::Metal::SetImportReleaseProbe(nullptr);
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    AgcDriver::Metal::SetImportReleaseProbe(nullptr);
    if (failures != 0) {
        std::fprintf(stderr, "%d host import retire checks failed\n", failures);
        return 1;
    }
    std::printf("host import retire tests passed\n");
    return 0;
}
