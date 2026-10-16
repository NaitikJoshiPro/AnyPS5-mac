// Metal twin of tests/RecorderKeptBytes.cpp (work package D2): the recorder's kept-bytes budget
// (Keep, KeepBytes, BoundKeptBytes) on the Metal recorder. The upstream test counts vkQueueSubmit
// and fence waits through a mock Vulkan device; here batches are real (empty) MTL4 command buffers,
// submissions come from Recorder::Submissions() and a batch the recorder waited for is one whose
// kept objects it released. FailedSubmission ports upstream's mocked vkEndCommandBuffer and
// vkQueueSubmit failures through Metal::FailNextRecorderSubmit (a batch failing after its command
// buffer ended, or at its commit); Metal's commit reports no device loss, so upstream's
// VK_ERROR_DEVICE_LOST idle-wait cases have no counterpart. The earlier batch is held in flight by
// a GPU event wait on the queue (upstream leaves its mock fence unsignaled).

#include "MetalTestDevice.h"
#include "../src/MetalCommandStream.h"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

using namespace AgcDriver::Graphics;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr std::size_t Budget = Recorder::KeptBytesBudget;

void OpenBatchUnderBudget(const Context& context) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    auto first = std::make_shared<int>(0);
    std::weak_ptr<int> watched = first;
    recorder.Keep(std::move(first));
    recorder.Keep(std::make_shared<int>(1), Budget - 1);
    recorder.BoundKeptBytes();
    Expect(recorder.Recording() && recorder.Submissions() == 0, "a batch keeping one byte less than the budget was submitted");
    recorder.Keep(std::make_shared<int>(2), 1);
    recorder.BoundKeptBytes();
    Expect(!recorder.Recording() && recorder.Submissions() == 1, "the open batch was not submitted once its kept bytes reached the budget");
    Expect(recorder.InFlightKeptBytes() == Budget, "the submitted batch counts " + std::to_string(recorder.InFlightKeptBytes()) + " kept bytes in flight, not the budget");
    Expect(recorder.InFlightBatches() == 1 && !watched.expired(), "the recorder finished a batch with one budget in flight");
    recorder.Sync();
    Expect(recorder.InFlightKeptBytes() == 0, "a synced recorder still counts " + std::to_string(recorder.InFlightKeptBytes()) + " kept bytes in flight");
}

void InFlightWithinTwiceTheBudget(const Context& context) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    std::vector<std::weak_ptr<int>> batches;
    for (int batch = 0; batch < 6; ++batch) {
        auto object = std::make_shared<int>(batch);
        batches.push_back(object);
        recorder.Keep(std::move(object), Budget / 2);
        recorder.Keep(std::make_shared<int>(batch), Budget / 2);
        recorder.BoundKeptBytes();
        Expect(recorder.InFlightKeptBytes() <= 2 * Budget, "batch " + std::to_string(batch) + " left " + std::to_string(recorder.InFlightKeptBytes()) + " kept bytes in flight, more than twice the budget");
    }
    Expect(recorder.Submissions() == 6, "six budget batches made " + std::to_string(recorder.Submissions()) + " submissions");
    Expect(recorder.InFlightBatches() == 2 && recorder.InFlightKeptBytes() == 2 * Budget, "the newest two batches are not the ones in flight");
    for (std::size_t batch = 0; batch < batches.size(); ++batch) {
        const bool released = batches[batch].expired();
        Expect(released == (batch < 4), "batch " + std::to_string(batch) + (released ? " released its kept objects while in flight" : " still holds its kept objects under the GPU mutex after the wait for it"));
    }
    recorder.Sync();
}

void SyncedWithinTwiceTheBudget(const Context& context) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    std::vector<std::weak_ptr<int>> batches;
    const auto alive = [&] { return std::count_if(batches.begin(), batches.end(), [](const auto& batch) { return !batch.expired(); }); };
    for (int batch = 0; batch < 6; ++batch) {
        auto object = std::make_shared<int>(batch);
        batches.push_back(object);
        recorder.Keep(std::move(object), Budget);
        recorder.BoundKeptBytes();
        recorder.Sync();
        Expect(alive() <= 2, "after batch " + std::to_string(batch) + ", " + std::to_string(alive()) + " synced budget batches still hold their kept objects under the GPU mutex");
    }
    Expect(recorder.Submissions() == 6 && recorder.InFlightKeptBytes() == 0, "six synced budget batches made " + std::to_string(recorder.Submissions()) + " submissions and left " + std::to_string(recorder.InFlightKeptBytes()) + " kept bytes in flight");
    Expect(!batches[4].expired() && !batches[5].expired(), "the newest two synced batches released their kept objects before the unlock");
}

void CountFollowsEveryPath(const Context& context) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    recorder.Keep(std::make_shared<int>(0), 3 * Budget);
    recorder.BoundKeptBytes();
    Expect(recorder.Submissions() == 1 && recorder.InFlightBatches() == 0 && recorder.InFlightKeptBytes() == 0, "a batch keeping three budgets was not submitted, waited for and released");
    recorder.Keep(std::make_shared<int>(1), Budget);
    recorder.Submit();
    Expect(recorder.InFlightKeptBytes() == Budget, "a plain Submit did not count the batch's kept bytes in flight");
    // The upstream test signals the mock fence; here the empty batch completes on the GPU.
    recorder.WaitSerial(recorder.Submissions());
    Expect(recorder.Reap() && recorder.InFlightKeptBytes() == 0, "a reaped batch still counts " + std::to_string(recorder.InFlightKeptBytes()) + " kept bytes in flight");
}

void SharedOwnerCountsOncePerBatch(const Context& context) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    Recorder recorder(context);
    auto shared = std::make_shared<int>(0);
    for (int keep = 0; keep < 3; ++keep) {
        recorder.Keep(shared);
        recorder.KeepBytes(shared.get(), Budget / 2);
        recorder.BoundKeptBytes();
    }
    Expect(recorder.Recording() && recorder.OpenKeptBytes() == Budget / 2 && recorder.Submissions() == 0, "three keeps of one owner's half budget counted " + std::to_string(recorder.OpenKeptBytes()) + " kept bytes, not half the budget once");
    auto other = std::make_shared<int>(1);
    recorder.Keep(other);
    recorder.KeepBytes(other.get(), Budget / 2);
    recorder.BoundKeptBytes();
    Expect(!recorder.Recording() && recorder.Submissions() == 1 && recorder.InFlightKeptBytes() == Budget, "a second owner's half budget did not complete the batch's budget");
    recorder.Keep(shared);
    recorder.KeepBytes(shared.get(), Budget / 2);
    Expect(recorder.OpenKeptBytes() == Budget / 2, "the next batch did not count an owner the submitted batch counted");
    recorder.Sync();
}

void FailedSubmission(const MetalTest::Device& device, bool beforeCommit) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    alignas(8) std::array<std::byte, 128> memory{};
    const auto address = reinterpret_cast<std::uint64_t>(memory.data());
    const char* operation = beforeCommit ? "MTL4 command buffer end (test failure)" : "MTL4 commit (test failure)";
    bool completed = false;
    bool earlierCompleted = false;
    bool earlierWrittenBack = false;
    id<MTLSharedEvent> gate = [device.device newSharedEvent];
    if (gate == nil) {
        Expect(false, "cannot make the queue gate");
        return;
    }
    [device.queue waitForEvent:gate value:1];
    {
        Recorder recorder(device.context);
        // Declared after the recorder: an exception opens the gate before the recorder waits for its batches.
        struct OpenGate {
            id<MTLSharedEvent> gate;
            ~OpenGate() { gate.signaledValue = 1; }
        } openGate{gate};
        recorder.Activate();
        recorder.Keep(std::make_shared<int>(0), 17);
        recorder.NoteLabel(address, std::span<const std::byte>(memory).first(4), 1, 0);
        recorder.OnComplete([&] { earlierWrittenBack = true; });
        Expect(recorder.AfterRecordedWork([&] { earlierCompleted = true; }), "the earlier batch did not accept its completion");
        recorder.Submit();
        auto kept = std::make_shared<int>(1);
        std::weak_ptr<int> released = kept;
        recorder.Keep(std::move(kept), Budget);
        recorder.NoteLabel(address + 8, std::span<const std::byte>(memory).first(4), 2, 0);
        recorder.NoteLabel(address + 16, std::span<const std::byte>(memory).first(Recorder::LabelTableBytes + 4), 3, 0);
        recorder.AfterCompletions(address + 96, std::span<const std::byte>(memory).first(4), 4, 0, false);
        recorder.OnComplete([&] { completed = true; });
        Expect(recorder.AfterRecordedWork([&] { completed = true; }), "the failed batch did not accept its completion");
        Metal::FailNextRecorderSubmit(operation, beforeCommit);
        bool failed = false;
        try {
            recorder.Submit();
        } catch (const std::exception& error) {
            failed = std::string_view(error.what()).find(operation) != std::string_view::npos;
        }
        Expect(failed, std::string("submission did not report its failure: ") + operation);
        Expect(released.expired() && !completed, "a failed batch retained objects or ran its completions");
        Expect(!recorder.Recording() && recorder.Submissions() == 1 && recorder.InFlightBatches() == 1 && recorder.InFlightKeptBytes() == 17,
            "the failed batch changed the earlier in-flight submission");
        Expect(recorder.PendingLabels() == 1 && recorder.PendingLabelIn(address, 4), "a failed batch retained labels or removed the earlier batch's label");
        Expect(!recorder.PendingLabelIn(address + 8, 4) && !recorder.PendingLabelIn(address + 16, Recorder::LabelTableBytes + 4),
            "a failed batch retained a narrow or wide label");
        Expect(!recorder.CompletionLabelIn(address + 96, 4) && Recorder::PendingCompletionLabels() == 1 && Recorder::PendingWriteBackCompletions() == 1,
            "a failed batch retained pending completions or removed earlier ones");
        Expect(!Recorder::PendingLabelSince().has_value(), "a failed batch left its label flush deadline pending");
        Expect(Recorder::SnapshotWriteOverlaps(address, 4) && !Recorder::SnapshotWriteOverlaps(address + 8, 100),
            "a failed batch left stale writes in the pending snapshot or removed an earlier write");
        gate.signaledValue = 1;
        recorder.Sync();
        Expect(recorder.Idle() && !completed && earlierCompleted && earlierWrittenBack,
            "sync ran the failed batch's completions or lost the earlier batch's completions");
        Expect(Recorder::PendingCompletionLabels() == 0 && Recorder::PendingWriteBackCompletions() == 0,
            "sync left stale completion counts after submission failure");
    }
}

}

int main() {
    @autoreleasepool {
        MetalTest::Device device;
        try {
            OpenBatchUnderBudget(device.context);
            InFlightWithinTwiceTheBudget(device.context);
            SyncedWithinTwiceTheBudget(device.context);
            CountFollowsEveryPath(device.context);
            SharedOwnerCountsOncePerBatch(device.context);
            FailedSubmission(device, true);
            FailedSubmission(device, false);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 1;
        }
        if (failures != 0) {
            std::fprintf(stderr, "%d recorder kept bytes checks failed\n", failures);
            return 1;
        }
        std::printf("recorder kept bytes tests passed\n");
    }
    return 0;
}
