#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string_view>
#include <thread>

namespace AgcDriver::DriverDetail {

void Driver::noteLabelStore(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue) {
    if (!KeepsLabelStore(address, bytes.size())) return;
    std::lock_guard lock(labelStoresMutex);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 4) {
        auto& history = labelStores[address + offset];
        std::shift_right(history.begin(), history.end(), 1);
        history[0].stamp = stamp;
        history[0].queue = queue;
        std::memcpy(&history[0].value, bytes.data() + offset, 4);
    }
}

bool Driver::storedSince(std::span<const std::uint32_t> packet, std::uint64_t address, std::size_t bytes, std::uint64_t received, std::uint32_t* writer) {
    std::lock_guard lock(labelStoresMutex);
    const auto low = labelStores.find(address);
    if (low == labelStores.end()) return false;
    const auto high = bytes == 8 ? labelStores.find(address + 4) : labelStores.end();
    if (bytes == 8 && high == labelStores.end()) return false;
    for (const auto& store : low->second) {
        if (store.stamp <= received) continue;
        std::uint64_t value = store.value;
        if (bytes == 8) {
            const auto upper = std::find_if(high->second.begin(), high->second.end(), [&](const LabelStore& other) { return other.stamp == store.stamp; });
            if (upper == high->second.end()) continue;
            value |= static_cast<std::uint64_t>(upper->value) << 32u;
        }
        if (Pm4::WaitComparesValue(packet, value)) {
            if (writer != nullptr) *writer = store.queue;
            return true;
        }
    }
    return false;
}

namespace {

// APS5_TRACE_LABEL_LANDING=<ms> (with APS5_TRACE_LABEL): a watcher thread polls the guest address
// of every traced RELEASE_MEM every 0.5 ms and reports when the CPU first sees the written value
// there, or what it saw instead after <ms> (default 1000). It answers "did the label land where the
// guest polls it", which the driver's own waits (label table shortcut) cannot.
struct LandingWatch {
    std::uint64_t address;
    std::uint64_t value;
    std::size_t bytes;
    double since;
};

void WatchLanding(std::uint64_t address, std::uint64_t value, std::size_t bytes) {
    static const double limitMs = [] {
        const char* value = std::getenv("APS5_TRACE_LABEL_LANDING");
        if (value == nullptr) return 0.0;
        const double parsed = std::strtod(value, nullptr);
        return parsed > 1.0 ? parsed : 1000.0;
    }();
    if (limitMs == 0.0 || address == 0 || address % bytes != 0) return;
    static std::mutex mutex;
    static std::condition_variable changed;
    static std::deque<LandingWatch> watches;
    static const bool started = [] {
        std::thread([] {
            std::unique_lock lock(mutex);
            for (;;) {
                changed.wait_for(lock, std::chrono::microseconds(500));
                const double now = TraceMs();
                for (auto it = watches.begin(); it != watches.end();) {
                    std::uint64_t seen = 0;
                    const bool readable = GuestMemory::Accessible(reinterpret_cast<const void*>(it->address), it->bytes);
                    if (readable) seen = it->bytes == 8 ? *reinterpret_cast<const volatile std::uint64_t*>(it->address) : *reinterpret_cast<const volatile std::uint32_t*>(it->address);
                    if (readable && seen == it->value) {
                        std::fprintf(stderr, "[landing] %.1f 0x%llx == 0x%llx seen by the CPU after %.1f ms\n", now, static_cast<unsigned long long>(it->address), static_cast<unsigned long long>(it->value), now - it->since);
                        it = watches.erase(it);
                    } else if (now - it->since > limitMs) {
                        std::fprintf(stderr, "[landing] %.1f 0x%llx did not show 0x%llx in %.0f ms (CPU sees 0x%llx%s)\n", now, static_cast<unsigned long long>(it->address), static_cast<unsigned long long>(it->value), limitMs, static_cast<unsigned long long>(seen), readable ? "" : ", unreadable");
                        it = watches.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }).detach();
        return true;
    }();
    static_cast<void>(started);
    std::lock_guard lock(mutex);
    watches.push_back({address, value, bytes, TraceMs()});
    changed.notify_one();
}

}

void Driver::traceLabel(std::span<const std::uint32_t> packet, std::uint32_t queue) {
    static const std::uint64_t watched = [] {
        // A hex guest address (packets within 0x100 of it), or "all" for every label packet.
        const char* value = std::getenv("APS5_TRACE_LABEL");
        if (value != nullptr && std::string_view(value) == "all") return ~0ull;
        return value ? std::strtoull(value, nullptr, 16) : 0ull;
    }();
    const auto opcode = (packet[0] >> 8u) & 0xffu;
    std::uint64_t target = 0;
    const char* kind = nullptr;
    if (opcode == 0x49 && packet.size() >= 7) { target = packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u); kind = "RELEASE_MEM"; }
    else if (opcode == 0x37 && packet.size() >= 5) { target = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u); kind = "WRITE_DATA"; }
    else if ((opcode == 0x3c || opcode == 0x93) && packet.size() >= 7) { target = (packet[2] & (Pm4::WaitAwaitedBytes(packet) == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u); kind = "WAIT_REG_MEM"; }
    else if (opcode == 0x40 && packet.size() >= 6) { target = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u); kind = "COPY_DATA"; }
    std::uint64_t length = 4;
    if (opcode == 0x50 && packet.size() >= 7) {
        target = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
        length = packet[6] & 0x3ffffffu;
        kind = "DMA_DATA";
    }
    if (kind == nullptr) return;
    if (std::string_view(kind) != "WAIT_REG_MEM") {
        static std::mutex historyMutex;
        std::lock_guard lock(historyMutex);
        auto& entry = writeHistory()[writeCursor()++ % writeHistory().size()];
        entry = {target, length, queue, opcode};
    }
    if (watched == 0 || (watched != ~0ull && (target + length + 0x100 < watched || target > watched + 0x100))) return;
    if (opcode == 0x49 && packet.size() >= 7) {
        // RELEASE_MEM DATA_SEL (dword 2 bits 29-31): 1 = 32-bit data, 2 = 64-bit data.
        const auto dataSel = (packet[2] >> 29u) & 7u;
        if (dataSel == 1) WatchLanding(target, packet[5], 4);
        else if (dataSel == 2) WatchLanding(target, packet[5] | (static_cast<std::uint64_t>(packet[6]) << 32u), 8);
    }
    std::fprintf(stderr, "[label] %lld ms queue 0x%x %s 0x%llx:", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()), queue, kind, static_cast<unsigned long long>(target));
    for (std::size_t i = 1; i < packet.size() && i < 9; ++i) std::fprintf(stderr, " %08x", packet[i]);
    std::fprintf(stderr, "\n");
}

std::array<WriteRecord, 16384>& Driver::writeHistory() {
    static std::array<WriteRecord, 16384> history{};
    return history;
}

std::size_t& Driver::writeCursor() {
    static std::size_t cursor = 0;
    return cursor;
}

}
