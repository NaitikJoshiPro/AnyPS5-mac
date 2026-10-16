// A storage image over memory that recorded GPU work is still writing (a buffer fill dispatched in
// the same submission, before the image store) takes the snapshot behind its untracked comparison
// (`original`) where its upload reads the bytes. On Metal that snapshot is a recorded GPU copy that
// lands at the batch's completion (DECISIONS D82) instead of a CPU read after a recorder sync. The
// write-back then compares the guest bytes against it: a wrong snapshot (the bytes from before the
// fill) would make every block look CPU-written and drop the image's results, and a CPU write made
// while the results are pending must still be kept.
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::StorageTexture;
constexpr std::uint32_t Side = 256;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t TileR64KBX = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::size_t Page = 0x1000;
constexpr std::size_t Block = 0x10000;
constexpr std::size_t SurfaceBytes = 4 * Block;
constexpr std::uint32_t Untouched = 0xdeadbeefu;
constexpr std::uint32_t Stored = 0x12345678u;
constexpr std::uint32_t Filled = 0x9abcdef0u;

// Image store of `Stored` by 8 threads (the StoreOverUntracked program).
alignas(256) constexpr std::array<std::uint32_t, 7> StoreCode{
    0x7e040300, 0x7e060280, 0x7e0202ff, Stored, 0xf0200108, 0x00020102, 0xbf810000,
};

// Buffer store of the four user-data dwords 4..7 at 16 bytes per thread.
alignas(256) constexpr std::array<std::uint32_t, 9> FillCode{
    0xd7460004, 0x04010c08, 0x7e000204, 0x7e020205, 0x7e040206, 0x7e060207, 0xe01c2000, 0x80000004, 0xbf810000,
};

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::array<std::uint32_t, 8> TextureDescriptor(const std::uint32_t* texels) {
    const auto address = AddressOf(texels);
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format32UInt << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

void AppendDispatch(std::vector<std::uint32_t>& words, std::span<const std::uint32_t> code, std::uint32_t threads, std::uint32_t resource, std::span<const std::uint32_t> userData, std::uint32_t groups) {
    const auto registers = [&](const std::uint32_t first, const std::span<const std::uint32_t> values) {
        words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
        words.push_back(first);
        words.insert(words.end(), values.begin(), values.end());
    };
    const std::array<std::uint32_t, 3> shape{threads, 1, 1};
    registers(0x207, shape);
    const auto address = AddressOf(code.data());
    const std::array<std::uint32_t, 2> program{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u)};
    registers(0x20c, program);
    const std::array<std::uint32_t, 1> layout{resource};
    registers(0x213, layout);
    registers(0x240, userData);
    words.insert(words.end(), {0xc0031500u, groups, 1, 1, 0x41u});
}

// One submission: the fill of the whole surface, then the image store into it.
void FillThenStore(const std::uint32_t* texels) {
    std::vector<std::uint32_t> words;
    const auto address = AddressOf(texels);
    const std::array<std::uint32_t, 8> fill{
        static_cast<std::uint32_t>(address),
        static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (16u << 16u),
        static_cast<std::uint32_t>(SurfaceBytes / 16u),
        0x4bfacu,
        Filled,
        Filled,
        Filled,
        Filled,
    };
    AppendDispatch(words, FillCode, 64, (static_cast<std::uint32_t>(fill.size()) << 1u) | 0x80u, fill, static_cast<std::uint32_t>(SurfaceBytes / 1024u));
    std::array<std::uint32_t, 16> store{};
    const auto texture = TextureDescriptor(texels);
    std::copy(texture.begin(), texture.end(), store.begin() + 8);
    AppendDispatch(words, StoreCode, 8, static_cast<std::uint32_t>(store.size()) << 1u, store, 1);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
}

std::uint8_t Pattern(std::size_t offset) {
    return static_cast<std::uint8_t>(0x40u + offset % 0x3bu);
}

std::size_t Count(const std::vector<std::byte>& bytes, std::uint32_t value) {
    std::size_t count = 0;
    for (std::size_t offset = 0; offset + 4 <= bytes.size(); offset += 4) {
        std::uint32_t word;
        std::memcpy(&word, bytes.data() + offset, 4);
        if (word == value) ++count;
    }
    return count;
}

std::uint32_t Word(const std::vector<std::byte>& bytes, std::size_t offset) {
    std::uint32_t word;
    std::memcpy(&word, bytes.data() + offset, 4);
    return word;
}

enum class Outcome { Passed, Missed };

Outcome Run(bool cpuWrite, std::uint64_t& gpuSnapshots) {
    std::vector<std::uint32_t> storage((SurfaceBytes + Block) / 4u);
    auto* texels = reinterpret_cast<std::uint32_t*>((reinterpret_cast<std::uintptr_t>(storage.data()) + Block - 1) & ~std::uintptr_t{Block - 1});
    std::fill(texels, texels + SurfaceBytes / 4u, Untouched);
    GuestAllocations::Mutation().Add(texels, SurfaceBytes, true, true, true);
    struct Unregister {
        std::uint32_t* texels;
        ~Unregister() { GuestAllocations::Mutation().Remove(texels); }
    } unregister{texels};
    const auto address = AddressOf(texels);
#ifdef __APPLE__
    const auto before = StorageTexture::TakeSnapshotCounts();
#endif
    std::unique_lock hold(AgcDriver::GuestMemory::GpuMutex(), std::defer_lock);
    FillThenStore(texels);
    // The store's results stay pending in the image (or reached guest memory already: a miss).
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        hold.lock();
        if (StorageTexture::AnyPending()) break;
        hold.unlock();
        // Raw bytes, not a driver read (that would store the pending results).
        const auto* words = static_cast<const volatile std::uint32_t*>(texels);
        for (std::size_t index = 0; index < SurfaceBytes / 4u; ++index) {
            if (words[index] == Stored) return Outcome::Missed;
        }
        Require(std::chrono::steady_clock::now() < deadline, "storage gpu snapshot: the image store neither became pending nor reached guest memory");
        std::this_thread::yield();
    }
#ifdef __APPLE__
    gpuSnapshots += StorageTexture::TakeSnapshotCounts().gpu - before.gpu;
#endif
    hold.unlock();
    // A CPU write into the last block while the results are pending (untracked memory: only the
    // comparison against the snapshot tells it apart).
    auto* bytes = reinterpret_cast<std::uint8_t*>(texels) + 3 * Block + Page;
    if (cpuWrite) {
        for (std::size_t offset = 0; offset < Page; ++offset) static_cast<volatile std::uint8_t*>(bytes)[offset] = Pattern(offset);
    }
    hold.lock();
    StorageTexture::FlushAllPending("storage gpu snapshot fixture");
    hold.unlock();
    AgcDriverWaitIdle_nid_postfix();
    std::vector<std::byte> whole(SurfaceBytes);
    AgcDriver::GuestMemory::Read(address, whole);
    const auto stored = Count(whole, Stored);
    const auto filled = Count(whole, Filled);
    std::fprintf(stderr, "[storage gpu snapshot] cpuWrite %d: stored %zu filled %zu untouched %zu; first words %08x %08x %08x, block 1 %08x, block 3 %08x\n", cpuWrite ? 1 : 0, stored, filled, Count(whole, Untouched), Word(whole, 0), Word(whole, 4), Word(whole, 64), Word(whole, Block), Word(whole, 3 * Block));
    Require(stored == 8u, "storage gpu snapshot: " + std::to_string(stored) + " of the 8 stored texels reached guest memory (the write-back compared against a wrong snapshot)");
    if (cpuWrite) {
        // The 64 KiB block the CPU wrote keeps the CPU's view of it (the write-back's keep rule);
        // the other blocks hold the fill and the stored texels.
        for (const std::size_t block : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) Require(Word(whole, block * Block + 64) == Filled, "storage gpu snapshot: block " + std::to_string(block) + " lost the fill");
        std::size_t lost = 0;
        for (std::size_t offset = 0; offset < Page; ++offset) lost += static_cast<std::size_t>(static_cast<std::uint8_t>(whole[3 * Block + Page + offset]) != Pattern(offset));
        Require(lost == 0u, "storage gpu snapshot: " + std::to_string(lost) + " CPU-written bytes were lost to the write-back");
    } else {
        Require(Count(whole, Untouched) == 0u, "storage gpu snapshot: bytes from before the fill came back (the image was uploaded from them)");
        Require(filled == SurfaceBytes / 4u - 8u, "storage gpu snapshot: the fill's bytes did not survive the write-back");
    }
    return Outcome::Passed;
}

}

int main() {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        struct ShutdownGuard {
            ~ShutdownGuard() { try { AgcDriverShutdown_nid_postfix(); } catch (...) {} }
        } shutdown;
        std::uint64_t gpuSnapshots = 0;
        for (const bool cpuWrite : {false, true}) {
            auto outcome = Outcome::Missed;
            for (std::uint32_t attempt = 0; attempt < 64u && outcome == Outcome::Missed; ++attempt) outcome = Run(cpuWrite, gpuSnapshots);
            Require(outcome == Outcome::Passed, "storage gpu snapshot: the image store was never held pending");
        }
#ifdef __APPLE__
        if (std::getenv("APS5_CPU_TEXTURE_SNAPSHOT") == nullptr) Require(gpuSnapshots != 0u, "storage gpu snapshot: the snapshot of fill-written memory was read on the CPU, not copied on the GPU");
        std::printf("storage gpu snapshot: %llu snapshots copied on the GPU\n", static_cast<unsigned long long>(gpuSnapshots));
#endif
        AgcDriverShutdown_nid_postfix();
        std::puts("storage gpu snapshot tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
