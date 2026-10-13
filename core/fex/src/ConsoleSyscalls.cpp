#include "ConsoleSyscalls.hpp"

#include "Bridge.hpp"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <sys/mman.h>

namespace Aps5Fex {

namespace {

constexpr std::size_t GadgetPage = 0x4000;
constexpr std::size_t HandlerImmediate = 32;

constexpr std::uint8_t GadgetCode[] {
    0x41, 0x51,
    0x41, 0x50,
    0x41, 0x52,
    0x52,
    0x56,
    0x57,
    0x50,
    0x48, 0x83, 0xec, 0x20,
    0xf3, 0x0f, 0x7f, 0x04, 0x24,
    0xf3, 0x0f, 0x7f, 0x4c, 0x24, 0x10,
    0x48, 0x8d, 0x7c, 0x24, 0x20,
    0x49, 0xbb, 0, 0, 0, 0, 0, 0, 0, 0,
    0x41, 0xff, 0xd3,
    0xf3, 0x0f, 0x6f, 0x04, 0x24,
    0xf3, 0x0f, 0x6f, 0x4c, 0x24, 0x10,
    0x48, 0x83, 0xc4, 0x28,
    0x5f,
    0x5e,
    0x5a,
    0x41, 0x5a,
    0x41, 0x58,
    0x41, 0x59,
    0x48, 0x85, 0xc0,
    0x79, 0x05,
    0x48, 0xf7, 0xd8,
    0xf9,
    0xc3,
    0xc3,
};

static_assert(GadgetCode[HandlerImmediate - 2] == 0x49 && GadgetCode[HandlerImmediate - 1] == 0xbb);

}

std::uint64_t CreateConsoleSyscallGadget(Bridge& bridge) {
    static std::mutex mutex;
    static std::uint64_t gadget = 0;
    std::lock_guard lock(mutex);
    if (gadget != 0) return gadget;
    void* handler = bridge.HostSymbol("Aps5ConsoleSyscall_nid_no_patch");
    if (handler == nullptr) return 0;
    const std::uint64_t stub = bridge.StubFor(handler, "Aps5ConsoleSyscall");
    if (stub == 0) throw std::runtime_error("the console system call handler has no guest stub");
    void* page = mmap(nullptr, GadgetPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (page == MAP_FAILED) throw std::runtime_error("cannot allocate the console system call gadget");
    auto* code = static_cast<std::uint8_t*>(page);
    std::memset(page, 0xcc, GadgetPage);
    std::memcpy(code, GadgetCode, sizeof(GadgetCode));
    std::memcpy(code + HandlerImmediate, &stub, sizeof(stub));
    if (mprotect(page, GadgetPage, PROT_READ | PROT_EXEC) != 0) throw std::runtime_error("cannot protect the console system call gadget");
    gadget = reinterpret_cast<std::uint64_t>(page);
    return gadget;
}

}
