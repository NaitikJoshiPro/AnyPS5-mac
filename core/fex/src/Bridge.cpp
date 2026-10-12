#include "Bridge.hpp"

#include "GuestCpu.hpp"
#include "GuestImage.hpp"

#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>

struct Aps5NativeArguments {
    std::uint64_t gpr[8];
    std::uint8_t vector[8][16];
    const std::uint64_t* stack;
    std::uint64_t stackSlots;
};
static_assert(offsetof(Aps5NativeArguments, vector) == 64 && offsetof(Aps5NativeArguments, stack) == 192 &&
              offsetof(Aps5NativeArguments, stackSlots) == 200);

struct Aps5NativeResult {
    std::uint64_t gpr[2];
    std::uint8_t vector[2][16];
};
static_assert(offsetof(Aps5NativeResult, vector) == 16);

extern "C" void Aps5NativeCall(const void* target, const Aps5NativeArguments* arguments, Aps5NativeResult* result);

namespace Aps5Fex {

namespace {

constexpr std::size_t BlockSize = 0xb8;
constexpr std::size_t BlockVectors = 0x30;

constexpr std::uint64_t StackArgumentSlots = 8;

using namespace FEXCore::X86State;

struct GuestCall {
    FEXCore::Core::CPUState* state;
    std::uint64_t block;
    const GuestCall* outer;
};

thread_local const GuestCall* currentCall = nullptr;

std::size_t ReadableBytes(const void* address, std::size_t size) {
    constexpr std::uintptr_t Page = 0x4000;
    const auto first = reinterpret_cast<std::uintptr_t>(address);
    const auto next = (first & ~(Page - 1)) + Page;
    if (first + size <= next) return size;
    char resident = 0;
    if (mincore(reinterpret_cast<void*>(next), 1, &resident) == 0) return size;
    return next - first;
}

constexpr std::uint32_t DwarfRegisters[16] = {REG_RAX, REG_RDX, REG_RCX, REG_RBX, REG_RSI, REG_RDI, REG_RBP, REG_RSP,
                                              REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};

void CaptureGuestFrame(std::uintptr_t* registers) {
    if (currentCall == nullptr) {
        std::fprintf(stderr, "[aps5-fex] a library unwound outside a guest call\n");
        std::abort();
    }
    const auto* saved = reinterpret_cast<const std::uint64_t*>(currentCall->block);
    for (std::size_t i = 0; i < 16; ++i) registers[i] = currentCall->state->gregs[DwarfRegisters[i]];
    registers[5] = saved[0];
    registers[4] = saved[1];
    registers[1] = saved[2];
    registers[2] = saved[3];
    registers[8] = saved[4];
    registers[9] = saved[5];
    registers[7] = currentCall->block + BlockSize + 8;
    registers[16] = *reinterpret_cast<const std::uint64_t*>(currentCall->block + BlockSize);
}

struct GuestFrame {
    std::uintptr_t registers[17];
};

[[noreturn]] void ResumeGuestFrame(const std::uintptr_t* registers) {
    GuestFrame frame;
    std::memcpy(frame.registers, registers, sizeof(frame.registers));
    throw frame;
}

std::vector<std::uint64_t> tlsOffsets;

void* TlsGetAddr(const std::uint64_t* index) {
    if (currentCall == nullptr || index[0] >= tlsOffsets.size()) {
        std::fprintf(stderr, "[aps5-fex] __tls_get_addr for TLS module %llu\n", static_cast<unsigned long long>(index[0]));
        std::abort();
    }
    return reinterpret_cast<void*>(currentCall->state->fs_cached - tlsOffsets[index[0]] + index[1]);
}

void GuestControl(std::uint32_t* mxcsr, std::uint16_t* fcw) {
    if (currentCall == nullptr) {
        std::fprintf(stderr, "[aps5-fex] a library read the guest's control words outside a guest call\n");
        std::abort();
    }
    *mxcsr = currentCall->state->mxcsr;
    *fcw = currentCall->state->FCW;
}

void FiberEntryReturned() {
    std::fprintf(stderr, "[aps5-fex] the entry function of a fiber returned\n");
    std::abort();
}

struct ImageInfo {
    std::uintptr_t address;
    const char* name;
    const void* programHeaders;
    std::uint16_t programHeaderCount;
};

struct VaList {
    std::uint32_t gpOffset;
    std::uint32_t fpOffset;
    void* overflowArea;
    void* registerSaveArea;
};
static_assert(sizeof(VaList) == 24);

struct VariadicExport {
    std::string_view name;
    std::uint8_t kind;
    std::uint8_t fixed;
};

constexpr std::uint8_t GuestList = 1;
constexpr std::uint8_t Integers = 2;
constexpr std::uint8_t Unsupported = 3;

constexpr VariadicExport VariadicExports[] = {
    {"asprintf_nid_postfix", GuestList, 2},
    {"fprintf_nid_postfix", GuestList, 2},
    {"fscanf_nid_postfix", GuestList, 2},
    {"libc_printf_nid_postfix", GuestList, 1},
    {"printf_nid_postfix", GuestList, 1},
    {"printf_s_nid_postfix", GuestList, 1},
    {"snprintf_nid_postfix", GuestList, 3},
    {"snprintf_s_nid_postfix", GuestList, 3},
    {"snwprintf_s_nid_postfix", GuestList, 3},
    {"sprintf_nid_postfix", GuestList, 2},
    {"sprintf_s_nid_postfix", GuestList, 3},
    {"sscanf_nid_postfix", GuestList, 2},
    {"sscanf_s_nid_postfix", GuestList, 2},
    {"wprintf_nid_postfix", GuestList, 1},
    {"_open_nid_postfix", Integers, 2},
    {"fcntl_nid_postfix", Integers, 2},
    {"swprintf_nid_postfix", Unsupported, 3},
};

constexpr std::string_view X87Results[] = {"strtold_nid_postfix", "wcstold_nid_postfix"};

template <std::size_t... Sizes>
constexpr std::array<std::uint8_t, (Sizes + ...)> Join(const std::array<std::uint8_t, Sizes>&... parts) {
    std::array<std::uint8_t, (Sizes + ...)> joined {};
    std::size_t at = 0;
    ((std::copy(parts.begin(), parts.end(), joined.begin() + at), at += Sizes), ...);
    return joined;
}

using Bytes4 = std::array<std::uint8_t, 4>;
using Bytes5 = std::array<std::uint8_t, 5>;
using Bytes6 = std::array<std::uint8_t, 6>;
using Bytes7 = std::array<std::uint8_t, 7>;
using Bytes9 = std::array<std::uint8_t, 9>;

constexpr Bytes7 SubRspBlock {0x48, 0x81, 0xec, 0xb8, 0x00, 0x00, 0x00};
constexpr Bytes7 AddRspBlock {0x48, 0x81, 0xc4, 0xb8, 0x00, 0x00, 0x00};
constexpr Bytes4 StoreRdi {0x48, 0x89, 0x3c, 0x24};
constexpr Bytes5 StoreRsi {0x48, 0x89, 0x74, 0x24, 0x08};
constexpr Bytes5 StoreRdx {0x48, 0x89, 0x54, 0x24, 0x10};
constexpr Bytes5 StoreRcx {0x48, 0x89, 0x4c, 0x24, 0x18};
constexpr Bytes5 StoreR8 {0x4c, 0x89, 0x44, 0x24, 0x20};
constexpr Bytes5 StoreR9 {0x4c, 0x89, 0x4c, 0x24, 0x28};
constexpr Bytes6 StoreXmm0 {0xf3, 0x0f, 0x7f, 0x44, 0x24, 0x30};
constexpr Bytes6 StoreXmm1 {0xf3, 0x0f, 0x7f, 0x4c, 0x24, 0x40};
constexpr Bytes6 StoreXmm2 {0xf3, 0x0f, 0x7f, 0x54, 0x24, 0x50};
constexpr Bytes6 StoreXmm3 {0xf3, 0x0f, 0x7f, 0x5c, 0x24, 0x60};
constexpr Bytes6 StoreXmm4 {0xf3, 0x0f, 0x7f, 0x64, 0x24, 0x70};
constexpr Bytes9 StoreXmm5 {0xf3, 0x0f, 0x7f, 0xac, 0x24, 0x80, 0x00, 0x00, 0x00};
constexpr Bytes9 StoreXmm6 {0xf3, 0x0f, 0x7f, 0xb4, 0x24, 0x90, 0x00, 0x00, 0x00};
constexpr Bytes9 StoreXmm7 {0xf3, 0x0f, 0x7f, 0xbc, 0x24, 0xa0, 0x00, 0x00, 0x00};
constexpr std::array<std::uint8_t, 3> MoveRspToRdi {0x48, 0x89, 0xe7};
constexpr std::array<std::uint8_t, 2> Syscall {0x0f, 0x05};
constexpr Bytes4 LoadRax {0x48, 0x8b, 0x04, 0x24};
constexpr Bytes5 LoadRdx {0x48, 0x8b, 0x54, 0x24, 0x08};
constexpr Bytes6 LoadXmm0 {0xf3, 0x0f, 0x6f, 0x44, 0x24, 0x30};
constexpr Bytes6 LoadXmm1 {0xf3, 0x0f, 0x6f, 0x4c, 0x24, 0x40};
constexpr std::array<std::uint8_t, 3> LoadSt0 {0xdb, 0x2c, 0x24};
constexpr std::array<std::uint8_t, 1> Return {0xc3};

constexpr auto TrampolineCall = Join(SubRspBlock, StoreRdi, StoreRsi, StoreRdx, StoreRcx, StoreR8, StoreR9, StoreXmm0, StoreXmm1,
                                     StoreXmm2, StoreXmm3, StoreXmm4, StoreXmm5, StoreXmm6, StoreXmm7, MoveRspToRdi, Syscall);
constexpr auto TrampolineReturn = Join(LoadRax, LoadRdx, LoadXmm0, LoadXmm1, AddRspBlock, Return);
constexpr auto X87TrampolineReturn = Join(LoadSt0, AddRspBlock, Return);

constexpr std::uint8_t MoveToEax = 0xb8;
constexpr std::uint8_t JumpRelative = 0xe9;

}

Bridge::Bridge(std::filesystem::path libraries) : directory(std::move(libraries)), trace(std::getenv("APS5_FEX_TRACE") != nullptr) {
    void* memory = mmap(nullptr, StubSize(), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (memory == MAP_FAILED) throw std::runtime_error("cannot allocate the import stubs");
    stubs = static_cast<std::uint8_t*>(memory);
    static_assert(TrampolineOffset + TrampolineCall.size() + TrampolineReturn.size() <= X87TrampolineOffset);
    static_assert(X87TrampolineOffset + TrampolineCall.size() + X87TrampolineReturn.size() <= FirstStubOffset);
    std::memset(stubs, 0xcc, FirstStubOffset);
    stubs[0] = 0x0f;
    stubs[1] = 0x3e;
    std::memcpy(stubs + TrampolineOffset, TrampolineCall.data(), TrampolineCall.size());
    std::memcpy(stubs + TrampolineOffset + TrampolineCall.size(), TrampolineReturn.data(), TrampolineReturn.size());
    std::memcpy(stubs + X87TrampolineOffset, TrampolineCall.data(), TrampolineCall.size());
    std::memcpy(stubs + X87TrampolineOffset + TrampolineCall.size(), X87TrampolineReturn.data(), X87TrampolineReturn.size());
}

Bridge::~Bridge() {
    munmap(stubs, StubSize());
}

void Bridge::Open(const std::string& name) {
    const auto path = directory / name;
    std::error_code error;
    const auto canonical = std::filesystem::canonical(path, error);
    if (error) throw std::runtime_error("missing library " + path.string());
    void* handle = dlopen(canonical.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) throw std::runtime_error(std::string("cannot load ") + canonical.string() + ": " + dlerror());
    if (std::find(handles.begin(), handles.end(), handle) != handles.end()) return;
    handles.push_back(handle);
    for (std::uint32_t i = 0; i < _dyld_image_count(); ++i) {
        const char* image = _dyld_get_image_name(i);
        std::error_code ignored;
        if (image != nullptr && std::filesystem::equivalent(image, canonical, ignored))
            images.push_back(reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(i)));
    }
}

void* Bridge::HostSymbol(const char* name) const {
    for (void* library : handles)
        if (void* symbol = dlsym(library, name)) return symbol;
    return nullptr;
}

void Bridge::Connect() {
    void* setUnwind = nullptr;
    for (void* library : handles)
        if (setUnwind == nullptr) setUnwind = dlsym(library, "Aps5SetBridgeUnwind_nid_no_patch");
    if (setUnwind == nullptr) throw std::runtime_error("libc has no unwind hooks");
    reinterpret_cast<void (*)(void (*)(std::uintptr_t*), void (*)(const std::uintptr_t*))>(setUnwind)(CaptureGuestFrame, ResumeGuestFrame);
    if (void* setControl = HostSymbol("Aps5SetBridgeControl_nid_no_patch"))
        reinterpret_cast<void (*)(void (*)(std::uint32_t*, std::uint16_t*))>(setControl)(GuestControl);
    if (void* setCall = HostSymbol("Aps5SetBridgeGuestCall_nid_no_patch"))
        reinterpret_cast<void (*)(std::uint64_t (*)(std::uint64_t, const std::uint64_t*))>(setCall)(CallGuestFunction);
    if (void* setFiber = HostSymbol("Aps5SetFiberBridge_nid_no_patch")) {
        using SetFiberBridge = void (*)(void (*)(std::uintptr_t*), void (*)(const std::uintptr_t*), void (*)(std::uint32_t*, std::uint16_t*),
                                        std::uint64_t);
        reinterpret_cast<SetFiberBridge>(setFiber)(CaptureGuestFrame, ResumeGuestFrame, GuestControl,
                                                   AddStub(reinterpret_cast<void*>(&FiberEntryReturned), "fiber entry return"));
    }
}

void Bridge::Register(const GuestImage& image, const std::string& name) {
    void* registerImage = nullptr;
    for (void* library : handles)
        if (registerImage == nullptr) registerImage = dlsym(library, "Aps5RegisterGuestImage_nid_no_patch");
    if (registerImage == nullptr) throw std::runtime_error("libc has no guest image hook");
    const ImageInfo info {image.Base(), imageNames.emplace_back(name).c_str(), image.ProgramHeaders(),
                          static_cast<std::uint16_t>(image.ProgramHeaderCount())};
    reinterpret_cast<void (*)(const ImageInfo*)>(registerImage)(&info);
}

void Bridge::SetTlsOffsets(std::vector<std::uint64_t> offsets) {
    tlsOffsets = std::move(offsets);
}

std::uint64_t Bridge::TlsGetAddrStub() {
    if (tlsGetAddrStub == 0) tlsGetAddrStub = AddStub(reinterpret_cast<void*>(&TlsGetAddr), "__tls_get_addr");
    return tlsGetAddrStub;
}

std::uint64_t Bridge::AddStub(void* address, const std::string& name, Variadic variadic, std::uint8_t fixed, bool x87Result) {
    if (FirstStubOffset + (functions.size() + 1) * StubBytes > StubSize()) throw std::runtime_error("too many imported functions");
    const auto number = static_cast<std::uint32_t>(FirstStub + functions.size());
    std::uint8_t* stub = stubs + FirstStubOffset + functions.size() * StubBytes;
    const auto target = static_cast<std::int32_t>(x87Result ? X87TrampolineOffset : TrampolineOffset);
    const auto jump = target - static_cast<std::int32_t>(stub + 10 - stubs);
    const std::uint8_t code[] = {MoveToEax, static_cast<std::uint8_t>(number), static_cast<std::uint8_t>(number >> 8), static_cast<std::uint8_t>(number >> 16),
        static_cast<std::uint8_t>(number >> 24), JumpRelative, static_cast<std::uint8_t>(jump), static_cast<std::uint8_t>(jump >> 8),
        static_cast<std::uint8_t>(jump >> 16), static_cast<std::uint8_t>(jump >> 24)};
    std::memset(stub, 0xcc, StubBytes);
    std::memcpy(stub, code, sizeof(code));
    functions.push_back({address, name, variadic, fixed, x87Result});
    return reinterpret_cast<std::uint64_t>(stub);
}

std::uint64_t Bridge::Resolve(const std::string& name, bool weak) {
    if (const auto found = resolved.find(name); found != resolved.end()) return found->second;
    for (void* handle : handles) {
        void* address = dlsym(handle, name.c_str());
        Dl_info info;
        if (address == nullptr || dladdr(address, &info) == 0) continue;
        const auto* image = static_cast<const mach_header_64*>(info.dli_fbase);
        if (std::find(images.begin(), images.end(), image) == images.end()) continue;
        unsigned long textSize = 0;
        const auto* text = getsectiondata(image, "__TEXT", "__text", &textSize);
        const auto* byte = static_cast<const std::uint8_t*>(address);
        const bool function = text != nullptr && byte >= text && byte < text + textSize;
        std::uint64_t guest = reinterpret_cast<std::uint64_t>(address);
        if (function) {
            Variadic variadic = Variadic::None;
            std::uint8_t fixed = 0;
            for (const auto& entry : VariadicExports) {
                if (info.dli_sname != nullptr && entry.name == info.dli_sname && info.dli_saddr == address) {
                    variadic = static_cast<Variadic>(entry.kind);
                    fixed = entry.fixed;
                }
            }
            if (variadic == Variadic::GuestList && setBridgeVaList == nullptr) {
                for (void* library : handles)
                    if (auto* setter = dlsym(library, "Aps5SetBridgeVaList_nid_no_patch")) setBridgeVaList = reinterpret_cast<void (*)(void*)>(setter);
                if (setBridgeVaList == nullptr) throw std::runtime_error("libc has no Aps5SetBridgeVaList_nid_no_patch");
            }
            const bool x87Result = info.dli_sname != nullptr && info.dli_saddr == address &&
                std::find(std::begin(X87Results), std::end(X87Results), info.dli_sname) != std::end(X87Results);
            guest = AddStub(address, name, variadic, fixed, x87Result);
        }
        resolved.emplace(name, guest);
        return guest;
    }
    if (weak) return 0;
    missing.push_back(name);
    const std::uint64_t guest = AddStub(nullptr, name);
    resolved.emplace(name, guest);
    return guest;
}

std::uint64_t Bridge::Call(FEXCore::Core::CPUState& state, std::uint64_t number, std::uint64_t block) {
    const std::uint64_t rip = state.rip;
    state.rip = rip + 2;
    if (number < FirstStub || number - FirstStub >= functions.size()) {
        std::fprintf(stderr, "[aps5-fex] guest syscall %llu at %#llx is not supported\n", static_cast<unsigned long long>(number),
                     static_cast<unsigned long long>(rip));
        return static_cast<std::uint64_t>(-ENOSYS);
    }
    const Function& function = functions[number - FirstStub];
    if (function.address == nullptr) {
        std::fprintf(stderr, "[aps5-fex] the guest called %s, which no library exports\n", function.name.c_str());
        std::abort();
    }
    auto* registers = reinterpret_cast<std::uint64_t*>(block);
    auto* guestStack = reinterpret_cast<std::uint64_t*>(block + BlockSize + 8);
    std::uint64_t slots[2 + StackArgumentSlots] {};
    std::memcpy(slots, guestStack, ReadableBytes(guestStack, sizeof(slots)));
    const std::uint64_t* stack = slots;
    if (trace) {
        std::fprintf(stderr, "[aps5-fex] %s(%#llx, %#llx, %#llx, %#llx, %#llx, %#llx)\n", function.name.c_str(),
                     static_cast<unsigned long long>(registers[0]), static_cast<unsigned long long>(registers[1]),
                     static_cast<unsigned long long>(registers[2]), static_cast<unsigned long long>(registers[3]),
                     static_cast<unsigned long long>(registers[4]), static_cast<unsigned long long>(registers[5]));
    }
    Aps5NativeArguments arguments {
        {registers[0], registers[1], registers[2], registers[3], registers[4], registers[5], stack[0], stack[1]},
        {},
        stack + 2,
        StackArgumentSlots,
    };
    std::memcpy(arguments.vector, reinterpret_cast<const void*>(block + BlockVectors), sizeof(arguments.vector));
    VaList list {static_cast<std::uint32_t>(8 * function.fixed), 48, guestStack, registers};
    std::uint64_t variadicSlots[6 + StackArgumentSlots];
    switch (function.variadic) {
    case Variadic::None: break;
    case Variadic::GuestList: setBridgeVaList(&list); break;
    case Variadic::Integers:
        std::memcpy(variadicSlots, registers + function.fixed, (6 - function.fixed) * sizeof(std::uint64_t));
        std::memcpy(variadicSlots + 6 - function.fixed, stack, StackArgumentSlots * sizeof(std::uint64_t));
        arguments.stack = variadicSlots;
        arguments.stackSlots = 6 - function.fixed + StackArgumentSlots;
        break;
    case Variadic::Unsupported:
        std::fprintf(stderr, "[aps5-fex] %s: its variadic arguments cannot be bridged\n", function.name.c_str());
        std::abort();
    }
    Aps5NativeResult result {};
    const GuestCall call {&state, block, currentCall};
    currentCall = &call;
    try {
        Aps5NativeCall(function.address, &arguments, &result);
    } catch (const GuestFrame& frame) {
        currentCall = call.outer;
        if (function.variadic == Variadic::GuestList) setBridgeVaList(nullptr);
        for (std::size_t i = 0; i < 16; ++i) state.gregs[DwarfRegisters[i]] = frame.registers[i];
        state.rip = frame.registers[16];
        return frame.registers[0];
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[aps5-fex] %s: %s\n", function.name.c_str(), error.what());
        std::abort();
    }
    currentCall = call.outer;
    if (function.variadic == Variadic::GuestList) setBridgeVaList(nullptr);
    registers[0] = result.gpr[0];
    registers[1] = result.gpr[1];
    std::memcpy(reinterpret_cast<void*>(block + BlockVectors), result.vector, sizeof(result.vector));
    return result.gpr[0];
}

}
