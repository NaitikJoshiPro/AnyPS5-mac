#ifndef CORE_FEX_SRC_BRIDGE_HPP
#define CORE_FEX_SRC_BRIDGE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct mach_header_64;

namespace FEXCore {
class ThunkHandler;
}

namespace FEXCore::Core {
struct CPUState;
struct CpuStateFrame;
}

namespace Aps5Fex {

class GuestImage;
struct BridgeSignature;

class Bridge {
public:
    explicit Bridge(std::filesystem::path libraries);
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;
    ~Bridge();

    void Open(const std::string& name);
    // Opens the library when the library directory has it.
    void OpenIfPresent(const std::string& name);
    void* HostSymbol(const char* name) const;

    void Connect();
    void Register(const GuestImage& image, const std::string& name);

    void SetTlsOffsets(std::vector<std::uint64_t> offsets);
    std::uint64_t TlsGetAddrStub();

    std::uint64_t Resolve(const std::string& name, bool weak);
    const std::vector<std::string>& Missing() const { return missing; }

    // What guest code gets for a native export: the x86 stub of a function (one per function), the
    // address itself for data, 0 when no PS5 library (*.prx) holds the address. Thread-safe.
    std::uint64_t StubFor(void* address, const std::string& name);

    std::uint64_t CallbackReturn() const { return StubBase(); }

    std::uint64_t StubBase() const { return reinterpret_cast<std::uint64_t>(stubs); }
    std::size_t StubSize() const { return StubCapacity * StubBytes; }
    bool OwnsStub(std::uint64_t address) const { return address >= StubBase() && address < StubBase() + StubSize(); }
    // The import whose stub holds `address` (a guest RIP inside the stub area), or "" (diagnostics).
    std::string StubName(std::uint64_t address);

    std::uint64_t Call(FEXCore::Core::CPUState& state, std::uint64_t number, std::uint64_t block);

    // The handler that names the fast trampolines' host function for FEXCore (Context::SetThunkHandler).
    static FEXCore::ThunkHandler* Thunks();
    void ThunkCall(FEXCore::Core::CpuStateFrame* frame, std::uint64_t block);

private:
    enum class Variadic : std::uint8_t {
        None,
        GuestList,
        Integers,
        Unsupported,
    };

    struct Function {
        void* address;
        std::string name;
        Variadic variadic;
        std::uint8_t fixed;
        bool x87Result;
        const BridgeSignature* signature = nullptr;
        bool shaped = false;
        bool callbacks = false;
        bool callbackResult = false;
        // Guest stack slots and whether vector registers carry arguments (all of them without a signature).
        std::uint8_t stackSlots = 10;
        bool vectors = true;
    };

    std::uint64_t NativeCallable(std::uint64_t value, const BridgeSignature* callback);
    std::uint64_t GuestCallable(std::uint64_t value);

    static bool IsShaped(const BridgeSignature* signature);
    static bool GuestX87Result(const BridgeSignature* signature);
    bool Invoke(FEXCore::Core::CPUState& state, const Function& function, std::uint64_t block);

    static constexpr std::size_t StubBytes = 16;
    static constexpr std::size_t StubCapacity = 0x10000;
    static constexpr std::size_t TrampolineOffset = 0x10;
    static constexpr std::size_t X87TrampolineOffset = 0x100;
    static constexpr std::size_t MakecontextOffset = 0x200;
    static constexpr std::size_t FirstStubOffset = 0x280;
    static constexpr std::uint32_t FirstStub = 0x41500000;

    std::uint64_t AddStub(void* address, const std::string& name, Variadic variadic = Variadic::None, std::uint8_t fixed = 0,
                          bool x87Result = false, const BridgeSignature* signature = nullptr);

    std::filesystem::path directory;
    std::vector<void*> handles;
    std::vector<const mach_header_64*> images;
    std::vector<Function> functions;
    std::unordered_map<std::string, std::uint64_t> resolved;
    std::vector<std::string> missing;
    std::deque<std::string> imageNames;
    std::uint64_t tlsGetAddrStub = 0;
    std::unordered_map<const void*, std::uint64_t> guestAddresses;
    std::recursive_mutex stubMutex;
    std::atomic<std::size_t> stubCount {0};
    std::uint8_t* stubs = nullptr;
    void (*setBridgeVaList)(void*) = nullptr;
    bool trace = false;
};

}

#endif
