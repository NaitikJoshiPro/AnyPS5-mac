#ifndef CORE_FEX_SRC_BRIDGE_HPP
#define CORE_FEX_SRC_BRIDGE_HPP

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

struct mach_header_64;

namespace FEXCore::Core {
struct CPUState;
}

namespace Aps5Fex {

class GuestImage;

class Bridge {
public:
    explicit Bridge(std::filesystem::path libraries);
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;
    ~Bridge();

    void Open(const std::string& name);
    void* HostSymbol(const char* name) const;

    void Connect();
    void Register(const GuestImage& image, const std::string& name);

    void SetTlsOffsets(std::vector<std::uint64_t> offsets);
    std::uint64_t TlsGetAddrStub();

    std::uint64_t Resolve(const std::string& name, bool weak);
    const std::vector<std::string>& Missing() const { return missing; }

    std::uint64_t CallbackReturn() const { return StubBase(); }

    std::uint64_t StubBase() const { return reinterpret_cast<std::uint64_t>(stubs); }
    std::size_t StubSize() const { return StubCapacity * StubBytes; }
    bool OwnsStub(std::uint64_t address) const { return address >= StubBase() && address < StubBase() + StubSize(); }

    std::uint64_t Call(FEXCore::Core::CPUState& state, std::uint64_t number, std::uint64_t block);

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
    };

    static constexpr std::size_t StubBytes = 16;
    static constexpr std::size_t StubCapacity = 0x10000;
    static constexpr std::size_t TrampolineOffset = 0x10;
    static constexpr std::size_t X87TrampolineOffset = 0x100;
    static constexpr std::size_t FirstStubOffset = 0x200;
    static constexpr std::uint32_t FirstStub = 0x41500000;

    std::uint64_t AddStub(void* address, const std::string& name, Variadic variadic = Variadic::None, std::uint8_t fixed = 0,
                          bool x87Result = false);

    std::filesystem::path directory;
    std::vector<void*> handles;
    std::vector<const mach_header_64*> images;
    std::vector<Function> functions;
    std::unordered_map<std::string, std::uint64_t> resolved;
    std::vector<std::string> missing;
    std::deque<std::string> imageNames;
    std::uint64_t tlsGetAddrStub = 0;
    std::uint8_t* stubs = nullptr;
    void (*setBridgeVaList)(void*) = nullptr;
    bool trace = false;
};

}

#endif
