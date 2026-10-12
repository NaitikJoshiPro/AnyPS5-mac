#ifndef CORE_FEX_SRC_GUESTIMAGE_HPP
#define CORE_FEX_SRC_GUESTIMAGE_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace Aps5Fex {

struct GuestTlsTemplate {
    std::uint64_t address = 0;
    std::size_t fileSize = 0;
    std::size_t memorySize = 0;
    std::size_t alignment = 1;
};

struct GuestSymbol {
    std::uint64_t address = 0;
    bool tls = false;
    std::uint64_t tlsModule = 0;
    std::uint64_t tlsValue = 0;
    std::uint64_t tlsOffset = 0;
};

class GuestImage {
public:
    using Resolver = std::function<std::optional<GuestSymbol>(const std::string& name, bool weak)>;

    explicit GuestImage(const std::filesystem::path& path);
    GuestImage(const GuestImage&) = delete;
    GuestImage& operator=(const GuestImage&) = delete;
    ~GuestImage();

    const std::filesystem::path& Path() const { return path; }
    std::uint64_t Base() const { return base; }
    std::uint64_t Start() const { return start; }
    std::size_t Size() const { return size; }
    std::uint64_t Entry() const { return entry; }
    std::optional<std::uint64_t> StartFunction() const;
    const std::vector<std::string>& Needed() const { return needed; }
    const GuestTlsTemplate& Tls() const { return tls; }
    const void* ProgramHeaders() const { return programHeaders.data(); }
    std::size_t ProgramHeaderCount() const { return programHeaders.size() / ProgramHeaderSize; }

    static constexpr std::size_t ProgramHeaderSize = 56;

    void SetTls(std::uint64_t module, std::uint64_t offset) {
        tlsModule = module;
        tlsOffset = offset;
    }

    std::optional<GuestSymbol> Export(const std::string& name) const;

    void Relocate(const Resolver& resolve);

    std::vector<std::uint64_t> Initializers() const;

private:
    struct Exported {
        std::uint64_t value;
        bool tls;
    };

    std::filesystem::path path;
    std::uint64_t base = 0;
    std::uint64_t start = 0;
    std::size_t size = 0;
    std::uint64_t entry = 0;
    std::uint64_t dynamic = 0;
    std::vector<std::string> needed;
    std::vector<unsigned char> programHeaders;
    std::unordered_map<std::string, Exported> exports;
    GuestTlsTemplate tls;
    std::uint64_t tlsModule = 0;
    std::uint64_t tlsOffset = 0;
};

}

#endif
