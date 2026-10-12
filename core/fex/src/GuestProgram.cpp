#include "GuestProgram.hpp"

#include "Bridge.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace Aps5Fex {

namespace {

constexpr std::string_view OriginPrefix = "$ORIGIN/";
constexpr std::string_view GuestSuffix = "#guest";
constexpr std::string_view DynamicLinker = "ld-linux-x86-64.so.2";

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

const GuestProgram* loadedProgram = nullptr;

void* OpenGuestModule(const char* path) {
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    if (error) return nullptr;
    for (const auto& image : loadedProgram->Images())
        if (image->Path() == canonical) return image.get();
    return nullptr;
}

void* GuestModuleSymbol(void* image, const char* name) {
    const auto symbol = static_cast<const GuestImage*>(image)->Export(name);
    return symbol && !symbol->tls ? reinterpret_cast<void*>(symbol->address) : nullptr;
}

}

GuestProgram::GuestProgram(const std::filesystem::path& executable, Bridge& bridge) {
    bridge.Open("libkernel.prx");
    bridge.Open("libc.prx");
    Load(std::filesystem::absolute(executable), bridge);

    std::vector<std::uint64_t> offsets {0};
    std::uint64_t offset = 0;
    for (std::size_t index = 0; index < images.size(); ++index) {
        const GuestTlsTemplate& image = images[index]->Tls();
        if (image.memorySize != 0) {
            offset = AlignUp(offset + image.memorySize, image.alignment);
            tls.blocks.push_back({images[index]->Base() + image.address, image.fileSize, offset});
            tls.alignment = std::max(tls.alignment, image.alignment);
        }
        images[index]->SetTls(index + 1, image.memorySize != 0 ? offset : 0);
        offsets.push_back(image.memorySize != 0 ? offset : 0);
    }
    tls.size = offset;
    bridge.SetTlsOffsets(offsets);

    for (const auto& image : images) {
        image->Relocate([&](const std::string& name, bool weak) -> std::optional<GuestSymbol> {
            if (name == "__tls_get_addr") return GuestSymbol {bridge.TlsGetAddrStub()};
            if (name.ends_with(GuestSuffix)) {
                for (const auto& provider : images)
                    if (const auto symbol = provider->Export(name)) return symbol;
                return std::nullopt;
            }
            const std::uint64_t address = bridge.Resolve(name, weak);
            if (address == 0) return std::nullopt;
            return GuestSymbol {address};
        });
    }
    bridge.Connect();
    if (auto* setLoader = bridge.HostSymbol("Aps5SetGuestLoader_nid_no_patch")) {
        loadedProgram = this;
        reinterpret_cast<void (*)(void* (*)(const char*), void* (*)(void*, const char*))>(setLoader)(OpenGuestModule, GuestModuleSymbol);
    }
    for (std::size_t index = 0; index < images.size(); ++index)
        bridge.Register(*images[index], index == 0 ? std::string() : images[index]->Path().string());
}

void GuestProgram::Load(const std::filesystem::path& path, Bridge& bridge) {
    const auto canonical = std::filesystem::weakly_canonical(path);
    for (const auto& image : images)
        if (image->Path() == canonical) return;
    images.push_back(std::make_unique<GuestImage>(canonical));
    const std::vector<std::string> needed = images.back()->Needed();
    for (const auto& name : needed) {
        if (name.starts_with(OriginPrefix)) Load(canonical.parent_path() / name.substr(OriginPrefix.size()), bridge);
        else if (name != DynamicLinker) bridge.Open(name);
    }
}

std::vector<std::uint64_t> GuestProgram::Initializers() const {
    std::vector<std::uint64_t> functions;
    for (std::size_t index = 1; index < images.size(); ++index) {
        const auto initializers = images[index]->Initializers();
        functions.insert(functions.end(), initializers.begin(), initializers.end());
    }
    return functions;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> GuestProgram::CodeRanges() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& image : images) ranges.emplace_back(image->Start(), image->Start() + image->Size());
    return ranges;
}

}
