#ifndef CORE_FEX_SRC_GUESTPROGRAM_HPP
#define CORE_FEX_SRC_GUESTPROGRAM_HPP

#include "GuestImage.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace Aps5Fex {

class Bridge;

struct GuestTlsLayout {
    struct Block {
        std::uint64_t templateAddress;
        std::size_t fileSize;
        std::uint64_t offset;
    };

    std::vector<Block> blocks;
    std::uint64_t size = 0;
    std::size_t alignment = 16;
};

class GuestProgram {
public:
    GuestProgram(const std::filesystem::path& executable, Bridge& bridge);

    const GuestImage& Executable() const { return *images.front(); }
    const std::vector<std::unique_ptr<GuestImage>>& Images() const { return images; }
    std::vector<std::uint64_t> Initializers() const;
    const GuestTlsLayout& Tls() const { return tls; }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> CodeRanges() const;

private:
    void Load(const std::filesystem::path& path, Bridge& bridge);

    std::vector<std::unique_ptr<GuestImage>> images;
    GuestTlsLayout tls;
};

}

#endif
