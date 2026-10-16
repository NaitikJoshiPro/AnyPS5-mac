#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALLIBRARYSTORE_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALLIBRARYSTORE_H

// DECISIONS D55: the persistent MSL library store. macOS 27.0.1 has no runtime API that saves a
// library compiled from MSL source for reloading without the MSL front end (MTLLibrary has no
// serialization; MTLDynamicLibrary cannot hold stage functions; binary archives and MTL4Archive skip
// only the back end, and the offline `metal` compiler needs Xcode plus its separately downloaded
// Metal toolchain). The system Metal compiler cache keeps a library only once its compile finished in
// a live process, so a guest that gives up on a slow first-frame compile never gets it cached.
//
// The store keeps every library source the device asks for (with its macros and language version),
// content-addressed, before compiling it; the next process compiles them again on a background
// thread as soon as the device exists (PipelineCache prewarm), most recently used first, so a
// title's libraries are ready (or in the system cache) before its first frames ask for them.
//
// Files: <directory>/<16 hex digits>.msl, each a header line "APS5MSL1 <checksum> <payload bytes>"
// followed by the payload ("lang <major>.<minor>", one "macro <name>=<value>" line per macro, "--",
// the source). Writes go to a temporary file renamed into place; a file whose header, length or
// checksum does not match is deleted when read. The directory is bounded by size: least recently
// used files (modification time, refreshed on use) are deleted beyond the bound.
//
// Environment:
//   APS5_METAL_LIBRARY_CACHE=<dir>      store directory (default <pipeline cache directory>/libraries;
//                                       none when the pipeline cache has no directory)
//   APS5_NO_LIBRARY_CACHE=1             no store
//   APS5_METAL_LIBRARY_CACHE_MIB=<n>    size bound (default 256 MiB)
//   APS5_METAL_LIBRARY_PREWARM=0        keep the store but compile nothing ahead of use

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics::Metal {

struct StoredLibrary {
    std::uint32_t languageMajor = 0;
    std::uint32_t languageMinor = 0;
    std::vector<std::pair<std::string, std::string>> macros;
    std::string source;

    bool operator==(const StoredLibrary&) const = default;
};

class MetalLibraryStore {
public:
    // `directory` empty: a disabled store (every call is a no-op).
    MetalLibraryStore(std::filesystem::path directory, std::uint64_t byteLimit);

    [[nodiscard]] bool Enabled() const { return !directory.empty(); }
    [[nodiscard]] const std::filesystem::path& Directory() const { return directory; }

    // The content address of a library (FNV-1a 64 of its serialized payload).
    [[nodiscard]] static std::string Key(const StoredLibrary& library);
    // The file format (exposed for tests).
    [[nodiscard]] static std::string Serialize(const StoredLibrary& library);
    [[nodiscard]] static std::optional<StoredLibrary> Parse(const std::string& file);

    // Records a library (a no-op except for the access time when it is stored already). Never throws.
    void Record(const StoredLibrary& library);
    // Forgets a library (its compile failed). Never throws.
    void Forget(const StoredLibrary& library);
    // The stored libraries, most recently used first, within the size bound; unreadable or
    // corrupt files are deleted. Never throws.
    [[nodiscard]] std::vector<StoredLibrary> Load();
    // Deletes least recently used files until the directory is within the bound. Never throws.
    void Trim();

    // The default directory and bound from the environment (see above).
    [[nodiscard]] static std::filesystem::path DefaultDirectory(const std::filesystem::path& pipelineDirectory);
    [[nodiscard]] static std::uint64_t DefaultByteLimit();
    [[nodiscard]] static bool PrewarmEnabled();

private:
    std::filesystem::path directory;
    std::uint64_t byteLimit;
    std::mutex mutex;
    std::uint64_t recordedBytes = 0;
};

}

#endif
