// DECISIONS D55: the persistent MSL library store (see MetalLibraryStore.h).

#include "MetalLibraryStore.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>
#include <unistd.h>

namespace AgcDriver::Graphics::Metal {

namespace {

constexpr const char* Magic = "APS5MSL1";
constexpr const char* Extension = ".msl";

std::uint64_t Fnv1a(const std::string& text) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const char value : text) hash = (hash ^ static_cast<std::uint8_t>(value)) * 0x100000001b3ull;
    return hash;
}

std::string Hex(std::uint64_t value) {
    char text[20];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
    return text;
}

std::string Payload(const StoredLibrary& library) {
    std::string payload = "lang " + std::to_string(library.languageMajor) + "." + std::to_string(library.languageMinor) + "\n";
    for (const auto& [name, value] : library.macros) payload += "macro " + name + "=" + value + "\n";
    payload += "--\n";
    payload += library.source;
    return payload;
}

bool ReadFile(const std::filesystem::path& path, std::string& text) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    text = buffer.str();
    return static_cast<bool>(stream) || stream.eof();
}

void Touch(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), error);
}

struct StoredFile {
    std::filesystem::path path;
    std::filesystem::file_time_type time;
    std::uintmax_t bytes = 0;
};

std::vector<StoredFile> List(const std::filesystem::path& directory) {
    std::vector<StoredFile> files;
    std::error_code error;
    for (std::filesystem::directory_iterator entry(directory, error), end; !error && entry != end; entry.increment(error)) {
        const auto& path = entry->path();
        std::error_code status;
        if (!entry->is_regular_file(status)) continue;
        const auto name = path.filename().string();
        // Temporary files of an interrupted write.
        if (name.starts_with(".tmp-")) {
            const auto time = std::filesystem::last_write_time(path, status);
            if (!status && std::filesystem::file_time_type::clock::now() - time > std::chrono::minutes(10)) std::filesystem::remove(path, status);
            continue;
        }
        if (path.extension() != Extension) continue;
        StoredFile file;
        file.path = path;
        file.time = std::filesystem::last_write_time(path, status);
        if (status) continue;
        file.bytes = std::filesystem::file_size(path, status);
        if (status) continue;
        files.push_back(std::move(file));
    }
    std::sort(files.begin(), files.end(), [](const StoredFile& left, const StoredFile& right) { return left.time != right.time ? left.time > right.time : left.path < right.path; });
    return files;
}

}

MetalLibraryStore::MetalLibraryStore(std::filesystem::path directory, std::uint64_t byteLimit) : directory(std::move(directory)), byteLimit(byteLimit) {
    if (this->directory.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(this->directory, error);
    if (error || !std::filesystem::is_directory(this->directory, error)) {
        std::fprintf(stderr, "[metal-libraries] no library store: cannot create %s\n", this->directory.c_str());
        this->directory.clear();
    }
}

std::string MetalLibraryStore::Key(const StoredLibrary& library) {
    return Hex(Fnv1a(Payload(library)));
}

std::string MetalLibraryStore::Serialize(const StoredLibrary& library) {
    const auto payload = Payload(library);
    return std::string(Magic) + " " + Hex(Fnv1a(payload)) + " " + std::to_string(payload.size()) + "\n" + payload;
}

std::optional<StoredLibrary> MetalLibraryStore::Parse(const std::string& file) {
    const auto newline = file.find('\n');
    if (newline == std::string::npos) return std::nullopt;
    std::istringstream header(file.substr(0, newline));
    std::string magic;
    std::string checksum;
    std::size_t bytes = 0;
    if (!(header >> magic >> checksum >> bytes) || magic != Magic) return std::nullopt;
    const std::string payload = file.substr(newline + 1u);
    if (payload.size() != bytes || Hex(Fnv1a(payload)) != checksum) return std::nullopt;
    StoredLibrary library;
    std::size_t position = 0;
    auto line = [&](std::string& text) {
        const auto end = payload.find('\n', position);
        if (end == std::string::npos) return false;
        text = payload.substr(position, end - position);
        position = end + 1u;
        return true;
    };
    std::string text;
    if (!line(text) || !text.starts_with("lang ")) return std::nullopt;
    const auto dot = text.find('.');
    if (dot == std::string::npos) return std::nullopt;
    try {
        library.languageMajor = static_cast<std::uint32_t>(std::stoul(text.substr(5u, dot - 5u)));
        library.languageMinor = static_cast<std::uint32_t>(std::stoul(text.substr(dot + 1u)));
    } catch (const std::exception&) {
        return std::nullopt;
    }
    while (true) {
        if (!line(text)) return std::nullopt;
        if (text == "--") break;
        if (!text.starts_with("macro ")) return std::nullopt;
        const auto equals = text.find('=');
        if (equals == std::string::npos) return std::nullopt;
        library.macros.emplace_back(text.substr(6u, equals - 6u), text.substr(equals + 1u));
    }
    library.source = payload.substr(position);
    return library;
}

void MetalLibraryStore::Record(const StoredLibrary& library) {
    if (!Enabled()) return;
    try {
        const auto path = directory / (Key(library) + Extension);
        std::error_code error;
        if (std::filesystem::exists(path, error)) {
            Touch(path);
            return;
        }
        static std::atomic<std::uint64_t> sequence{0};
        const auto temporary = directory / (".tmp-" + std::to_string(::getpid()) + "-" + std::to_string(sequence.fetch_add(1)));
        const auto text = Serialize(library);
        {
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            stream.write(text.data(), static_cast<std::streamsize>(text.size()));
            stream.flush();
            if (!stream) {
                std::filesystem::remove(temporary, error);
                return;
            }
        }
        std::filesystem::rename(temporary, path, error);
        if (error) {
            std::filesystem::remove(temporary, error);
            return;
        }
        bool trim = false;
        {
            std::lock_guard lock(mutex);
            recordedBytes += text.size();
            // Trim once the files added by this process could have pushed the directory past the bound.
            if (recordedBytes > byteLimit / 8u) {
                recordedBytes = 0;
                trim = true;
            }
        }
        if (trim) Trim();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[metal-libraries] cannot record a library: %s\n", error.what());
    }
}

void MetalLibraryStore::Forget(const StoredLibrary& library) {
    if (!Enabled()) return;
    std::error_code error;
    std::filesystem::remove(directory / (Key(library) + Extension), error);
}

std::vector<StoredLibrary> MetalLibraryStore::Load() {
    std::vector<StoredLibrary> libraries;
    if (!Enabled()) return libraries;
    try {
        std::uintmax_t total = 0;
        for (const auto& file : List(directory)) {
            std::error_code error;
            if (total + file.bytes > byteLimit) {
                std::filesystem::remove(file.path, error);
                continue;
            }
            std::string text;
            std::optional<StoredLibrary> library;
            if (ReadFile(file.path, text)) library = Parse(text);
            // A file named for other contents is as corrupt as one that does not parse.
            if (!library || file.path.stem().string() != Key(*library)) {
                std::fprintf(stderr, "[metal-libraries] deleting the unreadable library %s\n", file.path.filename().c_str());
                std::filesystem::remove(file.path, error);
                continue;
            }
            total += file.bytes;
            libraries.push_back(std::move(*library));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[metal-libraries] cannot read the library store: %s\n", error.what());
    }
    return libraries;
}

void MetalLibraryStore::Trim() {
    if (!Enabled()) return;
    try {
        std::uintmax_t total = 0;
        for (const auto& file : List(directory)) {
            total += file.bytes;
            if (total <= byteLimit) continue;
            std::error_code error;
            std::filesystem::remove(file.path, error);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[metal-libraries] cannot trim the library store: %s\n", error.what());
    }
}

std::filesystem::path MetalLibraryStore::DefaultDirectory(const std::filesystem::path& pipelineDirectory) {
    if (const char* disabled = std::getenv("APS5_NO_LIBRARY_CACHE"); disabled != nullptr && *disabled != '\0' && std::string(disabled) != "0") return {};
    if (const char* directory = std::getenv("APS5_METAL_LIBRARY_CACHE"); directory != nullptr && *directory != '\0') return directory;
    if (pipelineDirectory.empty()) return {};
    return pipelineDirectory / "libraries";
}

std::uint64_t MetalLibraryStore::DefaultByteLimit() {
    std::uint64_t mib = 256u;
    if (const char* value = std::getenv("APS5_METAL_LIBRARY_CACHE_MIB"); value != nullptr && *value != '\0') {
        try {
            mib = std::stoull(value);
        } catch (const std::exception&) {
            std::fprintf(stderr, "[metal-libraries] ignoring APS5_METAL_LIBRARY_CACHE_MIB=%s\n", value);
        }
    }
    return mib * 1024u * 1024u;
}

bool MetalLibraryStore::PrewarmEnabled() {
    const char* value = std::getenv("APS5_METAL_LIBRARY_PREWARM");
    return value == nullptr || std::string(value) != "0";
}

}
