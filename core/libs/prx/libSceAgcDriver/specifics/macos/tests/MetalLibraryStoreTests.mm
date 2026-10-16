// agc_metal_library_store (DECISIONS D55): the persistent MSL library store (src/MetalLibraryStore.h)
// and the pipeline cache's startup prewarm.
// - The file format round-trips sources and macros; the key is the payload's content address.
// - Corrupt, truncated, misnamed and stale temporary files are deleted when the store is read, and
//   a store over its size bound keeps the most recently used files.
// - A library one cache compiled is recorded before its compile; a second cache over the same
//   directory compiles it on the prewarm thread, and the request that follows is a memory hit (no
//   compile on the requesting thread). A library whose compile fails leaves nothing behind.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/MetalPipelineCache.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace AgcDriver::Graphics::Metal;

int failures = 0;

void Expect(bool condition, const std::string& message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++failures;
}

std::vector<std::filesystem::path> Files(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) files.push_back(entry.path());
    return files;
}

void WriteText(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream << text;
}

StoredLibrary Sample(std::uint32_t index, std::size_t padding = 0) {
    StoredLibrary library;
    library.languageMajor = ShaderRecompiler::MslAbi::LanguageMajor;
    library.languageMinor = ShaderRecompiler::MslAbi::LanguageMinor;
    library.macros = {{"APS5_A", std::to_string(index)}, {"APS5_B", "float4"}};
    library.source = "#include <metal_stdlib>\nusing namespace metal;\n// " + std::string(padding, 'x') + "\nkernel void aps5_main(device uint* out [[buffer(0)]]) { out[0] = " + std::to_string(index) + "u + APS5_A; }\n";
    return library;
}

void FormatAndCorruption(const std::filesystem::path& root) {
    const auto library = Sample(7);
    const auto text = MetalLibraryStore::Serialize(library);
    const auto parsed = MetalLibraryStore::Parse(text);
    Expect(parsed.has_value() && *parsed == library, "a serialized library parses back to itself");
    Expect(MetalLibraryStore::Key(library) == MetalLibraryStore::Key(*parsed) && MetalLibraryStore::Key(library) != MetalLibraryStore::Key(Sample(8)), "keys address contents");
    Expect(!MetalLibraryStore::Parse(text.substr(0, text.size() - 1)).has_value(), "a truncated file is rejected");
    std::string flipped = text;
    flipped[flipped.size() - 3] ^= 0x01;
    Expect(!MetalLibraryStore::Parse(flipped).has_value(), "a file with a changed byte is rejected");
    Expect(!MetalLibraryStore::Parse("APS5MSL2" + text.substr(8)).has_value(), "another format version is rejected");

    const auto directory = root / "format";
    MetalLibraryStore store(directory, 1u << 20u);
    Expect(store.Enabled(), "the store creates its directory");
    store.Record(library);
    store.Record(library);
    Expect(Files(directory).size() == 1u, "a library is stored once");
    const auto key = MetalLibraryStore::Key(library);
    WriteText(directory / ("0123456789abcdef.msl"), flipped);
    WriteText(directory / ("fedcba9876543210.msl"), text);
    WriteText(directory / "cafecafecafecafe.msl", "");
    WriteText(directory / ".tmp-1-1", "partial");
    std::filesystem::last_write_time(directory / ".tmp-1-1", std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
    const auto loaded = store.Load();
    Expect(loaded.size() == 1u && loaded[0] == library, "only the intact, correctly named library loads");
    const auto left = Files(directory);
    Expect(left.size() == 1u && left[0].filename() == key + ".msl", "corrupt, misnamed, empty and stale temporary files are deleted");
    store.Forget(library);
    Expect(Files(directory).empty(), "Forget deletes the library");

    // Size bound: newest first, the oldest beyond the bound deleted.
    const auto bounded = root / "bounded";
    const std::size_t entryBytes = MetalLibraryStore::Serialize(Sample(0, 1000)).size();
    MetalLibraryStore small(bounded, entryBytes * 3u + entryBytes / 2u);
    const auto now = std::filesystem::file_time_type::clock::now();
    for (std::uint32_t index = 0; index < 5u; ++index) {
        small.Record(Sample(index, 1000));
        std::filesystem::last_write_time(bounded / (MetalLibraryStore::Key(Sample(index, 1000)) + ".msl"), now - std::chrono::minutes(10 - index));
    }
    small.Record(Sample(1, 1000));  // used again: now the most recent
    const auto kept = small.Load();
    Expect(kept.size() == 3u, "the bound keeps three libraries (" + std::to_string(kept.size()) + ")");
    Expect(kept.size() == 3u && kept[0] == Sample(1, 1000) && kept[1] == Sample(4, 1000) && kept[2] == Sample(3, 1000), "the most recently used libraries are kept, newest first");
    Expect(Files(bounded).size() == 3u, "the others are deleted");
    MetalLibraryStore disabled({}, 1u << 20u);
    disabled.Record(library);
    Expect(!disabled.Enabled() && disabled.Load().empty(), "a store without a directory does nothing");
}

void Prewarm(id<MTLDevice> device, const std::filesystem::path& root) {
    const auto directory = root / "cache";
    const auto stored = Sample(3);
    const ShaderRecompiler::MslShader shader{stored.source, {}};
    {
        PipelineCache first(device, directory);
        Expect(first.Library(shader, stored.macros) != nil, "the first cache compiles the library");
        const auto statistics = first.Statistics();
        Expect(statistics.librariesCompiled == 1u && statistics.librariesStored == 0u, "the first cache compiled it itself");
        bool threw = false;
        try {
            static_cast<void>(first.Library(ShaderRecompiler::MslShader{"kernel void aps5_main( { not MSL", {}}));
        } catch (const std::runtime_error&) {
            threw = true;
        }
        Expect(threw, "a library that does not compile throws");
    }
    PipelineCache::ForgetSharedLibraries();  // a new process
    const auto libraries = directory / "libraries";
    Expect(std::filesystem::is_directory(libraries) && Files(libraries).size() == 1u, "the store holds the compiled library and not the failed one");
    {
        PipelineCache second(device, directory);
        second.WaitForPrewarm();
        auto statistics = second.Statistics();
        Expect(statistics.librariesStored == 1u && statistics.librariesPrewarmed == 1u, "the second cache prewarms the stored library (" + std::to_string(statistics.librariesPrewarmed) + ")");
        Expect(second.Library(shader, stored.macros) != nil, "the second cache serves the library");
        statistics = second.Statistics();
        Expect(statistics.librariesCompiled == 0u && statistics.libraryHits == 1u, "the request is a memory hit, no compile on the requesting thread");
    }
    // A request racing the prewarm waits for it instead of compiling the library a second time.
    PipelineCache::ForgetSharedLibraries();
    {
        PipelineCache third(device, directory);
        Expect(third.Library(shader, stored.macros) != nil, "a request during the prewarm is served");
        third.WaitForPrewarm();
        const auto statistics = third.Statistics();
        Expect(statistics.librariesCompiled + statistics.librariesPrewarmed == 1u, "the library compiled once (" + std::to_string(statistics.librariesCompiled) + " + " + std::to_string(statistics.librariesPrewarmed) + ")");
    }
    // Two caches on one device (a replaced device and its replacement): one prewarm, one compile.
    PipelineCache::ForgetSharedLibraries();
    {
        PipelineCache replaced(device, directory);
        PipelineCache replacing(device, directory);
        replaced.WaitForPrewarm();
        Expect(replacing.Library(shader, stored.macros) != nil, "the replacing device's cache serves the library");
        const auto a = replaced.Statistics();
        const auto b = replacing.Statistics();
        Expect(a.librariesPrewarmed + b.librariesPrewarmed + a.librariesCompiled + b.librariesCompiled == 1u, "two caches on one device compile a library once");
    }
    PipelineCache::ForgetSharedLibraries();
    setenv("APS5_METAL_LIBRARY_PREWARM", "0", 1);
    {
        PipelineCache off(device, directory);
        off.WaitForPrewarm();
        Expect(off.Statistics().librariesPrewarmed == 0u, "APS5_METAL_LIBRARY_PREWARM=0 compiles nothing ahead of use");
    }
    unsetenv("APS5_METAL_LIBRARY_PREWARM");
}

}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::printf("skipped, the machine has no Metal 4 device\n");
            return 77;
        }
        char pattern[] = "/tmp/aps5-library-store-XXXXXX";
        const char* made = mkdtemp(pattern);
        if (made == nullptr) {
            std::fprintf(stderr, "cannot create a temporary directory\n");
            return 1;
        }
        const std::filesystem::path root(made);
        try {
            FormatAndCorruption(root);
            Prewarm(device, root);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            ++failures;
        }
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
        if (failures != 0) return 1;
        std::puts("library store tests passed");
        return 0;
    }
}
