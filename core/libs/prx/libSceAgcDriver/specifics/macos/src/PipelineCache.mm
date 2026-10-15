// Work package D5: Metal twin of Graphics/src/PipelineCache.cpp (see MetalPipelineCache.h).
#include "MetalPipelineCache.h"
#include "ShaderCacheDirectory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"

#include <sys/sysctl.h>
#include <unistd.h>
#include <pthread.h>
#include <condition_variable>
#include <algorithm>
#include <atomic>
#include <map>
#include <unordered_set>
#include <memory>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <system_error>

namespace AgcDriver::Graphics::Metal {

namespace {

std::string Describe(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : std::string("unknown error");
}

// Reports a Metal compile still running after 5, 10, 15... s on stderr, as the recorder reports a
// slow GPU batch. MSL compiles are synchronous and unbounded, and under MTL_SHADER_VALIDATION a
// compile of a few seconds can grow past the GPU suite's 30 s cap when the compiler service is
// busy: without this line such a timeout looks like a GPU hang (ATOM, 2026-10-13).
class SlowCompileReport {
public:
    SlowCompileReport(const char* what, std::size_t sourceBytes) : state(std::make_shared<State>()) {
        state->what = what;
        state->bytes = sourceBytes;
        Arm(state, 5);
    }
    ~SlowCompileReport() { state->finished.store(true, std::memory_order_release); }
    SlowCompileReport(const SlowCompileReport&) = delete;
    SlowCompileReport& operator=(const SlowCompileReport&) = delete;

private:
    struct State {
        std::atomic<bool> finished{false};
        const char* what = "";
        std::size_t bytes = 0;
    };
    static void Arm(std::shared_ptr<State> state, unsigned seconds) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(seconds) * static_cast<int64_t>(NSEC_PER_SEC)), dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            if (state->finished.load(std::memory_order_acquire)) return;
            if (state->bytes != 0) std::fprintf(stderr, "[metal-pipelines] %s still compiling after %u s (%zu bytes of MSL)\n", state->what, seconds, state->bytes);
            else std::fprintf(stderr, "[metal-pipelines] %s still compiling after %u s\n", state->what, seconds);
            Arm(state, seconds + 5);
        });
    }
    std::shared_ptr<State> state;
};

bool Profile() {
    static const bool enabled = std::getenv("APS5_PROFILE_PIPELINES") != nullptr;
    return enabled;
}

double Milliseconds(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

std::string OsBuild() {
    char build[64] = {};
    std::size_t size = sizeof(build);
    if (sysctlbyname("kern.osversion", build, &size, nullptr, 0) != 0) return "unknown";
    return build;
}

std::string FileSafe(std::string text) {
    for (auto& character : text) {
        const bool safe = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-' || character == '.';
        if (!safe) character = '_';
    }
    return text;
}

// Archives serialized while Metal shader validation instruments the pipelines do not load again
// ("deployment target for architecture applegpu_g15g found in archive is not compatible with
// current OS", macOS 27.0.1), and archive lookups would hand out uninstrumented pipelines: no
// archive under MTL_SHADER_VALIDATION.
bool ShaderValidation() {
    const char* value = std::getenv("MTL_SHADER_VALIDATION");
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

std::filesystem::path DefaultDirectory() {
    if (std::getenv("APS5_NO_PIPELINE_CACHE") != nullptr) return {};
    if (const char* directory = std::getenv("APS5_METAL_PIPELINE_CACHE"); directory != nullptr && *directory != '\0') return directory;
    const auto shaders = ShaderRecompiler::ShaderCacheDirectory();
    if (shaders.empty()) return {};
    return shaders / "metal";
}

void AppendColor(std::string& key, const ColorAttachmentState& color) {
    key += std::to_string(color.format);
    if (color.format == MTLPixelFormatInvalid) {
        key += ';';
        return;
    }
    key += ',' + std::to_string(color.writeMask);
    if (color.blending) {
        for (const auto value : {static_cast<std::uint64_t>(color.sourceRgb), static_cast<std::uint64_t>(color.destinationRgb), static_cast<std::uint64_t>(color.rgbOperation), static_cast<std::uint64_t>(color.sourceAlpha), static_cast<std::uint64_t>(color.destinationAlpha), static_cast<std::uint64_t>(color.alphaOperation)}) {
            key += ',' + std::to_string(value);
        }
    }
    key += ';';
}

void AppendFunction(std::string& key, const PipelineFunction& function) {
    if (function.shader == nullptr) throw std::runtime_error("Metal pipeline cache (work package D5): a pipeline function has no MSL shader");
    key += std::to_string(function.shader->source.size());
    key += ':';
    key += function.shader->source;
    for (const auto& [name, value] : function.macros) key += "|#" + name + "=" + value;
    for (const auto& [id, value] : function.constants) key += "|" + std::to_string(id) + "=" + std::to_string(value);
    key += '\n';
}

}

struct PipelineCache::LibraryTask {
    std::mutex mutex;
    std::condition_variable done;
    bool finished = false;
    id<MTLLibrary> library = nil;
    std::string failure;
};

namespace {

// Libraries compiled or compiling for an MTLDevice, shared by every PipelineCache on it (a replaced
// windowless device and its windowed replacement, D43): neither compiles a library the other has,
// and one prewarm per MTLDevice. Never destroyed (exit handlers tear devices down after static
// destructors).
struct SharedLibraries {
    std::mutex mutex;
    std::unordered_map<std::string, id<MTLLibrary>> libraries;
    std::unordered_map<std::string, std::shared_ptr<void>> tasks;
    std::unordered_set<std::uint64_t> prewarmed;
};

SharedLibraries& Shared() {
    static auto* shared = new SharedLibraries;
    return *shared;
}

// The archive generations the first cache of this process chose for an archive stem (upstream
// 8c8cd86cf in Metal idiom). Every later cache of the process (the windowed device replacing the
// windowless one, compile-only devices) loads the same files and deletes none: a generation an
// earlier cache of this process wrote, or loaded and still resolves lookups through, must stay.
// Never destroyed, as SharedLibraries.
struct ArchiveChoices {
    std::mutex mutex;
    std::map<std::filesystem::path, std::vector<std::filesystem::path>> chosen;
};

ArchiveChoices& Choices() {
    static auto* choices = new ArchiveChoices;
    return *choices;
}

// The process id of a generation file "<stem>.<20-digit ns>-<pid>.mtl4archive" (empty: the legacy
// "<stem>.mtl4archive" or a foreign name).
std::string GenerationProcess(const std::filesystem::path& path, const std::string& stem) {
    const auto name = path.filename().string();
    constexpr std::size_t Stamp = 20, Suffix = 12;
    if (name.size() <= stem.size() + 1u + Stamp + 1u + Suffix) return {};
    const auto rest = name.substr(stem.size() + 1u);
    if (rest[Stamp] != '-') return {};
    return rest.substr(Stamp + 1u, rest.size() - Stamp - 1u - Suffix);
}

}

struct PipelineCache::Entry {
    std::mutex mutex;
    std::condition_variable done;
    bool ready = false;
    bool running = false;
    std::string failure;
    id<MTLComputePipelineState> compute = nil;
    id<MTLRenderPipelineState> render = nil;
};

PipelineFunction PipelineFunction::From(const ShaderRecompiler::RecompileResult& result) {
    PipelineFunction function;
    function.shader = std::make_shared<const ShaderRecompiler::MslShader>(result.msl.Shader());
    function.constants.reserve(result.specialization.size());
    for (const auto& constant : result.specialization) function.constants.emplace_back(constant.id, constant.value);
    return function;
}

PipelineCache::PipelineCache(id<MTLDevice> device, std::filesystem::path directory) : device(device) {
    if (device == nil) throw std::runtime_error("Metal pipeline cache (work package D5): no Metal device");
    if (directory.empty()) directory = DefaultDirectory();
    // The library store keeps sources, which shader validation does not change (it instruments
    // pipelines), so it is used under MTL_SHADER_VALIDATION too.
    const auto storeDirectory = MetalLibraryStore::DefaultDirectory(directory);
    if (ShaderValidation()) {
        if (!directory.empty() && Profile()) std::fprintf(stderr, "[metal-pipelines] no archive under MTL_SHADER_VALIDATION\n");
        directory.clear();
        statistics.archiveDisabled = "MTL_SHADER_VALIDATION";
    }
    if (Profile()) std::fprintf(stderr, "[metal-pipelines] archive directory: %s\n", directory.empty() ? "(none)" : directory.c_str());
    if (!directory.empty()) {
        // Archive generations (D51): <stem>.<ns>-<pid>.mtl4archive. The newest readable one is
        // loaded and never written by this process (Metal resolves a lookup archive through its
        // file, so the file a hit came from must stay as it was); this process writes its own
        // generation, a superset of its hits and its compiles, which the next process loads. Older
        // generations (and the single file of earlier builds) are deleted once a newer one loads.
        const std::string stem = FileSafe(std::string(device.name.UTF8String) + "-" + OsBuild());
        std::error_code error;
        std::vector<std::filesystem::path> generations;
        for (std::filesystem::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
            const auto name = it->path().filename().string();
            if (name.rfind(stem + ".", 0) == 0 && name.size() > 12 && name.compare(name.size() - 12, 12, ".mtl4archive") == 0) generations.push_back(it->path());
        }
        // Names sort by creation time (fixed-width nanoseconds; the legacy "<stem>.mtl4archive" first).
        const auto legacy = directory / (stem + ".mtl4archive");
        std::sort(generations.begin(), generations.end(), [&](const std::filesystem::path& a, const std::filesystem::path& b) {
            if ((a == legacy) != (b == legacy)) return a == legacy;
            return a < b;
        });
        // Merge, not overwrite (upstream 8c8cd86cf in Metal idiom): a process holds several caches
        // (the windowless device next to its windowed replacement), each writing its own generation,
        // so the newest generation alone would drop the other devices' pipelines. The first cache of
        // the process loads the newest readable generation together with the generations the same
        // process wrote (same pid suffix) and deletes the older rest; later caches of the process
        // load the generations this process has written so far and the same chosen set, and delete
        // nothing. Every written generation is a superset of its hits from all loaded generations.
        NSMutableArray<id<MTL4Archive>>* opened = [NSMutableArray array];
        auto openGeneration = [&](const std::filesystem::path& path) {
            NSError* failure = nil;
            id<MTL4Archive> generation = [device newArchiveWithURL:[NSURL fileURLWithPath:@(path.c_str())] error:&failure];
            if (generation == nil) {
                std::fprintf(stderr, "[metal-pipelines] ignoring the unreadable archive %s: %s\n", path.c_str(), Describe(failure).c_str());
                return false;
            }
            [opened addObject:generation];
            return true;
        };
        auto& choices = Choices();
        std::lock_guard choiceLock(choices.mutex);
        const auto key = directory / stem;
        if (const auto earlier = choices.chosen.find(key); earlier != choices.chosen.end()) {
            const auto self = std::to_string(::getpid());
            for (std::size_t index = generations.size(); index-- > 0;) {
                if (GenerationProcess(generations[index], stem) == self && std::find(earlier->second.begin(), earlier->second.end(), generations[index]) == earlier->second.end()) openGeneration(generations[index]);
            }
            for (const auto& path : earlier->second) openGeneration(path);
        } else {
            std::vector<std::filesystem::path> chosen;
            std::size_t newest = generations.size();
            std::string process;
            for (std::size_t index = generations.size(); index-- > 0;) {
                if (!chosen.empty() && (process.empty() || GenerationProcess(generations[index], stem) != process)) continue;
                if (!openGeneration(generations[index])) continue;
                if (chosen.empty()) {
                    newest = index;
                    process = GenerationProcess(generations[index], stem);
                }
                chosen.push_back(generations[index]);
            }
            for (std::size_t index = 0; index < newest && index < generations.size(); ++index) {
                if (std::find(chosen.begin(), chosen.end(), generations[index]) == chosen.end()) std::filesystem::remove(generations[index], error);
            }
            choices.chosen.emplace(key, std::move(chosen));
        }
        if (opened.count != 0) {
            archives = [opened copy];
            statistics.archiveLoaded = true;
        }
        char unique[64];
        std::snprintf(unique, sizeof(unique), ".%020llu-%d.mtl4archive", static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()), static_cast<int>(::getpid()));
        archivePath = directory / (stem + unique);
        statistics.archivePath = archivePath.string();
        MTL4PipelineDataSetSerializerDescriptor* serializerDescriptor = [MTL4PipelineDataSetSerializerDescriptor new];
        // Binaries only: binaries + descriptors fails to serialize on macOS 27.0.1 (D16).
        serializerDescriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
        serializer = [device newPipelineDataSetSerializerWithDescriptor:serializerDescriptor];
    }
    MTL4CompilerDescriptor* compilerDescriptor = [MTL4CompilerDescriptor new];
    compilerDescriptor.label = @"aps5";
    compilerDescriptor.pipelineDataSetSerializer = serializer;
    NSError* error = nil;
    compiler = [device newCompilerWithDescriptor:compilerDescriptor error:&error];
    if (compiler == nil) throw std::runtime_error("Metal pipeline cache (work package D5): no shader compiler: " + Describe(error));

    // The library store (D55) and its prewarm thread: one thread at background QoS, so it neither
    // takes the pipeline workers from the guest's draws nor a performance core from the guest.
    store = std::make_unique<MetalLibraryStore>(storeDirectory, MetalLibraryStore::DefaultByteLimit());
    if (store->Enabled()) {
        statistics.libraryStorePath = store->Directory().string();
        auto stored = store->Load();
        statistics.librariesStored = stored.size();
        if (Profile()) std::fprintf(stderr, "[metal-pipelines] library store %s: %zu libraries\n", store->Directory().c_str(), stored.size());
        bool first = false;
        if (!stored.empty() && MetalLibraryStore::PrewarmEnabled()) {
            auto& shared = Shared();
            std::lock_guard lock(shared.mutex);
            first = shared.prewarmed.insert(device.registryID).second;
        }
        if (first) {
            prewarmThread = std::thread([this, stored = std::move(stored)]() mutable { prewarm(std::move(stored)); });
        } else {
            statistics.prewarmFinished = true;
        }
    } else {
        statistics.prewarmFinished = true;
    }

    // Background workers: as many as the device compiles concurrently, at most 3 (the guest's own
    // threads keep the performance cores), at utility QoS so they prefer the efficiency cores.
    const auto count = std::clamp<std::size_t>(device.maximumConcurrentCompilationTaskCount, 1u, 3u);
    for (std::size_t index = 0; index < count; ++index) workers.emplace_back([this] { worker(); });
    if (serializer != nil && !archivePath.empty()) saverThread = std::thread([this] { saver(); });
}

PipelineCache::~PipelineCache() {
    prewarmStop.store(true, std::memory_order_release);
    if (prewarmThread.joinable()) prewarmThread.join();
    {
        std::lock_guard lock(queueMutex);
        stopping = true;
    }
    queueWake.notify_all();
    for (auto& thread : workers) thread.join();
    {
        std::lock_guard lock(saverMutex);
        saverStopping = true;
    }
    saverWake.notify_all();
    if (saverThread.joinable()) saverThread.join();
    Save();
    if (Profile()) {
        const auto stats = Statistics();
        std::fprintf(stderr, "[metal-pipelines] libraries %llu compiled %llu hits; pipelines %llu compiled, %llu from the archive, %llu memory hits; %llu background tasks, %llu waits (%.1f ms); compile %.1f ms; archive %s%s, %llu saves\n",
            static_cast<unsigned long long>(stats.librariesCompiled), static_cast<unsigned long long>(stats.libraryHits), static_cast<unsigned long long>(stats.pipelinesCompiled), static_cast<unsigned long long>(stats.archiveHits),
            static_cast<unsigned long long>(stats.memoryHits), static_cast<unsigned long long>(stats.backgroundTasks), static_cast<unsigned long long>(stats.waits), stats.waitMilliseconds, stats.compileMilliseconds,
            stats.archivePath.empty() ? "(none)" : stats.archivePath.c_str(), stats.archiveLoaded ? " (loaded)" : "", static_cast<unsigned long long>(stats.archiveSaves));
        std::fprintf(stderr, "[metal-pipelines] library store %s: %llu stored at start, %llu prewarmed, %llu requests served by the prewarm\n", stats.libraryStorePath.empty() ? "(none)" : stats.libraryStorePath.c_str(),
            static_cast<unsigned long long>(stats.librariesStored), static_cast<unsigned long long>(stats.librariesPrewarmed), static_cast<unsigned long long>(stats.prewarmHits));
    }
}

id<MTLLibrary> PipelineCache::Library(const ShaderRecompiler::MslShader& shader, const std::vector<std::pair<std::string, std::string>>& macros) {
    return library(shader.source, macros, false);
}

id<MTLLibrary> PipelineCache::library(const std::string& source, const std::vector<std::pair<std::string, std::string>>& macros, bool prewarm) {
    std::string key = "msl" + std::to_string(ShaderRecompiler::MslAbi::LanguageMajor) + "." + std::to_string(ShaderRecompiler::MslAbi::LanguageMinor) + "\n" + source;
    for (const auto& [name, value] : macros) key += "\n#" + name + "=" + value;
    std::shared_ptr<LibraryTask> task;
    bool owner = false;
    {
        std::lock_guard lock(mutex);
        if (const auto found = libraries.find(key); found != libraries.end()) {
            if (!prewarm) ++statistics.libraryHits;
            return found->second;
        }
    }
    const std::string sharedKey = std::to_string(device.registryID) + "\n" + key;
    {
        auto& shared = Shared();
        std::lock_guard lock(shared.mutex);
        if (const auto found = shared.libraries.find(sharedKey); found != shared.libraries.end()) {
            id<MTLLibrary> library = found->second;
            std::lock_guard own(mutex);
            if (!prewarm) ++statistics.libraryHits;
            libraries.emplace(key, library);
            return library;
        }
        if (const auto running = shared.tasks.find(sharedKey); running != shared.tasks.end()) {
            task = std::static_pointer_cast<LibraryTask>(running->second);
        } else {
            task = std::make_shared<LibraryTask>();
            shared.tasks.emplace(sharedKey, task);
            owner = true;
        }
    }
    if (!owner) {
        // Another thread (the prewarm thread, or a draw) compiles this library: wait for it.
        if (prewarm) return nil;
        std::unique_lock wait(task->mutex);
        task->done.wait(wait, [&] { return task->finished; });
        if (task->library == nil) throw std::runtime_error(task->failure);
        std::lock_guard lock(mutex);
        ++statistics.prewarmHits;
        libraries.emplace(key, task->library);
        return task->library;
    }
    const StoredLibrary stored{ShaderRecompiler::MslAbi::LanguageMajor, ShaderRecompiler::MslAbi::LanguageMinor, macros, source};
    // Recorded before the compile: a guest that gives up while it runs still leaves it for the next
    // process to compile ahead of use.
    if (!prewarm) store->Record(stored);
    id<MTLLibrary> compiled = nil;
    std::string failure;
    @autoreleasepool {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = static_cast<MTLLanguageVersion>((ShaderRecompiler::MslAbi::LanguageMajor << 16u) | ShaderRecompiler::MslAbi::LanguageMinor);
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        if (!macros.empty()) {
            NSMutableDictionary<NSString*, NSObject*>* defines = [NSMutableDictionary dictionary];
            for (const auto& [name, value] : macros) defines[@(name.c_str())] = @(value.c_str());
            options.preprocessorMacros = defines;
        }
        // Debug aid (APS5_METAL_DUMP_MSL=<dir>): every compiled library's source, its macros first
        // as #defines, written to <dir>/library_<n>.metal.
        if (const char* dumpDir = std::getenv("APS5_METAL_DUMP_MSL"); !prewarm && dumpDir != nullptr && *dumpDir != '\0') {
            static std::atomic<unsigned> dumped{0};
            const auto path = std::string(dumpDir) + "/library_" + std::to_string(dumped.fetch_add(1)) + ".metal";
            if (std::FILE* file = std::fopen(path.c_str(), "w"); file != nullptr) {
                for (const auto& [name, value] : macros) std::fprintf(file, "#define %s %s\n", name.c_str(), value.c_str());
                std::fputs(source.c_str(), file);
                std::fclose(file);
                std::fprintf(stderr, "[metal] MSL library written to %s\n", path.c_str());
            }
        }
        MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
        // Measurement aid (APS5_METAL_UNIQUE_LIBRARIES=1): every compiled source gets a comment unique
        // to the process, so neither the system compiler cache nor an earlier run serves it (cold
        // compile times; combine with APS5_NO_LIBRARY_CACHE=1).
        static const bool unique = [] {
            const char* value = std::getenv("APS5_METAL_UNIQUE_LIBRARIES");
            return value != nullptr && value[0] == '1';
        }();
        static const std::string uniqueComment = "\n// aps5 unique " + std::to_string(::getpid()) + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "\n";
        descriptor.source = [NSString stringWithUTF8String:(unique ? source + uniqueComment : source).c_str()];
        descriptor.options = options;
        NSError* error = nil;
        const auto start = std::chrono::steady_clock::now();
        const SlowCompileReport report(prewarm ? "a stored MSL library (prewarm)" : "an MSL library", source.size());
        {
            std::shared_lock gate(serializerGate);
            compiled = [compiler newLibraryWithDescriptor:descriptor error:&error];
        }
        if (compiled == nil) {
            failure = "Metal device: MSL compile failed: " + Describe(error);
            store->Forget(stored);
            if (!prewarm) std::fprintf(stderr, "[metal] MSL compile failed:\n%s\n", source.c_str());
            else std::fprintf(stderr, "[metal-pipelines] a stored library no longer compiles (dropped): %s\n", Describe(error).c_str());
        }
        {
            auto& shared = Shared();
            std::lock_guard lock(shared.mutex);
            if (compiled != nil) shared.libraries.emplace(sharedKey, compiled);
            shared.tasks.erase(sharedKey);
        }
        std::lock_guard lock(mutex);
        if (compiled != nil) {
            if (prewarm) ++statistics.librariesPrewarmed;
            else ++statistics.librariesCompiled;
            statistics.compileMilliseconds += Milliseconds(start);
            libraries.emplace(key, compiled);
        }
    }
    {
        std::lock_guard lock(task->mutex);
        task->library = compiled;
        task->failure = failure;
        task->finished = true;
    }
    task->done.notify_all();
    if (compiled == nil && !prewarm) throw std::runtime_error(failure);
    return compiled;
}

void PipelineCache::prewarm(std::vector<StoredLibrary> stored) {
    // Library compiles run concurrently in the compiler service (agc_metal_msl_corpus parallel
    // probe: 1.9x with 2 threads, about 3x with 4 through one MTL4Compiler), so the stored libraries
    // are compiled by as many background threads as the device compiles at once, at most 4.
    const auto start = std::chrono::steady_clock::now();
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> compiled{0};
    auto run = [&] {
        pthread_setname_np("pipeline prewarm");
        pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
        for (std::size_t index = next.fetch_add(1); index < stored.size(); index = next.fetch_add(1)) {
            if (prewarmStop.load(std::memory_order_acquire)) break;
            const auto& library = stored[index];
            if (library.languageMajor != ShaderRecompiler::MslAbi::LanguageMajor || library.languageMinor != ShaderRecompiler::MslAbi::LanguageMinor) {
                store->Forget(library);
                continue;
            }
            try {
                if (this->library(library.source, library.macros, true) != nil) compiled.fetch_add(1);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[metal-pipelines] prewarm: %s\n", error.what());
            }
        }
    };
    const auto threads = std::clamp<std::size_t>(std::min<std::size_t>(device.maximumConcurrentCompilationTaskCount, stored.size()), 1u, 4u);
    std::vector<std::thread> helpers;
    for (std::size_t index = 1; index < threads; ++index) helpers.emplace_back(run);
    run();
    for (auto& helper : helpers) helper.join();
    if (Profile()) std::fprintf(stderr, "[metal-pipelines] prewarm compiled %zu of %zu stored libraries on %zu threads in %.1f ms\n", compiled.load(), stored.size(), threads, Milliseconds(start));
    {
        std::lock_guard lock(mutex);
        statistics.prewarmFinished = true;
    }
    prewarmDone.notify_all();
}

void PipelineCache::ForgetSharedLibraries() {
    auto& shared = Shared();
    std::lock_guard lock(shared.mutex);
    shared.libraries.clear();
    shared.prewarmed.clear();
}

void PipelineCache::WaitForPrewarm() {
    std::unique_lock lock(mutex);
    prewarmDone.wait(lock, [&] { return statistics.prewarmFinished; });
}

MTL4FunctionDescriptor* PipelineCache::functionDescriptor(const PipelineFunction& function) {
    id<MTLLibrary> library = Library(*function.shader, function.macros);
    MTL4LibraryFunctionDescriptor* named = [MTL4LibraryFunctionDescriptor new];
    named.name = @(ShaderRecompiler::MslAbi::EntryPoint);
    named.library = library;
    if (function.constants.empty()) return named;
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    for (const auto& [id, value] : function.constants) {
        const std::uint32_t bits = value;
        [values setConstantValue:&bits type:MTLDataTypeUInt atIndex:id];
    }
    MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
    specialized.functionDescriptor = named;
    specialized.constantValues = values;
    return specialized;
}

std::string PipelineCache::ComputeKey(const PipelineFunction& function) {
    std::string key = "C\n";
    AppendFunction(key, function);
    return key;
}

std::string PipelineCache::RenderKey(const RenderPipelineRequest& request) {
    std::string key = "R\n";
    AppendFunction(key, request.vertex);
    if (request.fragment) AppendFunction(key, *request.fragment);
    else key += "-\n";
    if (request.object) AppendFunction(key, *request.object);
    else key += "-\n";
    key += std::to_string(request.objectThreads) + ',' + std::to_string(request.meshGroups) + '\n';
    for (const auto& color : request.colors) AppendColor(key, color);
    for (const auto value : {static_cast<std::uint64_t>(request.depthFormat), static_cast<std::uint64_t>(request.stencilFormat), static_cast<std::uint64_t>(request.rasterSampleCount), static_cast<std::uint64_t>(request.alphaToCoverage),
                             static_cast<std::uint64_t>(request.alphaToOne), static_cast<std::uint64_t>(request.rasterization), static_cast<std::uint64_t>(request.topology),
                             static_cast<std::uint64_t>(request.mesh), static_cast<std::uint64_t>(request.meshThreads)}) {
        key += std::to_string(value) + ',';
    }
    return key;
}

std::shared_ptr<PipelineCache::Entry> PipelineCache::entryFor(const std::string& key, bool* created) {
    std::lock_guard lock(mutex);
    auto& entry = entries[key];
    *created = entry == nullptr;
    if (entry == nullptr) entry = std::make_shared<Entry>();
    return entry;
}

id<MTLComputePipelineState> PipelineCache::obtainCompute(const std::function<MTL4ComputePipelineDescriptor*()>& describe, const char* what) {
    @autoreleasepool {
        MTL4ComputePipelineDescriptor* descriptor = describe();
        NSError* error = nil;
        for (id<MTL4Archive> archive in archives) {
            id<MTLComputePipelineState> found = [archive newComputePipelineStateWithDescriptor:descriptor error:&error];
            if (found != nil) {
                {
                    std::lock_guard lock(mutex);
                    ++statistics.archiveHits;
                }
                noteArchived([this, describe](NSArray<id<MTL4Archive>>* lookups) {
                    MTL4CompilerTaskOptions* options = [MTL4CompilerTaskOptions new];
                    options.lookupArchives = lookups;
                    NSError* ignored = nil;
                    (void)[compiler newComputePipelineStateWithDescriptor:describe() compilerTaskOptions:options error:&ignored];
                }, false);
                return found;
            }
            error = nil;
        }
        const auto start = std::chrono::steady_clock::now();
        const SlowCompileReport report(what, 0u);
        std::shared_lock gate(serializerGate);
        id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (pipeline == nil) throw std::runtime_error("Metal device: compute pipeline creation failed: " + Describe(error));
        const auto elapsed = Milliseconds(start);
        {
            std::lock_guard lock(mutex);
            ++statistics.pipelinesCompiled;
            statistics.compileMilliseconds += elapsed;
            ++unsaved;
        }
        // Still under the gate: the capture and this note are one step for Save's flush.
        noteArchived([this, describe](NSArray<id<MTL4Archive>>* lookups) {
            MTL4CompilerTaskOptions* options = [MTL4CompilerTaskOptions new];
            options.lookupArchives = lookups;
            NSError* ignored = nil;
            (void)[compiler newComputePipelineStateWithDescriptor:describe() compilerTaskOptions:options error:&ignored];
        }, true);
        if (Profile()) std::fprintf(stderr, "[metal-pipelines] compiled %s in %.1f ms\n", what, elapsed);
        requestSave();
        return pipeline;
    }
}

id<MTLRenderPipelineState> PipelineCache::obtainRender(const std::function<MTL4PipelineDescriptor*()>& describe, const char* what) {
    @autoreleasepool {
        MTL4PipelineDescriptor* descriptor = describe();
        NSError* error = nil;
        for (id<MTL4Archive> archive in archives) {
            id<MTLRenderPipelineState> found = [archive newRenderPipelineStateWithDescriptor:descriptor error:&error];
            if (found != nil) {
                {
                    std::lock_guard lock(mutex);
                    ++statistics.archiveHits;
                }
                noteArchived([this, describe](NSArray<id<MTL4Archive>>* lookups) {
                    MTL4CompilerTaskOptions* options = [MTL4CompilerTaskOptions new];
                    options.lookupArchives = lookups;
                    NSError* ignored = nil;
                    (void)[compiler newRenderPipelineStateWithDescriptor:describe() compilerTaskOptions:options error:&ignored];
                }, false);
                return found;
            }
            error = nil;
        }
        const auto start = std::chrono::steady_clock::now();
        const SlowCompileReport report(what, 0u);
        std::shared_lock gate(serializerGate);
        id<MTLRenderPipelineState> pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (pipeline == nil) throw std::runtime_error(std::string("Metal device: ") + what + " creation failed: " + Describe(error));
        const auto elapsed = Milliseconds(start);
        {
            std::lock_guard lock(mutex);
            ++statistics.pipelinesCompiled;
            statistics.compileMilliseconds += elapsed;
            ++unsaved;
        }
        noteArchived([this, describe](NSArray<id<MTL4Archive>>* lookups) {
            MTL4CompilerTaskOptions* options = [MTL4CompilerTaskOptions new];
            options.lookupArchives = lookups;
            NSError* ignored = nil;
            (void)[compiler newRenderPipelineStateWithDescriptor:describe() compilerTaskOptions:options error:&ignored];
        }, true);
        if (Profile()) std::fprintf(stderr, "[metal-pipelines] compiled %s in %.1f ms\n", what, elapsed);
        requestSave();
        return pipeline;
    }
}

void PipelineCache::noteArchived(std::function<void(NSArray<id<MTL4Archive>>*)> replay, bool captured) {
    if (serializer == nil) return;
    std::lock_guard lock(mutex);
    archived.push_back({std::move(replay), captured});
}

id<MTLComputePipelineState> PipelineCache::buildCompute(const PipelineFunction& function) {
    return obtainCompute([this, function] {
        MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = functionDescriptor(function);
        return descriptor;
    }, "a compute pipeline");
}

MTL4PipelineDescriptor* PipelineCache::renderDescriptor(const RenderPipelineRequest& request) {
    MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = functionDescriptor(request.vertex);
    if (request.fragment) descriptor.fragmentFunctionDescriptor = functionDescriptor(*request.fragment);
    for (std::size_t index = 0; index < request.colors.size(); ++index) {
        const auto& color = request.colors[index];
        if (color.format == MTLPixelFormatInvalid) continue;
        MTL4RenderPipelineColorAttachmentDescriptor* attachment = descriptor.colorAttachments[index];
        attachment.pixelFormat = color.format;
        attachment.writeMask = color.writeMask;
        attachment.blendingState = color.blending ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
        if (color.blending) {
            attachment.sourceRGBBlendFactor = color.sourceRgb;
            attachment.destinationRGBBlendFactor = color.destinationRgb;
            attachment.rgbBlendOperation = color.rgbOperation;
            attachment.sourceAlphaBlendFactor = color.sourceAlpha;
            attachment.destinationAlphaBlendFactor = color.destinationAlpha;
            attachment.alphaBlendOperation = color.alphaOperation;
        }
    }
    if (request.depthFormat != MTLPixelFormatInvalid || request.stencilFormat != MTLPixelFormatInvalid) {
        // Metal 4 render pipelines take the depth/stencil formats from the render pass at encode
        // time; nothing to set beyond validating the pair here.
        if (request.stencilFormat != MTLPixelFormatInvalid && request.depthFormat != MTLPixelFormatInvalid && request.stencilFormat != request.depthFormat) {
            throw std::runtime_error("Metal pipeline cache (work package D5): separate depth and stencil formats are not supported");
        }
    }
    descriptor.rasterSampleCount = request.rasterSampleCount;
    descriptor.alphaToCoverageState = request.alphaToCoverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
    descriptor.alphaToOneState = request.alphaToOne ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
    descriptor.rasterizationEnabled = request.rasterization;
    descriptor.inputPrimitiveTopology = request.topology;
    return descriptor;
}

id<MTLRenderPipelineState> PipelineCache::buildRender(const RenderPipelineRequest& request) {
    if (request.mesh) return buildMeshRender(request);
    return obtainRender([this, request] { return renderDescriptor(request); }, "a render pipeline");
}

namespace {
std::atomic<std::uint64_t> guestComputeCreated{0};
std::atomic<std::uint64_t> guestComputeCreatedLocked{0};

void NoteGuestComputePipeline() {
    guestComputeCreated.fetch_add(1, std::memory_order_relaxed);
    if (GuestMemory::GpuMutex().HeldByThisThread()) guestComputeCreatedLocked.fetch_add(1, std::memory_order_relaxed);
}
}

ComputePipelineCreation ComputePipelineCreationCounts() {
    return {guestComputeCreated.load(std::memory_order_relaxed), guestComputeCreatedLocked.load(std::memory_order_relaxed)};
}

id<MTLComputePipelineState> PipelineCache::Compute(const ShaderRecompiler::RecompileResult& shader) {
    return compute(PipelineFunction::From(shader), true);
}

id<MTLComputePipelineState> PipelineCache::Compute(const PipelineFunction& function) {
    return compute(function, false);
}

id<MTLComputePipelineState> PipelineCache::compute(const PipelineFunction& function, bool guest) {
    bool created = false;
    const auto entry = entryFor(ComputeKey(function), &created);
    std::unique_lock lock(entry->mutex);
    if (entry->ready) {
        lock.unlock();
        std::lock_guard stats(mutex);
        ++statistics.memoryHits;
        return entry->compute;
    }
    if (entry->running) {
        const auto start = std::chrono::steady_clock::now();
        entry->done.wait(lock, [&] { return entry->ready; });
        std::lock_guard stats(mutex);
        ++statistics.waits;
        statistics.waitMilliseconds += Milliseconds(start);
    } else {
        entry->running = true;
        lock.unlock();
        std::string failure;
        id<MTLComputePipelineState> pipeline = nil;
        try {
            pipeline = buildCompute(function);
            if (guest) NoteGuestComputePipeline();
        } catch (const std::exception& error) {
            failure = error.what();
        }
        lock.lock();
        entry->compute = pipeline;
        entry->failure = failure;
        entry->ready = true;
        entry->running = false;
        entry->done.notify_all();
    }
    if (!entry->failure.empty()) {
        const auto failure = entry->failure;
        lock.unlock();
        // A failed build is not cached: a later request tries again (and throws again).
        std::lock_guard stats(mutex);
        entries.erase(ComputeKey(function));
        throw std::runtime_error(failure);
    }
    return entry->compute;
}

void PipelineCache::PrepareCompute(const ShaderRecompiler::RecompileResult& shader) {
    auto function = PipelineFunction::From(shader);
    bool created = false;
    const auto entry = entryFor(ComputeKey(function), &created);
    {
        std::lock_guard lock(entry->mutex);
        if (entry->ready || entry->running) return;
        entry->running = true;
    }
    {
        std::lock_guard stats(mutex);
        ++statistics.backgroundTasks;
    }
    // Counted here, on the requesting thread (upstream 7fae4549e counts the pipeline PrepareDispatch creates before
    // the GPU mutex): the build itself runs on a worker, and the dispatch that later obtains it creates nothing.
    NoteGuestComputePipeline();
    submit([this, entry, function = std::move(function)] {
        std::string failure;
        id<MTLComputePipelineState> pipeline = nil;
        try {
            pipeline = buildCompute(function);
        } catch (const std::exception& error) {
            failure = error.what();
        }
        std::lock_guard lock(entry->mutex);
        entry->compute = pipeline;
        entry->failure = failure;
        entry->ready = true;
        entry->running = false;
        entry->done.notify_all();
    });
}

id<MTLRenderPipelineState> PipelineCache::Render(const RenderPipelineRequest& request) {
    const auto key = RenderKey(request);
    bool created = false;
    const auto entry = entryFor(key, &created);
    std::unique_lock lock(entry->mutex);
    if (entry->ready && entry->failure.empty()) {
        lock.unlock();
        std::lock_guard stats(mutex);
        ++statistics.memoryHits;
        return entry->render;
    }
    if (entry->running) {
        const auto start = std::chrono::steady_clock::now();
        entry->done.wait(lock, [&] { return entry->ready; });
        std::lock_guard stats(mutex);
        ++statistics.waits;
        statistics.waitMilliseconds += Milliseconds(start);
    } else {
        entry->running = true;
        lock.unlock();
        std::string failure;
        id<MTLRenderPipelineState> pipeline = nil;
        try {
            pipeline = buildRender(request);
        } catch (const std::exception& error) {
            failure = error.what();
        }
        lock.lock();
        entry->render = pipeline;
        entry->failure = failure;
        entry->ready = true;
        entry->running = false;
        entry->done.notify_all();
    }
    if (!entry->failure.empty()) {
        const auto failure = entry->failure;
        lock.unlock();
        std::lock_guard stats(mutex);
        entries.erase(key);
        throw std::runtime_error(failure);
    }
    return entry->render;
}

void PipelineCache::PrepareRender(const RenderPipelineRequest& request) {
    bool created = false;
    const auto entry = entryFor(RenderKey(request), &created);
    {
        std::lock_guard lock(entry->mutex);
        if (entry->ready || entry->running) return;
        entry->running = true;
    }
    {
        std::lock_guard stats(mutex);
        ++statistics.backgroundTasks;
    }
    submit([this, entry, request] {
        std::string failure;
        id<MTLRenderPipelineState> pipeline = nil;
        try {
            pipeline = buildRender(request);
        } catch (const std::exception& error) {
            failure = error.what();
        }
        std::lock_guard lock(entry->mutex);
        entry->render = pipeline;
        entry->failure = failure;
        entry->ready = true;
        entry->running = false;
        entry->done.notify_all();
    });
}

id<MTLDepthStencilState> PipelineCache::DepthStencil(const DepthStencilRequest& request) {
    std::lock_guard lock(mutex);
    for (const auto& [known, state] : depthStencilStates) {
        if (known == request) return state;
    }
    MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
    descriptor.depthCompareFunction = request.depthCompare;
    descriptor.depthWriteEnabled = request.depthWrite;
    if (request.stencil) {
        const auto face = [](const StencilFaceState& state) {
            MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
            stencil.stencilCompareFunction = state.compare;
            stencil.stencilFailureOperation = state.fail;
            stencil.depthFailureOperation = state.depthFail;
            stencil.depthStencilPassOperation = state.pass;
            stencil.readMask = state.readMask;
            stencil.writeMask = state.writeMask;
            return stencil;
        };
        descriptor.frontFaceStencil = face(request.front);
        descriptor.backFaceStencil = face(request.back);
    }
    id<MTLDepthStencilState> state = [device newDepthStencilStateWithDescriptor:descriptor];
    if (state == nil) throw std::runtime_error("Metal pipeline cache (work package D5): Metal refused a depth-stencil state");
    depthStencilStates.emplace_back(request, state);
    return state;
}

void PipelineCache::submit(std::function<void()> task) {
    {
        std::lock_guard lock(queueMutex);
        if (stopping) {
            // Shutting down: run inline so a waiter is never left behind.
            queue.push_front(std::move(task));
        } else {
            queue.push_back(std::move(task));
        }
    }
    queueWake.notify_one();
}

void PipelineCache::worker() {
    pthread_setname_np("pipeline compile");
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock lock(queueMutex);
            queueWake.wait(lock, [&] { return stopping || !queue.empty(); });
            if (queue.empty()) return;
            task = std::move(queue.front());
            queue.pop_front();
            ++running;
        }
        @autoreleasepool {
            try {
                task();
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[metal-pipelines] background task failed: %s\n", error.what());
            }
        }
        {
            std::lock_guard lock(queueMutex);
            --running;
        }
        queueIdle.notify_all();
    }
}

void PipelineCache::Drain() {
    std::unique_lock lock(queueMutex);
    queueIdle.wait(lock, [&] { return queue.empty() && running == 0; });
}

void PipelineCache::requestSave() {
    if (serializer == nil || archivePath.empty()) return;
    {
        std::lock_guard lock(saverMutex);
        saveRequested = true;
    }
    saverWake.notify_one();
}

void PipelineCache::saver() {
    pthread_setname_np("pipeline saver");
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    std::unique_lock lock(saverMutex);
    for (;;) {
        saverWake.wait(lock, [&] { return saverStopping || saveRequested; });
        if (saverStopping) return;
        // Coalesce: requests made while waiting for the spacing (or while saving) are this save.
        if (saverWake.wait_until(lock, nextSave, [&] { return saverStopping; })) return;
        saveRequested = false;
        lock.unlock();
        const auto start = std::chrono::steady_clock::now();
        const bool done = Save(false);
        const auto finished = std::chrono::steady_clock::now();
        lock.lock();
        // A save skipped behind a pipeline creation is asked again for the next interval.
        if (!done) saveRequested = true;
        nextSave = finished + std::max<std::chrono::steady_clock::duration>(SaveSpacing, (finished - start) * SaveCostFactor);
    }
}

bool PipelineCache::Save(bool waitForPipelines) {
    // Never destroyed: the first save runs early, so a function-local mutex would be destroyed at
    // exit() before the exit handler tears a live device (and its cache) down.
    static std::mutex* const saveMutex = new std::mutex;
    std::lock_guard saving(*saveMutex);
    std::uint64_t saved = 0;
    std::vector<std::function<void(NSArray<id<MTL4Archive>>*)>> replays;
    std::size_t known = 0;
    {
        std::lock_guard lock(mutex);
        if (serializer == nil || archivePath.empty() || unsaved == 0) return true;
        saved = unsaved;
        known = archived.size();
        for (std::size_t index = 0; index < known; ++index) {
            if (!archived[index].captured) replays.push_back(archived[index].replay);
        }
    }
    @autoreleasepool {
        // Replays (D51): serializeAsArchiveAndFlushToURL empties the serializer, so the pipelines an
        // earlier save flushed and the archive's hits are captured again, looked up (not compiled) in
        // the archive they came from.
        NSMutableArray<id<MTL4Archive>>* lookups = [NSMutableArray array];
        if (written != nil) [lookups addObject:written];
        if (archives != nil) [lookups addObjectsFromArray:archives];
        const auto replayStart = std::chrono::steady_clock::now();
        for (auto& replay : replays) {
            @autoreleasepool {
                try {
                    std::shared_lock gate(serializerGate);
                    replay(lookups);
                } catch (const std::exception&) {
                }
            }
        }
        const auto replayElapsed = Milliseconds(replayStart);
        std::error_code error;
        std::filesystem::create_directories(archivePath.parent_path(), error);
        auto temporary = archivePath;
        temporary += ".tmp" + std::to_string(::getpid());
        NSError* failure = nil;
        bool stored = false;
        {
            std::unique_lock gate(serializerGate, std::defer_lock);
            if (waitForPipelines) gate.lock();
            else if (!gate.try_lock()) return false;
            stored = [serializer serializeAsArchiveAndFlushToURL:[NSURL fileURLWithPath:@(temporary.c_str())] error:&failure];
            // Whatever succeeded or not, the serializer is empty now: everything noted so far must
            // be replayed before the next save (the first `known` entries were replayed above, the
            // rest were captured or are hits, all flushed by this call).
            std::lock_guard lock(mutex);
            for (auto& entry : archived) entry.captured = false;
            statistics.archiveReplays += replays.size();
            statistics.replayMilliseconds += replayElapsed;
        }
        if (!stored) {
            std::fprintf(stderr, "[metal-pipelines] cannot write %s: %s\n", temporary.c_str(), Describe(failure).c_str());
            std::filesystem::remove(temporary, error);
            return true;
        }
        std::filesystem::rename(temporary, archivePath, error);
        if (error) {
            std::fprintf(stderr, "[metal-pipelines] cannot rename %s: %s\n", temporary.c_str(), error.message().c_str());
            std::filesystem::remove(temporary, error);
            return true;
        }
        // The pipelines just flushed are replayed from this file at the next save. Opened at its
        // final path: Metal resolves a lookup archive through its file (one opened at the temporary
        // path made the compiler throw on a nil archive once renamed). Later saves replace the file
        // with a superset, so the lookups still find everything this one held.
        NSError* openFailure = nil;
        id<MTL4Archive> reopened = [device newArchiveWithURL:[NSURL fileURLWithPath:@(archivePath.c_str())] error:&openFailure];
        if (reopened == nil) {
            std::fprintf(stderr, "[metal-pipelines] cannot reopen the archive just written (%s): the next save compiles its pipelines again\n", Describe(openFailure).c_str());
        } else {
            written = reopened;
        }
        if (Profile()) std::fprintf(stderr, "[metal-pipelines] archive %s saved (%llu new pipelines, %zu replayed in %.1f ms)\n", archivePath.c_str(), static_cast<unsigned long long>(saved), replays.size(), replayElapsed);
    }
    std::lock_guard lock(mutex);
    unsaved -= std::min(unsaved, saved);
    ++statistics.archiveSaves;
    return true;
}

PipelineCacheStatistics PipelineCache::Statistics() const {
    std::lock_guard lock(mutex);
    return statistics;
}

}
