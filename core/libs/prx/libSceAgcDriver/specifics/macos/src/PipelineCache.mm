// Work package D5: Metal twin of Graphics/src/PipelineCache.cpp (see MetalPipelineCache.h).
#include "MetalPipelineCache.h"
#include "ShaderCacheDirectory.hpp"

#include <sys/sysctl.h>
#include <atomic>
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

MTLLanguageVersion Language(const ShaderRecompiler::MslShader&) {
    // The MSL version the backend emits (MslAbi::LanguageMajor/Minor), as MetalDevice.mm's ShaderLanguage().
    return static_cast<MTLLanguageVersion>((ShaderRecompiler::MslAbi::LanguageMajor << 16u) | ShaderRecompiler::MslAbi::LanguageMinor);
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
    if (ShaderValidation()) {
        if (!directory.empty() && Profile()) std::fprintf(stderr, "[metal-pipelines] no archive under MTL_SHADER_VALIDATION\n");
        directory.clear();
        statistics.archiveDisabled = "MTL_SHADER_VALIDATION";
    }
    if (!directory.empty()) {
        archivePath = directory / FileSafe(std::string(device.name.UTF8String) + "-" + OsBuild() + ".mtl4archive");
        statistics.archivePath = archivePath.string();
        std::error_code error;
        if (std::filesystem::exists(archivePath, error)) {
            NSError* failure = nil;
            archive = [device newArchiveWithURL:[NSURL fileURLWithPath:@(archivePath.c_str())] error:&failure];
            if (archive == nil) {
                std::fprintf(stderr, "[metal-pipelines] ignoring the unreadable archive %s: %s\n", archivePath.c_str(), Describe(failure).c_str());
            } else {
                statistics.archiveLoaded = true;
            }
        }
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

    // Background workers: as many as the device compiles concurrently, at most 3 (the guest's own
    // threads keep the performance cores), at utility QoS so they prefer the efficiency cores.
    const auto count = std::clamp<std::size_t>(device.maximumConcurrentCompilationTaskCount, 1u, 3u);
    for (std::size_t index = 0; index < count; ++index) workers.emplace_back([this] { worker(); });
}

PipelineCache::~PipelineCache() {
    {
        std::lock_guard lock(queueMutex);
        stopping = true;
    }
    queueWake.notify_all();
    for (auto& thread : workers) thread.join();
    Save();
    if (Profile()) {
        const auto stats = Statistics();
        std::fprintf(stderr, "[metal-pipelines] libraries %llu compiled %llu hits; pipelines %llu compiled, %llu from the archive, %llu memory hits; %llu background tasks, %llu waits (%.1f ms); compile %.1f ms; archive %s%s, %llu saves\n",
            static_cast<unsigned long long>(stats.librariesCompiled), static_cast<unsigned long long>(stats.libraryHits), static_cast<unsigned long long>(stats.pipelinesCompiled), static_cast<unsigned long long>(stats.archiveHits),
            static_cast<unsigned long long>(stats.memoryHits), static_cast<unsigned long long>(stats.backgroundTasks), static_cast<unsigned long long>(stats.waits), stats.waitMilliseconds, stats.compileMilliseconds,
            stats.archivePath.empty() ? "(none)" : stats.archivePath.c_str(), stats.archiveLoaded ? " (loaded)" : "", static_cast<unsigned long long>(stats.archiveSaves));
    }
}

id<MTLLibrary> PipelineCache::Library(const ShaderRecompiler::MslShader& shader, const std::vector<std::pair<std::string, std::string>>& macros) {
    std::string key = "msl" + std::to_string(ShaderRecompiler::MslAbi::LanguageMajor) + "." + std::to_string(ShaderRecompiler::MslAbi::LanguageMinor) + "\n" + shader.source;
    for (const auto& [name, value] : macros) key += "\n#" + name + "=" + value;
    {
        std::lock_guard lock(mutex);
        if (const auto found = libraries.find(key); found != libraries.end()) {
            ++statistics.libraryHits;
            return found->second;
        }
    }
    @autoreleasepool {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = Language(shader);
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        if (!macros.empty()) {
            NSMutableDictionary<NSString*, NSObject*>* defines = [NSMutableDictionary dictionary];
            for (const auto& [name, value] : macros) defines[@(name.c_str())] = @(value.c_str());
            options.preprocessorMacros = defines;
        }
        // Debug aid (APS5_METAL_DUMP_MSL=<dir>): every compiled library's source, its macros first
        // as #defines, written to <dir>/library_<n>.metal.
        if (const char* dumpDir = std::getenv("APS5_METAL_DUMP_MSL"); dumpDir != nullptr && *dumpDir != '\0') {
            static std::atomic<unsigned> dumped{0};
            const auto path = std::string(dumpDir) + "/library_" + std::to_string(dumped.fetch_add(1)) + ".metal";
            if (std::FILE* file = std::fopen(path.c_str(), "w"); file != nullptr) {
                for (const auto& [name, value] : macros) std::fprintf(file, "#define %s %s\n", name.c_str(), value.c_str());
                std::fputs(shader.source.c_str(), file);
                std::fclose(file);
                std::fprintf(stderr, "[metal] MSL library written to %s\n", path.c_str());
            }
        }
        MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
        descriptor.source = [NSString stringWithUTF8String:shader.source.c_str()];
        descriptor.options = options;
        NSError* error = nil;
        const auto start = std::chrono::steady_clock::now();
        id<MTLLibrary> library = [compiler newLibraryWithDescriptor:descriptor error:&error];
        if (library == nil) {
            std::fprintf(stderr, "[metal] MSL compile failed:\n%s\n", shader.source.c_str());
            throw std::runtime_error("Metal device: MSL compile failed: " + Describe(error));
        }
        std::lock_guard lock(mutex);
        ++statistics.librariesCompiled;
        statistics.compileMilliseconds += Milliseconds(start);
        return libraries.emplace(std::move(key), library).first->second;
    }
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

id<MTLComputePipelineState> PipelineCache::buildCompute(const PipelineFunction& function) {
    @autoreleasepool {
        MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = functionDescriptor(function);
        NSError* error = nil;
        if (archive != nil) {
            id<MTLComputePipelineState> found = [archive newComputePipelineStateWithDescriptor:descriptor error:&error];
            if (found != nil) {
                std::lock_guard lock(mutex);
                ++statistics.archiveHits;
                if (serializer != nil) {
                    archiveRecaptures.push_back([this, function] {
                        MTL4ComputePipelineDescriptor* again = [MTL4ComputePipelineDescriptor new];
                        again.computeFunctionDescriptor = functionDescriptor(function);
                        MTL4CompilerTaskOptions* options = [MTL4CompilerTaskOptions new];
                        options.lookupArchives = @[archive];
                        NSError* ignored = nil;
                        (void)[compiler newComputePipelineStateWithDescriptor:again compilerTaskOptions:options error:&ignored];
                    });
                }
                return found;
            }
            error = nil;
        }
        const auto start = std::chrono::steady_clock::now();
        id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (pipeline == nil) throw std::runtime_error("Metal device: compute pipeline creation failed: " + Describe(error));
        const auto elapsed = Milliseconds(start);
        bool save = false;
        {
            std::lock_guard lock(mutex);
            ++statistics.pipelinesCompiled;
            statistics.compileMilliseconds += elapsed;
            save = ++unsaved >= SaveInterval;
        }
        if (Profile()) std::fprintf(stderr, "[metal-pipelines] compiled a compute pipeline in %.1f ms\n", elapsed);
        if (save) submit([this] { Save(); });
        return pipeline;
    }
}

id<MTLRenderPipelineState> PipelineCache::buildRender(const RenderPipelineRequest& request) {
    if (request.mesh) return buildMeshRender(request);
    @autoreleasepool {
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
        NSError* error = nil;
        if (archive != nil) {
            id<MTLRenderPipelineState> found = [archive newRenderPipelineStateWithDescriptor:descriptor error:&error];
            if (found != nil) {
                std::lock_guard lock(mutex);
                ++statistics.archiveHits;
                if (serializer != nil) {
                    archiveRecaptures.push_back([this, request] {
                        @autoreleasepool {
                            MTL4CompilerTaskOptions* options = [MTL4CompilerTaskOptions new];
                            options.lookupArchives = @[archive];
                            MTL4RenderPipelineDescriptor* again = [MTL4RenderPipelineDescriptor new];
                            again.vertexFunctionDescriptor = functionDescriptor(request.vertex);
                            if (request.fragment) again.fragmentFunctionDescriptor = functionDescriptor(*request.fragment);
                            for (std::size_t index = 0; index < request.colors.size(); ++index) {
                                const auto& color = request.colors[index];
                                if (color.format == MTLPixelFormatInvalid) continue;
                                MTL4RenderPipelineColorAttachmentDescriptor* attachment = again.colorAttachments[index];
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
                            again.rasterSampleCount = request.rasterSampleCount;
                            again.alphaToCoverageState = request.alphaToCoverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
                            again.alphaToOneState = request.alphaToOne ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
                            again.rasterizationEnabled = request.rasterization;
                            again.inputPrimitiveTopology = request.topology;
                            NSError* ignored = nil;
                            (void)[compiler newRenderPipelineStateWithDescriptor:again compilerTaskOptions:options error:&ignored];
                        }
                    });
                }
                return found;
            }
            error = nil;
        }
        const auto start = std::chrono::steady_clock::now();
        id<MTLRenderPipelineState> pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        if (pipeline == nil) throw std::runtime_error("Metal device: render pipeline creation failed: " + Describe(error));
        const auto elapsed = Milliseconds(start);
        bool save = false;
        {
            std::lock_guard lock(mutex);
            ++statistics.pipelinesCompiled;
            statistics.compileMilliseconds += elapsed;
            save = ++unsaved >= SaveInterval;
        }
        if (Profile()) std::fprintf(stderr, "[metal-pipelines] compiled a render pipeline in %.1f ms\n", elapsed);
        if (save) submit([this] { Save(); });
        return pipeline;
    }
}

id<MTLComputePipelineState> PipelineCache::Compute(const ShaderRecompiler::RecompileResult& shader) {
    return Compute(PipelineFunction::From(shader));
}

id<MTLComputePipelineState> PipelineCache::Compute(const PipelineFunction& function) {
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

void PipelineCache::recaptureHits() {
    std::vector<std::function<void()>> recaptures;
    {
        std::lock_guard lock(mutex);
        recaptures.swap(archiveRecaptures);
    }
    for (auto& recapture : recaptures) {
        @autoreleasepool {
            try {
                recapture();
            } catch (const std::exception&) {
            }
        }
    }
}

void PipelineCache::Save() {
    static std::mutex saveMutex;
    std::lock_guard saving(saveMutex);
    {
        std::lock_guard lock(mutex);
        if (serializer == nil || archivePath.empty() || unsaved == 0) return;
    }
    recaptureHits();
    std::uint64_t saved = 0;
    {
        std::lock_guard lock(mutex);
        saved = unsaved;
    }
    @autoreleasepool {
        std::error_code error;
        std::filesystem::create_directories(archivePath.parent_path(), error);
        auto temporary = archivePath;
        temporary += ".tmp" + std::to_string(::getpid());
        NSError* failure = nil;
        if (![serializer serializeAsArchiveAndFlushToURL:[NSURL fileURLWithPath:@(temporary.c_str())] error:&failure]) {
            std::fprintf(stderr, "[metal-pipelines] cannot write %s: %s\n", temporary.c_str(), Describe(failure).c_str());
            std::filesystem::remove(temporary, error);
            return;
        }
        std::filesystem::rename(temporary, archivePath, error);
        if (error) {
            std::fprintf(stderr, "[metal-pipelines] cannot rename %s: %s\n", temporary.c_str(), error.message().c_str());
            std::filesystem::remove(temporary, error);
            return;
        }
    }
    std::lock_guard lock(mutex);
    unsaved -= std::min(unsaved, saved);
    ++statistics.archiveSaves;
}

PipelineCacheStatistics PipelineCache::Statistics() const {
    std::lock_guard lock(mutex);
    return statistics;
}

}
