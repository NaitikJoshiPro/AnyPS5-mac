// agc_metal_msl_corpus (M2 exit criterion 1: MetalBackend produces valid MSL for every shader in the test corpus).
// Replays recompile requests that APS5_DUMP_REQUESTS=<dir> wrote while the GPU suite ran
// (tools/mac/gpu_suite.sh all --dump-requests build-metal/msl-corpus) through MetalBackend, compiles each MSL
// library with the options the Metal device uses (MTL4Compiler, MSL 4.0, safe math, precise functions) and builds
// the compute pipeline with the result's function constants. No GPU work is submitted.
// A request whose recompile throws on the MSL path is replayed through SpirvBackend as well: when that throws too,
// the front end rejected it (a test that checks a rejection) and it does not count against MetalBackend.
// Likewise an MSL emission failure whose request SpirvBackend refuses too (spirv-rejected): the corpus is
// every shader SpirvBackend accepts. A request recorded for a 64-lane host that SpirvBackend refuses on a
// host as wide as the Metal device's (32 lanes) is host-width-rejected: reported, not counted.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "MslAbi.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

enum class Outcome {
    Compiled,
    Emitted,
    FunctionOnly,
    TessellationLibrary,
    FrontEndRejected,
    SpirvRejected,
    HostWidthRejected,
    EmissionFailed,
    MslPathFailed,
    LibraryFailed,
    PipelineFailed,
    Unreadable,
};

constexpr const char* OutcomeName(Outcome outcome) {
    switch (outcome) {
    case Outcome::Compiled: return "compiled";
    case Outcome::Emitted: return "emitted";
    case Outcome::FunctionOnly: return "function-only";
    case Outcome::TessellationLibrary: return "tessellation-library";
    case Outcome::FrontEndRejected: return "front-end-rejected";
    case Outcome::SpirvRejected: return "spirv-rejected";
    case Outcome::HostWidthRejected: return "host-width-rejected";
    case Outcome::EmissionFailed: return "emission-failed";
    case Outcome::MslPathFailed: return "msl-path-failed";
    case Outcome::LibraryFailed: return "library-failed";
    case Outcome::PipelineFailed: return "pipeline-failed";
    case Outcome::Unreadable: return "unreadable";
    }
    return "unknown";
}

constexpr bool Counts(Outcome outcome) {
    return outcome != Outcome::Compiled && outcome != Outcome::Emitted && outcome != Outcome::FunctionOnly && outcome != Outcome::TessellationLibrary && outcome != Outcome::FrontEndRejected && outcome != Outcome::SpirvRejected && outcome != Outcome::HostWidthRejected;
}

struct Result {
    fs::path file;
    Outcome outcome = Outcome::Unreadable;
    std::string stage;
    std::string detail;
    double seconds = 0.0;
    // Size and cost metrics (MSL requests that reached the Metal compiler): emitted MSL bytes and
    // lines, recompile (front end + MetalBackend), library (MSL -> AIR) and pipeline/function
    // specialization milliseconds.
    std::size_t mslBytes = 0;
    std::size_t mslLines = 0;
    double recompileMs = 0.0;
    double libraryMs = 0.0;
    double pipelineMs = 0.0;
};

struct Options {
    std::vector<fs::path> corpora;
    fs::path report;
    unsigned workers = 0;
    std::size_t limit = 0;
    bool includeSpirv = false;
    bool verbose = false;
    std::uint32_t replayWidth = 0;
    // --cold: every library source gets a unique trailing comment, so the system Metal compiler
    // cache cannot serve it (cold compile times). --dump-msl DIR writes each emitted library.
    // --match TEXT keeps only requests whose file name contains TEXT.
    bool cold = false;
    fs::path dumpMsl;
    std::string match;
    // APS5_MSL_CORPUS_SLICE=i/n: only every n-th request from the i-th (sorted by name), so a cold
    // corpus fits gpu_suite's 30 s per-test cap in slices.
    std::size_t sliceIndex = 0;
    std::size_t sliceCount = 1;
};

double Milliseconds(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

std::string Describe(NSError* error) {
    if (error == nil) return "unknown error";
    std::string text = error.localizedDescription.UTF8String;
    if (NSString* reason = error.userInfo[NSLocalizedFailureReasonErrorKey]; reason != nil) text += std::string(" ") + reason.UTF8String;
    return text;
}

std::string FirstLines(const std::string& text, std::size_t count) {
    std::istringstream stream(text);
    std::string line;
    std::string result;
    std::size_t taken = 0;
    while (taken < count && std::getline(stream, line)) {
        if (line.empty() || line == "RecompileRequest:") continue;
        if (line.size() > 120 && line.find(' ') == std::string::npos) continue;
        if (!result.empty()) result += " | ";
        result += line;
        ++taken;
    }
    return result.size() > 600 ? result.substr(0, 600) : result;
}

std::string MetalError(const std::string& text) {
    static const std::regex error(R"(error: [^\n]*)");
    std::smatch found;
    if (std::regex_search(text, found, error)) return found.str();
    return FirstLines(text, 2);
}

std::string StageName(ShaderRecompiler::ShaderStage stage) {
    switch (stage) {
    case ShaderRecompiler::ShaderStage::Compute: return "compute";
    case ShaderRecompiler::ShaderStage::Vertex: return "vertex";
    case ShaderRecompiler::ShaderStage::Fragment: return "fragment";
    default: return "stage" + std::to_string(static_cast<unsigned>(stage));
    }
}

std::string ReadFile(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

class Checker {
public:
    Checker(id<MTLDevice> metal, std::uint32_t hostSubgroupSize, std::uint32_t replayWidth, bool cold, fs::path dumpMsl) : device(metal), hostSubgroupSize(hostSubgroupSize), replayWidth(replayWidth), cold(cold), dumpMsl(std::move(dumpMsl)) {
        NSError* error = nil;
        compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        if (compiler == nil) throw std::runtime_error("agc_metal_msl_corpus (T1): cannot create an MTL4Compiler: " + Describe(error));
        options = [MTLCompileOptions new];
        options.languageVersion = static_cast<MTLLanguageVersion>((ShaderRecompiler::MslAbi::LanguageMajor << 16u) | ShaderRecompiler::MslAbi::LanguageMinor);
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        MTL4LibraryDescriptor* invalid = [MTL4LibraryDescriptor new];
        invalid.source = @"kernel void aps5_main(device uint* x [[buffer(0)]]) { x[0] = undefined_name; }";
        invalid.options = options;
        if ([compiler newLibraryWithDescriptor:invalid error:&error] != nil || error == nil) throw std::runtime_error("agc_metal_msl_corpus (T1): the checker accepted invalid MSL");
    }

    Result Check(const fs::path& file) const {
        Result result;
        result.file = file;
        const auto started = std::chrono::steady_clock::now();
        @autoreleasepool {
            Run(file, result);
        }
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return result;
    }

private:
    void Run(const fs::path& file, Result& result) const {
        ShaderRecompiler::DeserializedRequest replay;
        try {
            replay = ShaderRecompiler::RequestSerializer{}.Deserialize(ReadFile(file));
        } catch (const std::exception& error) {
            result.outcome = Outcome::Unreadable;
            result.detail = error.what();
            return;
        }
        auto& request = replay.request;
        result.stage = StageName(request.shader.stage);
        request.useCache = false;
        request.target.backend = ShaderRecompiler::ShaderBackend::Msl;
        // As the Metal device's target (MetalDevice.mm): wave64 compute at one lane per invocation
        // where exact (DECISIONS D62), also for requests recorded before the flag existed;
        // APS5_MSL_WAVE64_LANE_LOCAL=0 replays the two-half layout.
        static const bool wave64LaneLocal = [] {
            const char* value = std::getenv("APS5_MSL_WAVE64_LANE_LOCAL");
            return value == nullptr || value[0] != '0';
        }();
        request.target.mslWave64LaneLocal = wave64LaneLocal;
        // DECISIONS D63, as the device: APS5_MSL_NAN_TESTS=1 keeps every NaN test.
        static const bool nanTestElimination = [] {
            const char* value = std::getenv("APS5_MSL_NAN_TESTS");
            return value == nullptr || value[0] != '1';
        }();
        request.target.mslNanTestElimination = nanTestElimination;
        // Debug aid: --replay-width replays every request as if recorded for that host subgroup width.
        if (replayWidth != 0u) request.target.subgroupSize = replayWidth;
        ShaderRecompiler::RecompileResult compiled;
        const auto recompileStarted = std::chrono::steady_clock::now();
        try {
            compiled = ShaderRecompiler::Recompile(request);
            result.recompileMs = Milliseconds(recompileStarted);
        } catch (const std::exception& error) {
            const std::string message = error.what();
            if (message.find("MSL emission failed") != std::string::npos) {
                result.outcome = Outcome::EmissionFailed;
                result.detail = FirstLines(message.substr(message.find("MSL emission failed")), 1);
                // The shader is in the corpus only when SpirvBackend accepts it: a request both
                // backends refuse (VertexBarrier's LDS case, host-width limits) is a rejection the
                // test that recorded it checks, not a MetalBackend gap.
                request.target.backend = ShaderRecompiler::ShaderBackend::Spirv;
                try {
                    static_cast<void>(ShaderRecompiler::Recompile(request));
                } catch (const std::exception& reference) {
                    result.outcome = Outcome::SpirvRejected;
                    result.detail += " (SpirvBackend: " + FirstLines(reference.what(), 1) + ")";
                    return;
                }
                // A request recorded for a host subgroup wider than the Metal device's (a 64-lane
                // Vulkan target) that SpirvBackend refuses on a host as wide as Metal's: the shader
                // needs a 64-lane graphics wave, which no 32-lane host forms (TechnicalDebt
                // metal-b6-half-wave). Reported, not counted.
                if (request.target.subgroupSize > hostSubgroupSize) {
                    request.target.subgroupSize = hostSubgroupSize;
                    try {
                        static_cast<void>(ShaderRecompiler::Recompile(request));
                    } catch (const std::exception& reference) {
                        result.outcome = Outcome::HostWidthRejected;
                        result.detail += " (SpirvBackend on a " + std::to_string(hostSubgroupSize) + "-lane host: " + FirstLines(reference.what(), 1) + ")";
                    }
                }
                return;
            }
            request.target.backend = ShaderRecompiler::ShaderBackend::Spirv;
            result.detail = FirstLines(message, 1);
            try {
                static_cast<void>(ShaderRecompiler::Recompile(request));
                result.outcome = Outcome::MslPathFailed;
            } catch (const std::exception& reference) {
                const auto spirv = FirstLines(reference.what(), 1);
                result.outcome = spirv == result.detail ? Outcome::FrontEndRejected : Outcome::MslPathFailed;
                if (spirv != result.detail) result.detail += " (SpirvBackend fails differently: " + spirv + ")";
            }
            return;
        }
        if (compiled.msl.empty()) {
            result.outcome = Outcome::MslPathFailed;
            result.detail = "the recompiler returned no MSL for a Metal target";
            return;
        }
        const auto& shader = compiled.msl.Shader();
        if (shader.reflection.abiVersion != ShaderRecompiler::MslAbi::Version) {
            result.outcome = Outcome::MslPathFailed;
            result.detail = "reflection reports MslAbi version " + std::to_string(shader.reflection.abiVersion);
            return;
        }
        result.mslBytes = shader.source.size();
        result.mslLines = static_cast<std::size_t>(std::count(shader.source.begin(), shader.source.end(), '\n'));
        if (!dumpMsl.empty()) {
            std::ofstream dump(dumpMsl / (file.stem().string() + ".metal"), std::ios::binary);
            dump << shader.source;
        }
        // Measurement mode (APS5_MSL_CORPUS_EMIT_ONLY=1): sizes only, no Metal compile.
        static const bool emitOnly = [] {
            const char* value = std::getenv("APS5_MSL_CORPUS_EMIT_ONLY");
            return value != nullptr && value[0] == '1';
        }();
        if (emitOnly) {
            result.outcome = Outcome::Emitted;
            return;
        }
        std::string source = shader.source;
        if (cold) {
            static std::atomic<std::uint64_t> nonce{0};
            source += "\n// aps5 corpus cold " + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "." + std::to_string(nonce.fetch_add(1)) + "\n";
        }
        MTL4LibraryDescriptor* libraryDescriptor = [MTL4LibraryDescriptor new];
        libraryDescriptor.source = [NSString stringWithUTF8String:source.c_str()];
        libraryDescriptor.options = options;
        // A vertex function that writes cull distances (MslAbi Version 10) compiles only for the draws
        // MetalDraw.mm can emulate them in: as the driver does for triangle topologies, compile it as a
        // triangle-vertices draw (TriangleVerticesMacro).
        if (shader.reflection.stage == ShaderRecompiler::MslAbi::Stage::Vertex && shader.reflection.cullDistances != 0u) {
            MTLCompileOptions* cull = [options copy];
            NSMutableDictionary* macros = [NSMutableDictionary dictionaryWithDictionary:cull.preprocessorMacros ?: @{}];
            macros[@(ShaderRecompiler::MslAbi::TriangleVerticesMacro)] = @"1";
            cull.preprocessorMacros = macros;
            libraryDescriptor.options = cull;
        }
        NSError* error = nil;
        const auto libraryStarted = std::chrono::steady_clock::now();
        id<MTLLibrary> library = [compiler newLibraryWithDescriptor:libraryDescriptor error:&error];
        result.libraryMs = Milliseconds(libraryStarted);
        const auto pipelineStarted = std::chrono::steady_clock::now();
        if (library == nil) {
            result.outcome = Outcome::LibraryFailed;
            result.detail = MetalError(Describe(error));
            return;
        }
        // Tessellation stage libraries (MslAbi Version 7) have no stage function: each defines
        // aps5_stage(...) for LinkMslTessellation, which wraps local + hull into an object function and
        // the domain program into the tessellator's mesh function. Valid here = the library compiles
        // with the device's options and defines aps5_stage; the linked pipelines are checked by
        // agc_metal_msl_tessellation (every domain, partitioning and topology) and
        // agc_metal_tessellation_domains (GPU).
        const auto stage = shader.reflection.stage;
        if (stage == ShaderRecompiler::MslAbi::Stage::Local || stage == ShaderRecompiler::MslAbi::Stage::Hull || stage == ShaderRecompiler::MslAbi::Stage::Domain) {
            const bool defined = shader.source.find(" aps5_stage(") != std::string::npos;
            result.outcome = defined ? Outcome::TessellationLibrary : Outcome::PipelineFailed;
            if (!defined) result.detail = "the tessellation stage library defines no aps5_stage";
            return;
        }
        MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
        function.name = @(ShaderRecompiler::MslAbi::EntryPoint);
        function.library = library;
        MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
        for (const auto& constant : compiled.specialization) {
            const std::uint32_t value = constant.value;
            [values setConstantValue:&value type:MTLDataTypeUInt atIndex:constant.id];
        }
        if (shader.reflection.stage != ShaderRecompiler::MslAbi::Stage::Kernel) {
            id<MTLFunction> specialized = [library newFunctionWithName:@(ShaderRecompiler::MslAbi::EntryPoint) constantValues:values error:&error];
            result.pipelineMs = Milliseconds(pipelineStarted);
            result.outcome = specialized != nil ? Outcome::FunctionOnly : Outcome::PipelineFailed;
            if (specialized == nil) result.detail = MetalError(Describe(error));
            return;
        }
        MTL4FunctionDescriptor* entry = function;
        if (!compiled.specialization.empty()) {
            MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
            specialized.functionDescriptor = function;
            specialized.constantValues = values;
            entry = specialized;
        }
        MTL4ComputePipelineDescriptor* pipelineDescriptor = [MTL4ComputePipelineDescriptor new];
        pipelineDescriptor.computeFunctionDescriptor = entry;
        id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:pipelineDescriptor compilerTaskOptions:nil error:&error];
        result.pipelineMs = Milliseconds(pipelineStarted);
        if (pipeline == nil) {
            result.outcome = Outcome::PipelineFailed;
            result.detail = MetalError(Describe(error));
            return;
        }
        const auto& threads = shader.reflection.threadsPerThreadgroup;
        const std::uint64_t total = std::uint64_t{threads[0]} * threads[1] * threads[2];
        if (pipeline.maxTotalThreadsPerThreadgroup < total) {
            result.outcome = Outcome::PipelineFailed;
            result.detail = "the pipeline runs at most " + std::to_string(pipeline.maxTotalThreadsPerThreadgroup) + " threads per threadgroup, the shader needs " + std::to_string(total);
            return;
        }
        result.outcome = Outcome::Compiled;
    }

    id<MTLDevice> device;
    std::uint32_t hostSubgroupSize;
    std::uint32_t replayWidth;
    bool cold;
    fs::path dumpMsl;
    id<MTL4Compiler> compiler;
    MTLCompileOptions* options;
};

std::vector<fs::path> CorpusFiles(const Options& options, std::vector<std::string>& notes) {
    std::map<std::string, fs::path> files;
    for (const auto& corpus : options.corpora) {
        std::error_code error;
        if (!fs::is_directory(corpus, error)) {
            notes.push_back("no corpus at " + corpus.string());
            continue;
        }
        std::size_t found = 0;
        for (const auto& entry : fs::directory_iterator(corpus, error)) {
            const auto name = entry.path().filename().string();
            const bool msl = name.ends_with(".msl.req");
            const bool other = name.ends_with(".req") && !msl;
            if (!entry.is_regular_file() || !(msl || (other && options.includeSpirv))) continue;
            if (!options.match.empty() && name.find(options.match) == std::string::npos) continue;
            files.emplace(name, entry.path());
            ++found;
        }
        notes.push_back(std::to_string(found) + " request(s) in " + corpus.string());
    }
    std::vector<fs::path> ordered;
    ordered.reserve(files.size());
    for (auto& [name, path] : files) ordered.push_back(std::move(path));
    if (options.sliceCount > 1) {
        std::vector<fs::path> slice;
        for (std::size_t index = options.sliceIndex; index < ordered.size(); index += options.sliceCount) slice.push_back(ordered[index]);
        ordered = std::move(slice);
    }
    if (options.limit != 0 && ordered.size() > options.limit) ordered.resize(options.limit);
    return ordered;
}

Options Parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (index + 1 >= argc) throw std::runtime_error(argument + " needs a value");
            return argv[++index];
        };
        if (argument == "--corpus") options.corpora.emplace_back(value());
        else if (argument == "--report") options.report = value();
        else if (argument == "--workers") options.workers = static_cast<unsigned>(std::stoul(value()));
        else if (argument == "--limit") options.limit = std::stoull(value());
        else if (argument == "--include-spirv") options.includeSpirv = true;
        else if (argument == "--verbose") options.verbose = true;
        else if (argument == "--replay-width") options.replayWidth = static_cast<std::uint32_t>(std::stoul(value()));
        else if (argument == "--cold") options.cold = true;
        else if (argument == "--dump-msl") options.dumpMsl = value();
        else if (argument == "--match") options.match = value();
        else throw std::runtime_error("unknown argument " + argument + " (--corpus DIR... --report FILE --workers N --limit N --include-spirv --verbose --replay-width LANES --cold --dump-msl DIR --match TEXT)");
    }
    // The same options from the environment, for runs through ctest/gpu_suite (measurement runs).
    auto environment = [](const char* name) -> std::string {
        const char* text = std::getenv(name);
        return text != nullptr ? text : "";
    };
    if (const auto report = environment("APS5_MSL_CORPUS_REPORT"); !report.empty()) options.report = report;
    if (const auto workers = environment("APS5_MSL_CORPUS_WORKERS"); !workers.empty()) options.workers = static_cast<unsigned>(std::stoul(workers));
    if (environment("APS5_MSL_CORPUS_COLD") == "1") options.cold = true;
    if (const auto dump = environment("APS5_MSL_CORPUS_DUMP_MSL"); !dump.empty()) options.dumpMsl = dump;
    if (const auto match = environment("APS5_MSL_CORPUS_MATCH"); !match.empty()) options.match = match;
    if (const auto slice = environment("APS5_MSL_CORPUS_SLICE"); !slice.empty()) {
        const auto slash = slice.find('/');
        if (slash == std::string::npos) throw std::runtime_error("APS5_MSL_CORPUS_SLICE must be i/n");
        options.sliceIndex = std::stoull(slice.substr(0, slash));
        options.sliceCount = std::stoull(slice.substr(slash + 1));
        if (options.sliceCount == 0 || options.sliceIndex >= options.sliceCount) throw std::runtime_error("APS5_MSL_CORPUS_SLICE: i must be below n");
    }
    if (!options.dumpMsl.empty()) fs::create_directories(options.dumpMsl);
    if (const char* list = std::getenv("APS5_MSL_CORPUS"); list != nullptr && *list != '\0') {
        std::stringstream stream(list);
        std::string item;
        while (std::getline(stream, item, ':')) {
            if (!item.empty()) options.corpora.emplace_back(item);
        }
    }
    if (options.corpora.empty()) throw std::runtime_error("no corpus: pass --corpus DIR or set APS5_MSL_CORPUS");
    return options;
}

}

// p50/p90/p99/max and totals of the size and cost metrics over the requests that reached the
// Metal compiler (sizes in bytes, times in ms).
void PrintDistributions(const std::vector<Result>& results, bool cold) {
    auto print = [&](const char* name, auto metric) {
        std::vector<double> values;
        for (const auto& result : results) {
            if (result.mslBytes != 0) values.push_back(static_cast<double>(metric(result)));
        }
        if (values.empty()) return;
        std::sort(values.begin(), values.end());
        double total = 0.0;
        for (const double value : values) total += value;
        auto at = [&](double fraction) { return values[std::min(values.size() - 1, static_cast<std::size_t>(fraction * static_cast<double>(values.size())))]; };
        std::printf("metric %-12s n %zu total %.0f p50 %.1f p90 %.1f p99 %.1f max %.1f\n", name, values.size(), total, at(0.5), at(0.9), at(0.99), values.back());
    };
    std::printf("metrics (%s compiler cache):\n", cold ? "cold: unique sources" : "warm or as found");
    print("msl_bytes", [](const Result& result) { return result.mslBytes; });
    print("msl_lines", [](const Result& result) { return result.mslLines; });
    print("recompile_ms", [](const Result& result) { return result.recompileMs; });
    print("library_ms", [](const Result& result) { return result.libraryMs; });
    print("pipeline_ms", [](const Result& result) { return result.pipelineMs; });
}

// Measurement mode (APS5_MSL_CORPUS_METAL_FILES=<dir>): compiles every <dir>/*.metal (MSL libraries
// the driver wrote with APS5_METAL_DUMP_MSL, macros as #defines) with the device's options and
// prints its size and library compile time; --cold applies. Returns nonzero when one fails.
int CompileMetalFiles(id<MTLDevice> metal, const fs::path& directory, bool cold) {
    NSError* error = nil;
    id<MTL4Compiler> compiler = [metal newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    if (compiler == nil) throw std::runtime_error("cannot create an MTL4Compiler: " + Describe(error));
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((ShaderRecompiler::MslAbi::LanguageMajor << 16u) | ShaderRecompiler::MslAbi::LanguageMinor);
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".metal") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    int failures = 0;
    for (const auto& file : files) {
        @autoreleasepool {
            std::string source = ReadFile(file);
            if (cold) source += "\n// aps5 corpus cold " + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "\n";
            MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
            descriptor.source = [NSString stringWithUTF8String:source.c_str()];
            descriptor.options = options;
            const auto started = std::chrono::steady_clock::now();
            id<MTLLibrary> library = [compiler newLibraryWithDescriptor:descriptor error:&error];
            const double ms = Milliseconds(started);
            std::printf("metal-file %s bytes %zu lines %zu library_ms %.1f %s\n", file.filename().c_str(), source.size(), static_cast<std::size_t>(std::count(source.begin(), source.end(), '\n')), ms, library != nil ? "compiled" : MetalError(Describe(error)).c_str());
            failures += library == nil ? 1 : 0;
        }
    }
    return failures == 0 ? 0 : 1;
}

// Measurement mode (APS5_MSL_CORPUS_PARALLEL=<threads> with APS5_MSL_CORPUS_METAL_FILES=<dir>): how
// many MSL library compiles run at once. Every .metal file is compiled `threads` times (each copy a
// unique source, cold) by `threads` threads, through (a) one shared MTL4Compiler, (b) one MTL4Compiler
// per thread, (c) MTLDevice newLibraryWithSource; prints the wall time of each and of one compile.
std::string MslHexLiteral(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llxu", static_cast<unsigned long long>(value) + 0x100u);
    return text;
}

int CompileParallel(id<MTLDevice> metal, const fs::path& directory, unsigned threads) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = static_cast<MTLLanguageVersion>((ShaderRecompiler::MslAbi::LanguageMajor << 16u) | ShaderRecompiler::MslAbi::LanguageMinor);
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    std::vector<std::string> sources;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".metal") sources.push_back(ReadFile(entry.path()));
    }
    if (sources.empty()) throw std::runtime_error("no .metal files");
    std::atomic<std::uint64_t> nonce{static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())};
    auto unique = [&](const std::string& source) { return source + "\n// aps5 parallel " + std::to_string(nonce.fetch_add(1)) + "\n"; };
    NSError* error = nil;
    id<MTL4Compiler> shared = [metal newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    std::vector<id<MTL4Compiler>> own;
    for (unsigned index = 0; index < threads; ++index) own.push_back([metal newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error]);
    auto compile = [&](int mode, unsigned thread, const std::string& source) {
        @autoreleasepool {
            NSError* failure = nil;
            id<MTLLibrary> library = nil;
            if (mode == 2) {
                library = [metal newLibraryWithSource:@(unique(source).c_str()) options:options error:&failure];
            } else {
                MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
                descriptor.source = @(unique(source).c_str());
                descriptor.options = options;
                library = [(mode == 0 ? shared : own[thread]) newLibraryWithDescriptor:descriptor error:&failure];
            }
            if (library == nil) throw std::runtime_error("compile failed: " + Describe(failure));
        }
    };
    // Kernels: the back-end compile (compute pipeline state) of `threads` distinct libraries, one
    // after another and then from `threads` threads (one shared compiler; one compiler per thread).
    for (const auto& source : sources) {
        if (source.find("kernel void aps5_main") == std::string::npos) continue;
        for (int mode = 0; mode < 2; ++mode) {
            std::vector<id<MTLLibrary>> libraries(threads);
            for (unsigned thread = 0; thread < threads; ++thread) {
                // A comment does not reach the back end: change the first literal of the body so
                // every pipeline compiles distinct code.
                std::string variant = unique(source);
                static const std::regex literal(R"(0x[0-9a-f]+u)");
                const auto body = variant.find("aps5_body(");
                std::smatch found;
                const std::string tail = variant.substr(body == std::string::npos ? 0 : body);
                if (std::regex_search(tail, found, literal)) variant.replace((body == std::string::npos ? 0 : body) + static_cast<std::size_t>(found.position(0)), static_cast<std::size_t>(found.length(0)), MslHexLiteral(nonce.fetch_add(1) & 0xffffu));
                MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
                descriptor.source = @(variant.c_str());
                descriptor.options = options;
                libraries[thread] = [shared newLibraryWithDescriptor:descriptor error:&error];
                if (libraries[thread] == nil) throw std::runtime_error("compile failed: " + Describe(error));
            }
            auto pipeline = [&](unsigned thread, id<MTL4Compiler> compiler) {
                @autoreleasepool {
                    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
                    function.name = @(ShaderRecompiler::MslAbi::EntryPoint);
                    function.library = libraries[thread];
                    // The kernel declares function constants (their defaults apply).
                    MTL4SpecializedFunctionDescriptor* specialized = [MTL4SpecializedFunctionDescriptor new];
                    specialized.functionDescriptor = function;
                    specialized.constantValues = [MTLFunctionConstantValues new];
                    MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
                    descriptor.computeFunctionDescriptor = specialized;
                    NSError* failure = nil;
                    if ([compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&failure] == nil) throw std::runtime_error("pipeline failed: " + Describe(failure));
                }
            };
            const auto first = std::chrono::steady_clock::now();
            pipeline(0, shared);
            const double one = Milliseconds(first);
            const auto started = std::chrono::steady_clock::now();
            std::vector<std::thread> pool;
            for (unsigned thread = 1; thread < threads; ++thread) pool.emplace_back([&, thread] { pipeline(thread, mode == 0 ? shared : own[thread]); });
            for (auto& thread : pool) thread.join();
            const double wall = Milliseconds(started);
            std::printf("parallel pipelines bytes %zu threads %u %-24s wall %.1f ms (one pipeline %.1f ms, speedup %.2fx)\n", source.size(), threads - 1u, mode == 0 ? "one MTL4Compiler" : "MTL4Compiler per thread", wall, one, one * (threads - 1u) / wall);
        }
    }
    for (const auto& source : sources) {
        const auto single = std::chrono::steady_clock::now();
        compile(1, 0, source);
        const double one = Milliseconds(single);
        for (int mode = 0; mode < 3; ++mode) {
            const auto started = std::chrono::steady_clock::now();
            std::vector<std::thread> pool;
            for (unsigned thread = 0; thread < threads; ++thread) pool.emplace_back([&, thread] { compile(mode, thread, source); });
            for (auto& thread : pool) thread.join();
            const double wall = Milliseconds(started);
            static const char* const Names[] = {"one MTL4Compiler", "MTL4Compiler per thread", "MTLDevice newLibraryWithSource"};
            std::printf("parallel bytes %zu threads %u %-32s wall %.1f ms (one compile %.1f ms, speedup %.2fx)\n", source.size(), threads, Names[mode], wall, one, one * threads / wall);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        const auto options = Parse(argc, argv);
        @autoreleasepool {
            id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
            if (metal == nil || ![metal supportsFamily:MTLGPUFamilyMetal4]) {
                std::puts("skipped, the device has no Metal 4 support");
                return 77;
            }
            if (const char* directory = std::getenv("APS5_MSL_CORPUS_METAL_FILES"); directory != nullptr && *directory != '\0') {
                if (const char* parallel = std::getenv("APS5_MSL_CORPUS_PARALLEL"); parallel != nullptr && *parallel != '\0') return CompileParallel(metal, directory, static_cast<unsigned>(std::stoul(parallel)));
                return CompileMetalFiles(metal, directory, options.cold);
            }
            const AgcDriver::VulkanDevice device{AgcDriver::VulkanDevice::CompileOnly{}};
            if (device.Target().backend != ShaderRecompiler::ShaderBackend::Msl) throw std::runtime_error("agc_metal_msl_corpus (T1): the Metal device target does not select the MSL backend");
            std::vector<std::string> notes;
            const auto files = CorpusFiles(options, notes);
            for (const auto& note : notes) std::printf("%s\n", note.c_str());
            if (files.empty()) throw std::runtime_error("agc_metal_msl_corpus (T1): the corpus is empty; collect one with tools/mac/gpu_suite.sh all --dump-requests build-metal/msl-corpus");
            const Checker checker(metal, device.Target().subgroupSize, options.replayWidth, options.cold, options.dumpMsl);
            std::vector<Result> results(files.size());
            std::atomic<std::size_t> next{0};
            const unsigned workers = std::max(1u, std::min<unsigned>(options.workers != 0 ? options.workers : std::min(8u, std::max(1u, std::thread::hardware_concurrency())), static_cast<unsigned>(files.size())));
            const auto started = std::chrono::steady_clock::now();
            std::vector<std::thread> threads;
            for (unsigned worker = 0; worker < workers; ++worker) {
                threads.emplace_back([&] {
                    for (std::size_t index = next.fetch_add(1); index < files.size(); index = next.fetch_add(1)) results[index] = checker.Check(files[index]);
                });
            }
            for (auto& thread : threads) thread.join();
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

            std::map<Outcome, std::size_t> counts;
            std::map<std::string, std::size_t> emission;
            static const std::regex opcode(R"(opcode=(\S+))");
            static const std::regex reason(R"(reason=(.*))");
            for (const auto& result : results) {
                ++counts[result.outcome];
                if (result.outcome != Outcome::EmissionFailed) continue;
                std::smatch found;
                if (std::regex_search(result.detail, found, opcode)) ++emission["opcode=" + found[1].str()];
                else if (std::regex_search(result.detail, found, reason)) ++emission["reason=" + found[1].str().substr(0, 100)];
                else ++emission[result.detail.substr(0, 100)];
            }
            std::size_t failing = 0;
            for (const auto& result : results) {
                if (!Counts(result.outcome) && result.outcome != Outcome::HostWidthRejected && !options.verbose) continue;
                failing += Counts(result.outcome) ? 1u : 0u;
                std::printf("%s %s %s: %s\n", result.file.filename().c_str(), result.stage.c_str(), OutcomeName(result.outcome), result.detail.c_str());
            }
            if (options.verbose) failing = static_cast<std::size_t>(std::count_if(results.begin(), results.end(), [](const Result& result) { return Counts(result.outcome); }));
            if (!emission.empty()) {
                std::vector<std::pair<std::string, std::size_t>> sorted(emission.begin(), emission.end());
                std::sort(sorted.begin(), sorted.end(), [](const auto& left, const auto& right) { return left.second != right.second ? left.second > right.second : left.first < right.first; });
                std::puts("MSL emission failures by opcode or reason:");
                for (const auto& [key, count] : sorted) std::printf("  %5zu  %s\n", count, key.c_str());
            }
            if (!options.report.empty()) {
                std::ofstream report(options.report);
                report << "request\tstage\toutcome\tseconds\tmsl_bytes\tmsl_lines\trecompile_ms\tlibrary_ms\tpipeline_ms\tdetail\n";
                for (const auto& result : results) {
                    report << result.file.filename().string() << '\t' << result.stage << '\t' << OutcomeName(result.outcome) << '\t' << result.seconds << '\t' << result.mslBytes << '\t' << result.mslLines << '\t' << result.recompileMs << '\t' << result.libraryMs << '\t' << result.pipelineMs << '\t' << result.detail << '\n';
                }
            }
            PrintDistributions(results, options.cold);
            std::printf("%zu request(s) in %.1f s with %u worker(s):", results.size(), seconds, workers);
            for (const auto& [outcome, count] : counts) std::printf(" %s %zu", OutcomeName(outcome), count);
            std::printf("\n");
            const std::size_t valid = counts[Outcome::Compiled] + counts[Outcome::Emitted] + counts[Outcome::FunctionOnly] + counts[Outcome::TessellationLibrary];
            const std::size_t applicable = results.size() - counts[Outcome::FrontEndRejected] - counts[Outcome::SpirvRejected] - counts[Outcome::HostWidthRejected];
            std::printf("MetalBackend produced valid MSL for %zu of %zu request(s) the front end accepts (%zu not valid)\n", valid, applicable, failing);
            if (failing != 0) return 1;
        }
        std::puts("msl corpus check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
