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
    return outcome != Outcome::Compiled && outcome != Outcome::FunctionOnly && outcome != Outcome::TessellationLibrary && outcome != Outcome::FrontEndRejected && outcome != Outcome::SpirvRejected && outcome != Outcome::HostWidthRejected;
}

struct Result {
    fs::path file;
    Outcome outcome = Outcome::Unreadable;
    std::string stage;
    std::string detail;
    double seconds = 0.0;
};

struct Options {
    std::vector<fs::path> corpora;
    fs::path report;
    unsigned workers = 0;
    std::size_t limit = 0;
    bool includeSpirv = false;
    bool verbose = false;
    std::uint32_t replayWidth = 0;
};

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
    Checker(id<MTLDevice> metal, std::uint32_t hostSubgroupSize, std::uint32_t replayWidth) : device(metal), hostSubgroupSize(hostSubgroupSize), replayWidth(replayWidth) {
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
        // Debug aid: --replay-width replays every request as if recorded for that host subgroup width.
        if (replayWidth != 0u) request.target.subgroupSize = replayWidth;
        ShaderRecompiler::RecompileResult compiled;
        try {
            compiled = ShaderRecompiler::Recompile(request);
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
        MTL4LibraryDescriptor* libraryDescriptor = [MTL4LibraryDescriptor new];
        libraryDescriptor.source = [NSString stringWithUTF8String:shader.source.c_str()];
        libraryDescriptor.options = options;
        NSError* error = nil;
        id<MTLLibrary> library = [compiler newLibraryWithDescriptor:libraryDescriptor error:&error];
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
            files.emplace(name, entry.path());
            ++found;
        }
        notes.push_back(std::to_string(found) + " request(s) in " + corpus.string());
    }
    std::vector<fs::path> ordered;
    ordered.reserve(files.size());
    for (auto& [name, path] : files) ordered.push_back(std::move(path));
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
        else throw std::runtime_error("unknown argument " + argument + " (--corpus DIR... --report FILE --workers N --limit N --include-spirv --verbose --replay-width LANES)");
    }
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

int main(int argc, char** argv) {
    try {
        const auto options = Parse(argc, argv);
        @autoreleasepool {
            id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
            if (metal == nil || ![metal supportsFamily:MTLGPUFamilyMetal4]) {
                std::puts("skipped, the device has no Metal 4 support");
                return 77;
            }
            const AgcDriver::VulkanDevice device{AgcDriver::VulkanDevice::CompileOnly{}};
            if (device.Target().backend != ShaderRecompiler::ShaderBackend::Msl) throw std::runtime_error("agc_metal_msl_corpus (T1): the Metal device target does not select the MSL backend");
            std::vector<std::string> notes;
            const auto files = CorpusFiles(options, notes);
            for (const auto& note : notes) std::printf("%s\n", note.c_str());
            if (files.empty()) throw std::runtime_error("agc_metal_msl_corpus (T1): the corpus is empty; collect one with tools/mac/gpu_suite.sh all --dump-requests build-metal/msl-corpus");
            const Checker checker(metal, device.Target().subgroupSize, options.replayWidth);
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
                report << "request\tstage\toutcome\tseconds\tdetail\n";
                for (const auto& result : results) report << result.file.filename().string() << '\t' << result.stage << '\t' << OutcomeName(result.outcome) << '\t' << result.seconds << '\t' << result.detail << '\n';
            }
            std::printf("%zu request(s) in %.1f s with %u worker(s):", results.size(), seconds, workers);
            for (const auto& [outcome, count] : counts) std::printf(" %s %zu", OutcomeName(outcome), count);
            std::printf("\n");
            const std::size_t valid = counts[Outcome::Compiled] + counts[Outcome::FunctionOnly] + counts[Outcome::TessellationLibrary];
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
