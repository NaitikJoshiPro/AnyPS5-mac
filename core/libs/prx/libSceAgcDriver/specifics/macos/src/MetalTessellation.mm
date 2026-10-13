// Tessellated draws on Metal (DECISIONS D24, MetalTessellation.h): the linked object and mesh
// libraries, cached per stage combination. The cache holds each program's MSL (SharedMsl) so a
// key's shader addresses stay unique while the entry lives.

#include "MetalTessellation.h"
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace AgcDriver::Graphics::Metal {

namespace {

struct Linked {
    ShaderRecompiler::SharedMsl local;
    ShaderRecompiler::SharedMsl hull;
    ShaderRecompiler::SharedMsl domain;
    std::shared_ptr<const ShaderRecompiler::MslTessellationPipeline> pipeline;
};

using Key = std::tuple<const void*, const void*, const void*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;

ShaderRecompiler::MslTessellationProgram Program(const Graphics::CompiledShader& stage) {
    if (stage.program == nullptr || stage.program->msl.empty()) throw std::runtime_error("Metal device: a tessellation stage has no MSL form");
    ShaderRecompiler::MslTessellationProgram program;
    program.shader = &stage.program->msl.Shader();
    for (const auto& constant : stage.program->specialization) program.constants.emplace_back(constant.id, constant.value);
    return program;
}

}

std::shared_ptr<const ShaderRecompiler::MslTessellationPipeline> LinkTessellation(const TessellationStages& stages, const ShaderRecompiler::TessellationConfiguration& config) {
    if (stages.local == nullptr || stages.hull == nullptr || stages.domain == nullptr) throw std::runtime_error("Metal device: a tessellated draw needs local, hull and domain stages");
    static std::mutex mutex;
    static std::map<Key, Linked> cache;
    const Key key{&stages.local->program->msl.Shader(), &stages.hull->program->msl.Shader(), &stages.domain->program->msl.Shader(), config.inputControlPoints, config.outputControlPoints, config.domain, config.partitioning, config.outputTopology};
    std::lock_guard lock(mutex);
    if (const auto found = cache.find(key); found != cache.end()) return found->second.pipeline;
    const auto local = Program(*stages.local);
    const auto hull = Program(*stages.hull);
    const auto domain = Program(*stages.domain);
    auto pipeline = std::make_shared<const ShaderRecompiler::MslTessellationPipeline>(ShaderRecompiler::LinkMslTessellation(local, hull, domain, config));
    // Bounded: tessellated pipelines are few; a title that cycles more starts over.
    if (cache.size() >= 256u) cache.clear();
    cache.emplace(key, Linked{stages.local->program->msl, stages.hull->program->msl, stages.domain->program->msl, pipeline});
    return pipeline;
}

}
