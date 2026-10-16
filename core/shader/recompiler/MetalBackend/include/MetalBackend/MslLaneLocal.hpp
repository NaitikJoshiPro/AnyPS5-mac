#ifndef CORE_SHADER_RECOMPILER_METALBACKEND_MSLLANELOCAL_HPP
#define CORE_SHADER_RECOMPILER_METALBACKEND_MSLLANELOCAL_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <cstdint>
#include <string>
#include <unordered_set>

namespace ShaderRecompiler {

struct MslLaneLocalPlan {
    bool eligible = false;
    std::string blocker;
    // Wave-mask branches decided by each lane's own condition.
    std::uint32_t laneMaskBranches = 0;
    // A one-wave threadgroup (64 invocations) whose program needs the other lanes of its wave:
    // ballots read beyond the lane's own bit (fullBallots), v_readlane/v_readfirstlane, or
    // wave-mask branches that cannot be decided per lane (votedBranches: block ids) exchange
    // values through threadgroup memory at threadgroup barriers.
    bool threadgroupExchange = false;
    bool fullBallots = false;
    std::unordered_set<std::uint32_t> votedBranches;
};

// The largest workgroup laid out at one lane per invocation: it has twice the host invocations of the
// two-half layout, and a pipeline's maxTotalThreadsPerThreadgroup falls with its register use
// (TechnicalDebt metal-wave64-lane-local).
inline constexpr std::uint32_t MslLaneLocalMaxThreads = 256u;

// Whether a wave64 compute program of `threads` invocations per workgroup can run at one guest lane
// per host invocation (MslLaneLocal.cpp has the conditions and the proof sketch). Measurement aid:
// APS5_MSL_LANE_SURVEY=<file> appends the decision and its blocker per program (and MslEmitter the
// library size).
[[nodiscard]] MslLaneLocalPlan MslPlanWave64LaneLocal(const IrProgram& program, std::uint32_t threads);

}

#endif
