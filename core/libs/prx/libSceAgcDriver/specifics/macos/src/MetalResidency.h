#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALRESIDENCY_H
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_SPECIFICS_MACOS_SRC_METALRESIDENCY_H

// Residency-set changes and queue commits, serialized (DECISIONS D44).
//
// A Metal 4 queue commit walks every residency set attached to the queue (under the debug layer:
// -[MTL4GPUDebugCommandBuffer applyResidencySets:], which type-checks each allocation). The
// driver changes those sets from other threads: the recorder's release thread destroys a batch's
// kept objects (a retired host import removes its no-copy buffer from the import set and releases
// it, GuestBufferMemory.mm), the queue worker records and commits. A buffer removed and deallocated
// while a commit walks its set is a freed object in the walk: objc_opt_isKindOfClass traps
// (EXC_BREAKPOINT) under validation (SuperTuxKart, aps5-fex-2026-10-14-065055.ips; the concurrent
// -[MTLGPUDebugBuffer dealloc] under ~RetiredImport on the release thread in
// aps5-fex-2026-10-14-073953.ips).
//
// So every residency-set change goes through this header: Add/Remove/RemoveAll/Commit take one
// process-wide lock, which the queue commits (CommitQueue) hold too, and Remove/RemoveAll keep the
// removed allocations alive (the "graveyard") until the next queue commit has returned, so no
// allocation that was in a set dies while a commit could still be walking that set. Drain() frees
// the graveyard without a commit (device teardown, after the queue is idle).
//
// (A residency set itself keeps its allocations alive; the hazard is only the concurrent walk.
// Reproduced standalone: one thread committing, another adding/removing/releasing buffers in an
// attached set, crashes in applyResidencySets within seconds under MTL_DEBUG_LAYER=1
// MTL_SHADER_VALIDATION=1, and runs clean with both under one mutex; test agc_metal_device_residency.)
// Objective-C++ with ARC only, header-only (the device, recorder and test twins all include it).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <mutex>
#include <vector>

#if !__has_feature(objc_arc)
#error "MetalResidency.h needs ARC (-fobjc-arc): the graveyard holds strong references"
#endif

namespace AgcDriver::Metal::Residency {

struct State {
    // Residency-set changes and queue commits. Recursive: a commit's drain may release objects
    // whose teardown changes a set again.
    std::recursive_mutex lock;
    // Allocations removed from a set since the last queue commit (strong references).
    std::vector<id> graveyard;
    std::uint64_t buried = 0;
    std::uint64_t drained = 0;
    std::uint64_t queueCommits = 0;
};

inline State& Shared() {
    static State* const state = new State;
    return *state;
}

struct Counters {
    std::size_t graveyard = 0;   // allocations waiting for the next queue commit
    std::uint64_t buried = 0;    // allocations ever kept past their removal
    std::uint64_t drained = 0;   // allocations released after a commit or Drain
    std::uint64_t queueCommits = 0;
};

inline Counters Snapshot() {
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    return {state.graveyard.size(), state.buried, state.drained, state.queueCommits};
}

namespace Detail {

// Releases the graveyard; the caller holds state.lock.
inline void DrainLocked(State& state) {
    std::vector<id> released;
    released.swap(state.graveyard);
    state.drained += released.size();
    released.clear();
}

} // namespace Detail

// [set addAllocation:allocation].
inline void Add(id<MTLResidencySet> set, id<MTLAllocation> allocation) {
    if (set == nil || allocation == nil) return;
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    [set addAllocation:allocation];
}

// [set removeAllocation:allocation]; the allocation stays alive until the next queue commit returned.
inline void Remove(id<MTLResidencySet> set, id<MTLAllocation> allocation) {
    if (set == nil || allocation == nil) return;
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    [set removeAllocation:allocation];
    state.graveyard.push_back(allocation);
    ++state.buried;
}

// [set removeAllAllocations]; the allocations stay alive until the next queue commit returned.
inline void RemoveAll(id<MTLResidencySet> set) {
    if (set == nil) return;
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    NSArray<id<MTLAllocation>>* all = set.allAllocations;
    [set removeAllAllocations];
    for (id<MTLAllocation> allocation in all) {
        state.graveyard.push_back(allocation);
        ++state.buried;
    }
}

// [queue addResidencySet:set] / [queue removeResidencySet:set], ordered with the commits.
inline void Attach(id<MTL4CommandQueue> queue, id<MTLResidencySet> set) {
    if (queue == nil || set == nil) return;
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    [queue addResidencySet:set];
}

inline void Detach(id<MTL4CommandQueue> queue, id<MTLResidencySet> set) {
    if (queue == nil || set == nil) return;
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    [queue removeResidencySet:set];
}

// [set commit].
inline void Commit(id<MTLResidencySet> set) {
    if (set == nil) return;
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    [set commit];
}

// A queue commit (commit plus whatever must stay ordered with it, e.g. the timeline signal): runs
// under the residency lock, then releases the allocations removed before it.
template<typename Fn>
void CommitQueue(Fn&& commit) {
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    commit();
    ++state.queueCommits;
    Detail::DrainLocked(state);
}

// Releases the graveyard without a commit: only when no queue commit can be walking the sets
// (the device's queue is idle or its sets are already removed from it).
inline void Drain() {
    auto& state = Shared();
    std::lock_guard lock(state.lock);
    Detail::DrainLocked(state);
}

} // namespace AgcDriver::Metal::Residency

#endif
