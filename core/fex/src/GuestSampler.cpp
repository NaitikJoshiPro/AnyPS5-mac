#include "GuestSampler.hpp"

#include "CodeCache.hpp"

#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Debug/InternalThreadState.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <signal.h>
#include <sys/ucontext.h>
#include <mutex>
#include <pthread.h>
#include <string_view>
#include <thread>
#include <tuple>

namespace Aps5Fex {

namespace {

using ThreadState = FEXCore::Core::InternalThreadState;

struct Registered {
    ThreadState* thread = nullptr;
    thread_act_t port = MACH_PORT_NULL;
    pthread_t pthread {};
    unsigned index = 0;
};

// One RIP of one thread over a report window.
struct Hits {
    std::uint64_t count = 0;
    bool inJit = false;
    std::uint64_t hostPc = 0;          // last sample
    std::uint64_t signalled = 0;       // samples caught entering a signal handler
    int signal = 0;                    // last such signal, its si_code and fault address
    int code = 0;
    std::uint64_t faultAddress = 0;
    std::uint32_t instruction = 0;
    std::array<std::uint64_t, 16> gpr {}; // guest registers of the last sample (X86State order: rax, rcx, rdx, rbx, rsp, ...)
};

struct Sampler {
    std::mutex mutex; // threads, stats, totals
    std::vector<Registered> threads;
    unsigned nextIndex = 0;
    std::map<std::tuple<unsigned, std::uint64_t, bool>, Hits> stats;
    std::map<unsigned, std::uint64_t> totals;
    std::map<unsigned, std::string> names; // threads seen in this window, by index
    FEXCore::Context::Context* context = nullptr;
    const GuestModules* modules = nullptr;
    std::function<std::string(std::uint64_t)> importName;
    bool started = false;
    FEXCore::SignalDelegatorConfig::SRAIndexMapping gprMapping {};
    unsigned periodMs = 0;
    unsigned reportSeconds = 10;
    unsigned top = 8;
    std::uint64_t windows = 0;
    std::chrono::steady_clock::time_point windowStart;
};

Sampler& Instance() {
    static auto* sampler = new Sampler; // never destroyed: the sampler thread and atexit outlive statics
    return *sampler;
}

unsigned EnvNumber(const char* name, unsigned fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == 0) return fallback;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != 0) {
        std::fprintf(stderr, "[fex-sample] %s=%s is not a number; using %u\n", name, value, fallback);
        return fallback;
    }
    return static_cast<unsigned>(parsed);
}

unsigned PeriodFromEnvironment() {
    static const unsigned period = EnvNumber("APS5_FEX_SAMPLE", 0);
    return period;
}

const char* RegisterName(std::size_t i) {
    static constexpr const char* names[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                              "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};
    return names[i];
}

// Snapshot of one thread, taken while it is suspended: plain loads and Mach reads only, no allocation, no locks.
struct Snapshot {
    std::uint64_t pc = 0;
    std::uint64_t rip = 0;
    std::array<std::uint64_t, 29> x {};
    std::array<std::uint64_t, 16> stateGpr {};
    // Set when the thread sat at the entry of libSystem's _sigtramp: the signal and the interrupted context.
    int signal = 0;
    int code = 0;
    std::uint64_t faultAddress = 0;
    std::uint64_t trampolinePc = 0;
    std::uint32_t instruction = 0; // at the interrupted pc
};

template <typename T>
bool ReadMemory(std::uint64_t address, T& out) {
    mach_vm_size_t read = 0;
    return mach_vm_read_overwrite(mach_task_self(), address, sizeof(T), reinterpret_cast<mach_vm_address_t>(&out), &read) == KERN_SUCCESS &&
           read == sizeof(T);
}

// libSystem's signal trampoline. dlsym does not find it (not exported), and dladdr must not run while a thread is
// suspended (it takes dyld's lock), so the sampler learns it from the first sample dladdr names _sigtramp+0.
std::atomic<std::uint64_t> sigtramp {0};

std::uint64_t SigtrampAddress() { return sigtramp.load(std::memory_order_relaxed); }

void LearnSigtramp(std::uint64_t pc) {
    if (SigtrampAddress() != 0 || pc == 0) return;
    Dl_info info {};
    if (dladdr(reinterpret_cast<const void*>(pc), &info) != 0 && info.dli_sname != nullptr && reinterpret_cast<std::uint64_t>(info.dli_saddr) == pc &&
        std::string_view(info.dli_sname) == "_sigtramp")
        sigtramp.store(pc, std::memory_order_relaxed);
}

// At _sigtramp's first instruction x2 is the signal, x3 the siginfo and x4 the ucontext (libplatform's
// _sigtramp(catcher, infostyle, sig, sinfo, uctx, token)). A thread that faults in a loop is almost always caught
// there, so report the interrupted context instead of the trampoline.
void UnwrapSignal(Snapshot& out) {
    if (out.pc == 0 || out.pc != SigtrampAddress()) return;
    siginfo_t info {};
    ucontext_t context {};
    __darwin_mcontext64 machine {};
    if (!ReadMemory(out.x[3], info) || !ReadMemory(out.x[4], context) ||
        !ReadMemory(reinterpret_cast<std::uint64_t>(context.uc_mcontext), machine))
        return;
    out.signal = static_cast<int>(out.x[2]);
    out.code = info.si_code;
    out.faultAddress = reinterpret_cast<std::uint64_t>(info.si_addr);
    out.trampolinePc = out.pc;
    out.pc = arm_thread_state64_get_pc(machine.__ss);
    for (std::size_t i = 0; i < out.x.size(); ++i) out.x[i] = machine.__ss.__x[i];
    (void)ReadMemory(out.pc, out.instruction);
}

bool Take(const Registered& entry, FEXCore::Context::Context* context, Snapshot& out) {
    if (thread_suspend(entry.port) != KERN_SUCCESS) return false;
    arm_thread_state64_t machine {};
    mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
    const bool ok = thread_get_state(entry.port, ARM_THREAD_STATE64, reinterpret_cast<thread_state_t>(&machine), &count) == KERN_SUCCESS;
    if (ok) {
        out.pc = arm_thread_state64_get_pc(machine);
        for (std::size_t i = 0; i < out.x.size(); ++i) out.x[i] = machine.__x[i];
        UnwrapSignal(out);
        std::memcpy(out.stateGpr.data(), entry.thread->CurrentFrame->State.gregs, sizeof(out.stateGpr));
        // Walks the RIP entries of the block the thread is in (FEX writes InlineJITBlockHeader at block entry);
        // outside that block it returns the RIP FEX stored when it left JIT code.
        out.rip = context->RestoreRIPFromHostPC(entry.thread, out.pc);
    }
    thread_resume(entry.port);
    return ok;
}

void SampleOnce(Sampler& sampler) {
    std::lock_guard lock(sampler.mutex);
    if (sampler.context == nullptr) return;
    for (const auto& entry : sampler.threads) {
        Snapshot snapshot;
        if (!Take(entry, sampler.context, snapshot)) continue;
        const bool inJit = sampler.context->IsAddressInCodeBuffer(entry.thread, snapshot.pc);
        if (!inJit && snapshot.signal == 0) LearnSigtramp(snapshot.pc);
        auto& hits = sampler.stats[{entry.index, snapshot.rip, inJit}];
        ++hits.count;
        hits.inJit = inJit;
        hits.hostPc = snapshot.pc;
        hits.signal = snapshot.signal;
        hits.code = snapshot.code;
        hits.faultAddress = snapshot.faultAddress;
        hits.instruction = snapshot.instruction;
        if (snapshot.signal != 0) ++hits.signalled;
        for (std::size_t i = 0; i < hits.gpr.size(); ++i) {
            const unsigned host = sampler.gprMapping[i];
            // In JIT code the guest registers live in the static register allocation; outside it FEX spilled them.
            hits.gpr[i] = inJit && host < snapshot.x.size() ? snapshot.x[host] : snapshot.stateGpr[i];
        }
        ++sampler.totals[entry.index];
        if (!sampler.names.contains(entry.index)) {
            char name[64] {};
            pthread_getname_np(entry.pthread, name, sizeof(name));
            sampler.names[entry.index] = name;
        }
    }
}

void Run() {
    auto& sampler = Instance();
    sampler.windowStart = std::chrono::steady_clock::now();
    auto nextReport = sampler.windowStart + std::chrono::seconds(sampler.reportSeconds);
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sampler.periodMs));
        SampleOnce(sampler);
        if (std::chrono::steady_clock::now() >= nextReport) {
            GuestSampler::Report(stderr);
            nextReport += std::chrono::seconds(sampler.reportSeconds);
        }
    }
}

void ReportAtExit() { GuestSampler::Report(stderr); }

}

bool GuestSampler::Enabled() { return PeriodFromEnvironment() != 0; }

GuestSampler::Location GuestSampler::Locate(const GuestModules* modules, std::uint64_t rip) {
    if (modules != nullptr)
        for (const auto& module : modules->Modules())
            if (rip >= module.start && rip - module.start < module.size) {
                const auto slash = module.path.find_last_of('/');
                return {slash == std::string::npos ? module.path : module.path.substr(slash + 1), rip - module.start};
            }
    return {"?", rip};
}

void GuestSampler::Start(FEXCore::Context::Context* context, const GuestModules* modules,
                         const FEXCore::SignalDelegatorConfig::SRAIndexMapping& gprMapping,
                         std::function<std::string(std::uint64_t)> importName) {
    if (!Enabled()) return;
    auto& sampler = Instance();
    {
        std::lock_guard lock(sampler.mutex);
        sampler.context = context;
        sampler.modules = modules;
        sampler.importName = std::move(importName);
        sampler.gprMapping = gprMapping;
        if (sampler.started) return;
        sampler.started = true;
        sampler.periodMs = PeriodFromEnvironment();
        sampler.reportSeconds = std::max(1u, EnvNumber("APS5_FEX_SAMPLE_REPORT", 10));
        sampler.top = std::max(1u, EnvNumber("APS5_FEX_SAMPLE_TOP", 8));
    }
    std::fprintf(stderr, "[fex-sample] sampling guest threads every %u ms, report every %u s\n", sampler.periodMs, sampler.reportSeconds);
    std::atexit(ReportAtExit);
    std::thread(Run).detach();
}

void GuestSampler::Stop(FEXCore::Context::Context* context) {
    if (!Enabled()) return;
    auto& sampler = Instance();
    std::lock_guard lock(sampler.mutex);
    if (sampler.context != context) return;
    sampler.context = nullptr;
    sampler.modules = nullptr;
    sampler.importName = nullptr;
}

void GuestSampler::Register(ThreadState* thread) {
    if (!Enabled()) return;
    auto& sampler = Instance();
    std::lock_guard lock(sampler.mutex);
    sampler.threads.push_back({thread, pthread_mach_thread_np(pthread_self()), pthread_self(), sampler.nextIndex++});
}

void GuestSampler::Unregister(ThreadState* thread) {
    if (!Enabled()) return;
    auto& sampler = Instance();
    std::lock_guard lock(sampler.mutex);
    std::erase_if(sampler.threads, [thread](const Registered& entry) { return entry.thread == thread; });
}

void GuestSampler::Report(std::FILE* out) {
    auto& sampler = Instance();
    std::lock_guard lock(sampler.mutex);
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - sampler.windowStart).count();
    std::fprintf(out, "[fex-sample] window %" PRIu64 ": %.1f s, %zu guest threads registered\n", sampler.windows++, seconds,
                 sampler.threads.size());
    for (const auto& [index, total] : sampler.totals) {
        std::vector<std::pair<std::tuple<unsigned, std::uint64_t, bool>, Hits>> rows;
        for (const auto& [key, hits] : sampler.stats)
            if (std::get<0>(key) == index) rows.emplace_back(key, hits);
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
        std::fprintf(out, "[fex-sample] thread %u \"%s\": %" PRIu64 " samples, %zu distinct RIPs\n", index, sampler.names[index].c_str(), total,
                     rows.size());
        for (std::size_t i = 0; i < rows.size() && i < sampler.top; ++i) {
            const auto& [key, hits] = rows[i];
            const std::uint64_t rip = std::get<1>(key);
            auto location = Locate(sampler.modules, rip);
            if (location.module == "?" && sampler.importName)
                if (auto name = sampler.importName(rip); !name.empty()) location = {"import:" + name, 0};
            std::fprintf(out, "[fex-sample]   %5.1f%% rip 0x%" PRIx64 " %s+0x%" PRIx64, 100.0 * double(hits.count) / double(total), rip,
                         location.module.c_str(), location.offset);
            if (hits.inJit) {
                std::fprintf(out, " jit\n");
            } else {
                Dl_info info {};
                if (dladdr(reinterpret_cast<const void*>(hits.hostPc), &info) != 0 && info.dli_fname != nullptr) {
                    const std::string_view image(info.dli_fname);
                    std::fprintf(out, " host %s`%s+0x%" PRIx64 "\n", std::string(image.substr(image.find_last_of('/') + 1)).c_str(),
                                 info.dli_sname != nullptr ? info.dli_sname : "?",
                                 hits.hostPc - reinterpret_cast<std::uint64_t>(info.dli_saddr != nullptr ? info.dli_saddr : info.dli_fbase));
                } else {
                    std::fprintf(out, " host 0x%" PRIx64 "\n", hits.hostPc);
                }
            }
            if (hits.signalled != 0)
                std::fprintf(out, "[fex-sample]          %" PRIu64 " of these samples entered a signal handler: last signal %d code %d addr 0x%" PRIx64
                             " at host instruction 0x%08x\n", hits.signalled, hits.signal, hits.code, hits.faultAddress, hits.instruction);
            std::fprintf(out, "[fex-sample]          ");
            for (std::size_t r = 0; r < hits.gpr.size(); ++r)
                std::fprintf(out, "%s=%" PRIx64 "%s", RegisterName(r), hits.gpr[r], r == 7 ? "\n[fex-sample]          " : " ");
            std::fprintf(out, "\n");
        }
    }
    std::fflush(out);
    sampler.stats.clear();
    sampler.totals.clear();
    sampler.names.clear();
    sampler.windowStart = now;
}

}
