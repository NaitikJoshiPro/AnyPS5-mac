#include "BareGuest.hpp"

#include <FEXCore/Config/Config.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/fextl/string.h>

#include <dlfcn.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int Passed = 0;
constexpr int Failed = 1;
constexpr int HarnessError = 2;
constexpr int GuestFault = 3;
constexpr int Skipped = 125;
constexpr std::uint8_t Halt = 0xF4;

struct Expectation {
    std::string name;
    std::vector<std::uint64_t> values;
    std::vector<std::vector<std::uint64_t>> alternatives;
};

struct TestConfig {
    std::string configPath;
    std::string binaryPath;
    std::string test;
    std::optional<std::string> skip;
    std::optional<std::string> fail;
    std::vector<std::pair<std::string, std::string>> knownFailures;
    std::uint64_t code = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> maps;
    std::vector<std::pair<std::uint64_t, std::vector<std::uint8_t>>> memory;
    std::vector<std::pair<std::string, std::string>> options;
    std::vector<std::string> features;
    std::vector<Expectation> registers;
};

struct Arguments {
    std::string variant = "default";
    std::vector<std::pair<std::string, std::string>> options;
    unsigned timeout = 25;
    std::string configPath;
};

struct Outcome {
    bool passed;
    std::string detail;
    std::string log;
};

std::uint64_t Number(const std::string& text) {
    std::size_t used = 0;
    const std::uint64_t value = std::stoull(text, &used, 0);
    if (used != text.size()) throw std::runtime_error("not a number: " + text);
    return value;
}

std::string Rest(std::istringstream& stream) {
    std::string rest;
    std::getline(stream >> std::ws, rest);
    return rest;
}

std::vector<std::uint8_t> HexBytes(const std::string& text) {
    if (text.size() % 2 != 0) throw std::runtime_error("odd number of hex digits in memory data");
    std::vector<std::uint8_t> bytes;
    for (std::size_t index = 0; index < text.size(); index += 2)
        bytes.push_back(static_cast<std::uint8_t>(std::stoul(text.substr(index, 2), nullptr, 16)));
    return bytes;
}

TestConfig ReadConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path + "; is the fex_asm_prepare test run before it?");
    TestConfig config;
    config.configPath = path;
    config.binaryPath = path.size() > 4 && path.ends_with(".cfg") ? path.substr(0, path.size() - 4) + ".bin" : path + ".bin";
    std::string line;
    if (!std::getline(file, line) || line != "aps5-fex-asm 1") throw std::runtime_error(path + " is not an aps5-fex-asm 1 configuration");
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string key;
        stream >> key;
        if (key.empty() || key == "hash" || key == "relocations" || key == "instructions") continue;
        if (key == "test") {
            config.test = Rest(stream);
        } else if (key == "skip") {
            config.skip = Rest(stream);
        } else if (key == "fail") {
            config.fail = Rest(stream);
        } else if (key == "known_failure") {
            std::string variant;
            stream >> variant;
            config.knownFailures.emplace_back(variant, Rest(stream));
        } else if (key == "code") {
            std::string address;
            stream >> address;
            config.code = Number(address);
        } else if (key == "map") {
            std::string address, size;
            stream >> address >> size;
            config.maps.emplace_back(Number(address), Number(size));
        } else if (key == "memory") {
            std::string address, bytes;
            stream >> address >> bytes;
            config.memory.emplace_back(Number(address), HexBytes(bytes));
        } else if (key == "option") {
            std::string name;
            stream >> name;
            config.options.emplace_back(name, Rest(stream));
        } else if (key == "feature") {
            config.features.push_back(Rest(stream));
        } else if (key == "reg") {
            Expectation expectation;
            stream >> expectation.name;
            std::string value;
            while (stream >> value)
                expectation.values.push_back(Number(value));
            if (expectation.values.empty()) throw std::runtime_error("no value for register " + expectation.name);
            expectation.alternatives.resize(expectation.values.size());
            config.registers.push_back(std::move(expectation));
        } else if (key == "alt") {
            std::string name, lane, value;
            stream >> name >> lane >> value;
            auto found = std::find_if(config.registers.rbegin(), config.registers.rend(), [&](const auto& entry) { return entry.name == name; });
            if (found == config.registers.rend() || Number(lane) >= found->values.size())
                throw std::runtime_error("alternative for unknown register lane " + name + " " + lane);
            found->alternatives[Number(lane)].push_back(Number(value));
        } else {
            throw std::runtime_error(path + ": unknown entry " + key);
        }
    }
    return config;
}

std::optional<std::string> MissingFeature(const TestConfig& config) {
    const FEXCore::HostFeatures host = Aps5Fex::RunnerHostFeatures();
    const std::pair<std::string_view, bool> checked[] = {
        {"AVX", host.SupportsAVX}, {"RAND", host.SupportsRAND}, {"SHA", host.SupportsSHA}, {"CLZERO", host.SupportsCLZERO},
        {"AES256", host.SupportsAES256}, {"AFP", host.SupportsAFP},
        // FEX advertises AVX-VNNI only with AVX and I8MM or DotProd (CPUID.cpp).
        {"AVX_VNNI", host.SupportsAVX && (host.SupportsI8MM || host.SupportsDotProd)},
    };
    constexpr std::string_view emulated[] = {"3DNOW", "SSE4A", "BMI1", "BMI2", "CLWB", "LINUX", "SSSE3", "SSE4.1", "SSE4.2", "AES",
                                             "PCLMUL", "MOVBE", "ADX", "XSAVE", "RDPID", "CLFLOPT", "FSGSBASE", "EMMI"};
    for (const std::string& feature : config.features) {
        const auto found = std::find_if(std::begin(checked), std::end(checked), [&](const auto& entry) { return entry.first == feature; });
        if (found != std::end(checked)) {
            if (!found->second) return feature;
            continue;
        }
        if (std::find(std::begin(emulated), std::end(emulated), feature) == std::end(emulated))
            throw std::runtime_error("unknown HostFeatures entry " + feature);
    }
    return std::nullopt;
}

std::optional<std::pair<FEXCore::Config::ConfigOption, std::string>> ConfigEntry(std::string_view Key, std::string_view Value_View) {
    static constexpr std::pair<std::string_view, FEXCore::Config::ConfigOption> names[] = {
#define OPT_BASE(type, group, enum, json, default) {"FEX_" #enum, FEXCore::Config::CONFIG_##enum},
#include <FEXCore/Config/ConfigValues.inl>
    };
    std::optional<fextl::string> Value;
#define ENVLOADER
#include <FEXCore/Config/ConfigOptions.inl>
    for (const auto& [name, option] : names)
        if (name == Key) return std::pair {option, Value ? std::string(Value->c_str()) : std::string(Value_View)};
    return std::nullopt;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> Merged(std::vector<std::pair<std::uint64_t, std::uint64_t>> maps) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& [address, size] : maps)
        ranges.emplace_back(address, address + size);
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
    for (const auto& range : ranges) {
        if (!merged.empty() && range.first <= merged.back().second)
            merged.back().second = std::max(merged.back().second, range.second);
        else
            merged.push_back(range);
    }
    return merged;
}

std::size_t RegisterOffset(const std::string& name) {
    using FEXCore::Core::CPUState;
    using namespace FEXCore::X86State;
    constexpr std::pair<std::string_view, int> gprs[] = {
        {"RAX", REG_RAX}, {"RBX", REG_RBX}, {"RCX", REG_RCX}, {"RDX", REG_RDX}, {"RSI", REG_RSI}, {"RDI", REG_RDI},
        {"RBP", REG_RBP}, {"RSP", REG_RSP}, {"R8", REG_R8},   {"R9", REG_R9},   {"R10", REG_R10}, {"R11", REG_R11},
        {"R12", REG_R12}, {"R13", REG_R13}, {"R14", REG_R14}, {"R15", REG_R15},
    };
    if (name == "RIP") return offsetof(CPUState, rip);
    if (name == "FS") return offsetof(CPUState, fs_cached);
    if (name == "GS") return offsetof(CPUState, gs_cached);
    for (const auto& [gpr, index] : gprs)
        if (name == gpr) return offsetof(CPUState, gregs) + sizeof(std::uint64_t) * index;
    if (name.starts_with("XMM")) {
        const std::size_t index = std::stoul(name.substr(3));
        if (index < CPUState::NUM_XMMS) return offsetof(CPUState, xmm.avx.data) + sizeof(CPUState::xmm.avx.data[0]) * index;
    }
    if (name.starts_with("MM")) {
        const std::size_t index = std::stoul(name.substr(2));
        if (index < 8) return offsetof(CPUState, mm) + sizeof(CPUState::mm[0]) * index;
    }
    throw std::runtime_error("unknown register " + name);
}

std::string Hex(std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(value));
    return text;
}

std::vector<std::string> Mismatches(const TestConfig& config, const FEXCore::Core::CPUState& state) {
    std::vector<std::string> found;
    for (const Expectation& expectation : config.registers) {
        const std::size_t offset = RegisterOffset(expectation.name);
        if (offset + expectation.values.size() * sizeof(std::uint64_t) > sizeof(state))
            throw std::runtime_error("too many values for register " + expectation.name);
        for (std::size_t lane = 0; lane < expectation.values.size(); ++lane) {
            std::uint64_t actual = 0;
            std::memcpy(&actual, reinterpret_cast<const std::uint8_t*>(&state) + offset + lane * sizeof(std::uint64_t), sizeof(actual));
            const auto& alternatives = expectation.alternatives[lane];
            if (actual == expectation.values[lane] || std::find(alternatives.begin(), alternatives.end(), actual) != alternatives.end())
                continue;
            std::string message = expectation.name + "[" + std::to_string(lane) + "]: got " + Hex(actual) + ", expected " + Hex(expectation.values[lane]);
            for (const std::uint64_t alternative : alternatives)
                message += " or " + Hex(alternative);
            found.push_back(std::move(message));
        }
    }
    return found;
}

void OnGuestSignal(int signal, siginfo_t* info, void* context) {
    const auto* machine = static_cast<ucontext_t*>(context)->uc_mcontext;
    const std::uint64_t pc = arm_thread_state64_get_pc(machine->__ss);
    Dl_info symbol {};
    const bool named = dladdr(reinterpret_cast<const void*>(pc), &symbol) != 0 && symbol.dli_sname != nullptr;
    char text[512];
    const int length = std::snprintf(text, sizeof(text), "aps5-fex-asm: signal %d (%s) at host pc 0x%llx (%s+0x%llx), address 0x%llx, guest rip 0x%llx\n",
                                      signal, strsignal(signal), static_cast<unsigned long long>(pc), named ? symbol.dli_sname : "JIT code",
                                      static_cast<unsigned long long>(named ? pc - reinterpret_cast<std::uintptr_t>(symbol.dli_saddr) : 0),
                                      static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)),
                                      static_cast<unsigned long long>(Aps5Fex::BareGuestRip(pc)));
    if (length > 0) (void)write(STDOUT_FILENO, text, static_cast<std::size_t>(std::min(length, static_cast<int>(sizeof(text)) - 1)));
    _exit(GuestFault);
}

void InstallSignalHandlers() {
    struct sigaction action {};
    action.sa_sigaction = OnGuestSignal;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    for (const int signal : {SIGSEGV, SIGBUS, SIGILL, SIGTRAP, SIGFPE})
        sigaction(signal, &action, nullptr);
}

int RunChild(const TestConfig& config, const Arguments& arguments) {
    alarm(arguments.timeout);
    InstallSignalHandlers();
    const auto ranges = Merged(config.maps);
    for (const auto& [begin, end] : ranges) {
        mach_vm_address_t address = begin;
        const kern_return_t result = mach_vm_allocate(mach_task_self(), &address, end - begin, VM_FLAGS_FIXED);
        if (result != KERN_SUCCESS) {
            std::printf("aps5-fex-asm: cannot map [%s, %s): kern_return %d\n", Hex(begin).c_str(), Hex(end).c_str(), result);
            return HarnessError;
        }
    }
    const auto inside = [&](std::uint64_t address, std::size_t size) {
        return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return address >= range.first && address + size <= range.second; });
    };
    std::ifstream binary(config.binaryPath, std::ios::binary);
    const std::vector<char> code((std::istreambuf_iterator<char>(binary)), std::istreambuf_iterator<char>());
    if (!binary.good() && !binary.eof()) throw std::runtime_error("cannot read " + config.binaryPath);
    if (code.empty() || !inside(config.code, code.size())) throw std::runtime_error("the code of " + config.binaryPath + " is empty or outside the mapped memory");
    std::memcpy(reinterpret_cast<void*>(config.code), code.data(), code.size());
    for (const auto& [address, bytes] : config.memory) {
        if (!inside(address, bytes.size())) throw std::runtime_error("memory data at " + Hex(address) + " is outside the mapped memory");
        std::memcpy(reinterpret_cast<void*>(address), bytes.data(), bytes.size());
    }

    Aps5Fex::BareGuest guest;
    guest.rip = config.code;
    guest.executable = ranges;
    for (const auto* options : {&config.options, &arguments.options}) {
        for (const auto& [name, value] : *options) {
            if (auto entry = ConfigEntry(name, value))
                guest.config.push_back(std::move(*entry));
            else
                std::printf("aps5-fex-asm: ignoring unknown FEX option %s\n", name.c_str());
        }
    }
    Aps5Fex::BareGuestResult result;
    Aps5Fex::RunBareGuest(guest, result);
    const std::uint64_t rip = result.state.rip;
    if (!inside(rip, 1) || *reinterpret_cast<const std::uint8_t*>(rip) != Halt) {
        std::printf("aps5-fex-asm: the guest stopped at rip %s without reaching hlt (a guest fault)\n", Hex(rip).c_str());
        return Failed;
    }
    if (result.syscalls != 0) {
        std::printf("aps5-fex-asm: the guest made %u syscalls (the first was %llu); a bare guest has no syscall layer\n", result.syscalls,
                    static_cast<unsigned long long>(result.firstSyscall));
        return Failed;
    }
    const auto mismatches = Mismatches(config, result.state);
    for (const std::string& mismatch : mismatches)
        std::printf("aps5-fex-asm: %s\n", mismatch.c_str());
    return mismatches.empty() ? Passed : Failed;
}

std::string Findings(const std::string& log) {
    std::string findings;
    std::istringstream lines(log);
    for (std::string line; std::getline(lines, line);) {
        if (!line.starts_with("aps5-fex-asm: ")) continue;
        if (!findings.empty()) findings += "; ";
        findings += line.substr(14);
        if (findings.size() > 400) return findings.substr(0, 400) + "...";
    }
    return findings;
}

Outcome RunIsolated(const TestConfig& config, const Arguments& arguments) {
    int output[2];
    if (pipe(output) != 0) throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
    std::fflush(nullptr);
    const pid_t child = fork();
    if (child < 0) throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
    if (child == 0) {
        dup2(output[1], STDOUT_FILENO);
        dup2(output[1], STDERR_FILENO);
        close(output[0]);
        close(output[1]);
        int code = HarnessError;
        try {
            code = RunChild(config, arguments);
        } catch (const std::exception& error) {
            std::printf("aps5-fex-asm: %s\n", error.what());
        }
        std::fflush(nullptr);
        _exit(code);
    }
    close(output[1]);
    std::string log;
    char buffer[4096];
    for (ssize_t length; (length = read(output[0], buffer, sizeof(buffer))) != 0;) {
        if (length < 0) {
            if (errno == EINTR) continue;
            break;
        }
        log.append(buffer, static_cast<std::size_t>(length));
    }
    close(output[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error(std::string("waitpid failed: ") + std::strerror(errno));
    }
    const std::string findings = Findings(log);
    const auto with = [&](std::string what) { return findings.empty() ? what : what + ": " + findings; };
    if (WIFSIGNALED(status)) {
        if (WTERMSIG(status) == SIGALRM) return {false, with("timed out after " + std::to_string(arguments.timeout) + " s"), log};
        return {false, with(std::string("crashed with signal ") + std::to_string(WTERMSIG(status)) + " (" + strsignal(WTERMSIG(status)) + ")"),
                log};
    }
    switch (WEXITSTATUS(status)) {
    case Passed:
        return {true, "", log};
    case Failed:
        return {false, with("failed"), log};
    case GuestFault:
        return {false, with("faulted"), log};
    default:
        return {false, with("harness error " + std::to_string(WEXITSTATUS(status))), log};
    }
}

Arguments ParseArguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if ((argument == "--variant" || argument == "--option" || argument == "--timeout") && index + 1 >= argc)
            throw std::runtime_error(std::string(argument) + " needs a value");
        if (argument == "--variant") {
            arguments.variant = argv[++index];
        } else if (argument == "--option") {
            const std::string_view option = argv[++index];
            const auto equals = option.find('=');
            if (equals == std::string_view::npos) throw std::runtime_error("--option needs FEX_NAME=VALUE");
            arguments.options.emplace_back(std::string(option.substr(0, equals)), std::string(option.substr(equals + 1)));
        } else if (argument == "--timeout") {
            arguments.timeout = static_cast<unsigned>(Number(argv[++index]));
        } else if (arguments.configPath.empty() && !argument.starts_with("--")) {
            arguments.configPath = argument;
        } else {
            throw std::runtime_error("unexpected argument " + std::string(argument));
        }
    }
    if (arguments.configPath.empty()) throw std::runtime_error("usage: aps5-fex-asm [--variant NAME] [--option FEX_NAME=VALUE]... [--timeout S] TEST.cfg");
    return arguments;
}

}

int main(int argc, char** argv) {
    Arguments arguments;
    TestConfig config;
    try {
        arguments = ParseArguments(argc, argv);
        config = ReadConfig(arguments.configPath);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "aps5-fex-asm: %s\n", error.what());
        return HarnessError;
    }
    const std::string name = config.test + " [" + arguments.variant + "]";
    if (config.skip) {
        std::printf("aps5-fex-asm: SKIP %s: %s\n", name.c_str(), config.skip->c_str());
        return Skipped;
    }
    std::optional<std::string> known;
    for (const auto& [variant, reason] : config.knownFailures)
        if (variant == "*" || variant == arguments.variant) known = reason;
    Outcome outcome {false, ""};
    try {
        if (config.fail) {
            outcome = {false, *config.fail};
        } else if (const auto missing = MissingFeature(config)) {
            std::printf("aps5-fex-asm: SKIP %s: the runner's host features lack %s\n", name.c_str(), missing->c_str());
            return Skipped;
        } else {
            outcome = RunIsolated(config, arguments);
        }
    } catch (const std::exception& error) {
        outcome = {false, error.what()};
    }
    int code = Failed;
    if (known && outcome.passed) {
        std::printf("aps5-fex-asm: XPASS %s: passes but is a known failure (%s); remove it from the list\n", name.c_str(), known->c_str());
    } else if (known) {
        std::printf("aps5-fex-asm: XFAIL %s: %s; known failure: %s\n", name.c_str(), outcome.detail.c_str(), known->c_str());
        code = Passed;
    } else if (outcome.passed) {
        std::printf("aps5-fex-asm: PASS %s\n", name.c_str());
        return Passed;
    } else {
        std::printf("aps5-fex-asm: FAIL %s: %s\n", name.c_str(), outcome.detail.c_str());
    }
    std::fwrite(outcome.log.data(), 1, outcome.log.size(), stdout);
    return code;
}
