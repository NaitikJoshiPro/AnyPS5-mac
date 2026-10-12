// agc_metal_request_dump: APS5_DUMP_REQUESTS=<dir> writes every recompile request once, named by a hash of its
// serialized form and the target backend, including requests the recompiler rejects; each file deserializes back
// into the request it came from. This is the collection half of agc_metal_msl_corpus. No GPU work is submitted.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("agc_metal_request_dump (T1): " + message);
}

alignas(256) constexpr std::array<std::uint32_t, 17> Vop1ControlCode{
    0x34020082, 0xe0302000, 0x80000401, 0xbf8c3f70, 0x7e000000, 0x7e003600, 0x7e008200, 0x4a080881,
    0xd5800000, 0x00000000, 0xd59b0000, 0x00000000, 0xd5c10000, 0x00000000, 0xe0702000, 0x80010401, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 1> UnterminatedCode{0x7e000000};

struct Built {
    std::vector<std::uint32_t> userData;
    std::array<ShaderRecompiler::MemoryRegion, 1> memory;
    ShaderRecompiler::RecompileRequest request;
};

Built MakeRequest(const AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint32_t buffers, std::span<std::uint32_t> backing) {
    Built built;
    const auto address = reinterpret_cast<std::uintptr_t>(backing.data());
    for (std::uint32_t buffer = 0; buffer < buffers; ++buffer) {
        built.userData.insert(built.userData.end(), {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), static_cast<std::uint32_t>(backing.size()), 0x01016facu});
    }
    built.memory = {{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{64u, 1, 1}, 0, {false, false, false}, false, 1};
    built.request = ShaderRecompiler::RecompileRequest{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, built.userData, compute, std::nullopt, std::nullopt, built.memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    built.request.useCache = false;
    return built;
}

bool Recompiles(const ShaderRecompiler::RecompileRequest& request) {
    try {
        static_cast<void>(ShaderRecompiler::Recompile(request));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string ReadFile(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

struct Dumped {
    fs::path path;
    std::string backend;
    ShaderRecompiler::DeserializedRequest replay;
};

std::vector<Dumped> ReadDumps(const fs::path& directory) {
    std::vector<Dumped> found;
    for (const auto& entry : fs::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        Require(!name.ends_with(".tmp"), "a temporary file was left behind: " + name);
        Require(name.ends_with(".msl.req") || name.ends_with(".spirv.req"), "unexpected file in the dump directory: " + name);
        const auto text = ReadFile(entry.path());
        auto replay = ShaderRecompiler::RequestSerializer{}.Deserialize(text);
        Require(ShaderRecompiler::RequestSerializer{}.Serialize(replay.request) == text, name + " does not serialize back to the same text");
        found.push_back({entry.path(), name.ends_with(".msl.req") ? "msl" : "spirv", std::move(replay)});
    }
    return found;
}

std::size_t Count(const std::vector<Dumped>& dumps, const std::string& backend, std::span<const std::uint32_t> code, std::size_t userDataWords) {
    std::size_t count = 0;
    for (const auto& dump : dumps) {
        const auto& request = dump.replay.request;
        if (dump.backend == backend && std::equal(request.shader.code.begin(), request.shader.code.end(), code.begin(), code.end()) && request.context.userData.size() == userDataWords) ++count;
    }
    return count;
}

}

int main() {
    const fs::path directory = fs::temp_directory_path() / ("aps5-request-dump-" + std::to_string(getpid()));
    try {
        @autoreleasepool {
            id<MTLDevice> metal = MTLCreateSystemDefaultDevice();
            if (metal == nil || ![metal supportsFamily:MTLGPUFamilyMetal4]) {
                std::puts("skipped, the device has no Metal 4 support");
                return 77;
            }
            fs::remove_all(directory);
            fs::create_directories(directory);
            Require(setenv("APS5_DUMP_REQUESTS", directory.c_str(), 1) == 0, "cannot set APS5_DUMP_REQUESTS");
            const AgcDriver::VulkanDevice device{AgcDriver::VulkanDevice::CompileOnly{}};
            Require(device.Target().backend == ShaderRecompiler::ShaderBackend::Msl, "the Metal device target does not select the MSL backend");
            std::vector<std::uint32_t> backing(256u, 0u);

            auto twoBuffers = MakeRequest(device, Vop1ControlCode, 2u, backing);
            Require(Recompiles(twoBuffers.request), "the vop1_control program does not recompile for the Metal target");
            Require(Recompiles(twoBuffers.request), "the vop1_control program does not recompile a second time");
            auto oneBuffer = MakeRequest(device, Vop1ControlCode, 1u, backing);
            static_cast<void>(Recompiles(oneBuffer.request));
            auto unterminated = MakeRequest(device, UnterminatedCode, 1u, backing);
            static_cast<void>(Recompiles(unterminated.request));
            auto spirv = MakeRequest(device, Vop1ControlCode, 2u, backing);
            spirv.request.target.backend = ShaderRecompiler::ShaderBackend::Spirv;
            static_cast<void>(Recompiles(spirv.request));

            const auto dumps = ReadDumps(directory);
            Require(Count(dumps, "msl", Vop1ControlCode, 8u) == 1u, "the repeated request was not written exactly once");
            Require(Count(dumps, "msl", Vop1ControlCode, 4u) == 1u, "a request that differs only in its user data was not written");
            Require(Count(dumps, "msl", UnterminatedCode, 4u) == 1u, "a request the recompiler rejects was not written");
            Require(Count(dumps, "spirv", Vop1ControlCode, 8u) == 1u, "a SPIR-V request was not written under its own backend name");
            for (const auto& dump : dumps) {
                Require(dump.replay.request.target.subgroupSize == device.Target().subgroupSize, dump.path.filename().string() + " lost the target");
            }
            std::printf("%zu request file(s) written and read back\n", dumps.size());
        }
        fs::remove_all(directory);
        std::puts("request dump check passed");
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored;
        fs::remove_all(directory, ignored);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
