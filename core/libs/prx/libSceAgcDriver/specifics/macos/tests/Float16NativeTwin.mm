// Metal twin of tests/execution/Float16Native.cpp (agc_driver_float16_native; upstream 89e70ab80d, scalar f16
// add, sub and mul in native f16 math), swapped in by twins/UPSYNC3-tests.cmake. The upstream file is compiled here
// with its main renamed, so its program, modes, inputs and helpers keep tracking upstream. Upstream checks which
// math the module uses by scanning result.spirv, which is empty for an MSL target; this main checks the MSL source
// instead: the native compile must use half2 add and mul (MslAluMath.cpp FPAdd16x2/FPMul16x2:
// `as_type<half2>(a) + as_type<half2>(b)`) exactly in the modes upstream expects, and the compile without
// CapabilityFloat16 (the f32 path) must use none. The GPU comparison of the two compiles is upstream's, unchanged.

#define main AgcDriverFloat16NativeVulkanMain
#include "../../../tests/execution/Float16Native.cpp"
#undef main

#include <string_view>

namespace MetalFloat16NativeTwin {

F16x2Math FindMslF16x2Math(const ShaderRecompiler::RecompileResult& result, std::string_view what) {
    Require(!result.msl.empty(), std::string("f16 native: the ") + std::string(what) + " compile has no MSL source");
    const std::string& source = result.msl.Source();
    F16x2Math math;
    math.add = source.find(") + as_type<half2>(") != std::string::npos;
    math.mul = source.find(") * as_type<half2>(") != std::string::npos;
    math.fma = source.find("fma(as_type<half2>(") != std::string::npos;
    return math;
}

}

int main() {
    try {
        // Upstream's SPIR-V scan has no input on Metal; it stays compiled for the renamed main.
        (void)&FindF16x2Math;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        const auto supports = [&](spv::Capability capability) {
            return std::ranges::find(target.supportedCapabilities, static_cast<std::uint32_t>(capability)) != target.supportedCapabilities.end();
        };
        if (!supports(spv::CapabilityFloat16) || !supports(spv::CapabilityRoundingModeRTE) || !supports(spv::CapabilityDenormPreserve)) {
            std::puts("skipped, the Metal target has no native f16 with RTE rounding and kept denormals");
            return VulkanTestSkipped;
        }
        std::vector<std::uint32_t> withoutFloat16;
        std::ranges::copy_if(target.supportedCapabilities, std::back_inserter(withoutFloat16), [](std::uint32_t capability) { return capability != spv::CapabilityFloat16; });
        auto emulatedTarget = target;
        emulatedTarget.supportedCapabilities = withoutFloat16;

        const std::span<const std::uint32_t> code(Code);
        std::vector<std::uint32_t> userData(8, 0u);
        const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
        const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
        std::copy(input.begin(), input.end(), userData.begin());
        std::copy(output.begin(), output.end(), userData.begin() + 4);
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
        const auto compile = [&](const ShaderRecompiler::SpirvTarget& shaderTarget, const Mode& mode) {
            ShaderRecompiler::RecompileRequest request{
                {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
                {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
                shaderTarget,
                {0, 0, 0, 128}
            };
            request.useCache = false;
            request.context.floatMode = mode.floatMode;
            return ShaderRecompiler::Recompile(request);
        };
        const auto run = [&](const auto& result) {
            Output.fill(0xdeadbeefu);
            device->Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
            device->WaitIdle();
            return Output;
        };

        std::uint64_t compared = 0u;
        std::uint64_t nanPayloads = 0u;
        const auto isNan16 = [](std::uint32_t value) { return (value & 0x7fffu) > 0x7c00u; };
        for (const auto& mode : Modes) {
            const auto native = compile(target, mode);
            const auto emulated = compile(emulatedTarget, mode);
            const F16x2Math nativeMath = MetalFloat16NativeTwin::FindMslF16x2Math(native, "native");
            const F16x2Math emulatedMath = MetalFloat16NativeTwin::FindMslF16x2Math(emulated, "f32 path");
            Require(nativeMath.add == mode.native && nativeMath.mul == mode.native, std::string("f16 native: ") + mode.name + (mode.native ? " does not use" : " uses") + " half2 add and mul in MSL");
            Require(!nativeMath.fma, std::string("f16 native: ") + mode.name + " uses a half2 fma in MSL");
            Require(!emulatedMath.add && !emulatedMath.mul && !emulatedMath.fma, std::string("f16 native: ") + mode.name + " uses half2 math in MSL without Float16");
            for (std::uint32_t batch = 0; batch < Batches; ++batch) {
                for (auto& word : Input) word = Half() | (Half() << 16u);
                const auto nativeOutput = run(native);
                const auto emulatedOutput = run(emulated);
                for (std::uint32_t tid = 0; tid < Threads; ++tid) {
                    for (std::uint32_t index = 0; index < Results; ++index) {
                        const std::uint32_t actual = nativeOutput[tid * Stride + index];
                        const std::uint32_t expected = emulatedOutput[tid * Stride + index];
                        const std::uint32_t* row = &Input[tid * Inputs];
                        const bool bothNan = (actual >> 16u) == (expected >> 16u) && isNan16(actual) && isNan16(expected);
                        if (actual != expected && bothNan) ++nanPayloads;
                        Require(actual == expected || bothNan, std::string("f16 native: ") + mode.name + ", " + Names[index] + " with v4=" + Hex(row[0]) + " v5=" + Hex(row[1]) + " v6=" + Hex(row[2]) + " v7=" + Hex(row[3]) + " is " + Hex(actual) + ", f32 path gives " + Hex(expected));
                    }
                }
                compared += Threads * Results;
            }
        }
        std::printf("f16 native tests passed on Metal (%llu results, %llu NaN results with another payload)\n", static_cast<unsigned long long>(compared), static_cast<unsigned long long>(nanPayloads));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
