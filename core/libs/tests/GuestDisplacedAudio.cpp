// NGS2 and AJM on guest memory in the displaced window (D31, D33): the context buffers that hold NGS2's system and
// rack objects, the waveform, the parameter lists, the render output, and AJM's batch buffer, bitstream, PCM output
// and sideband all live at displaced guest addresses (the shared region tail 0x400000000 and the GPU carveout
// 0x4000000000), as a title's audio heap can. Both libraries must reach them through host views: the same results as
// in ordinary memory, and no access through the fault handler (each such access costs two signals, and NGS2 renders
// per sample on every audio tick).
#include "SceTypes.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestDisplaced.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libSceNgs2.native/include/Ngs2Types.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
int APS5_VABI sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t, std::size_t, int, std::int64_t*);
int APS5_VABI sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
int APS5_VABI sceNgs2SystemQueryBufferSize(const Ngs2SystemOption*, Ngs2ContextBufferInfo*);
int APS5_VABI sceNgs2SystemCreate(const Ngs2SystemOption*, const Ngs2ContextBufferInfo*, uintptr_t*);
int APS5_VABI sceNgs2SystemDestroy(uintptr_t, Ngs2ContextBufferInfo*);
int APS5_VABI sceNgs2SystemSetGrainSamples(uintptr_t, uint32_t);
int APS5_VABI sceNgs2SystemRender(uintptr_t, const Ngs2RenderBufferInfo*, uint32_t);
int APS5_VABI sceNgs2RackQueryBufferSize(uint32_t, const Ngs2RackOption*, Ngs2ContextBufferInfo*);
int APS5_VABI sceNgs2RackCreate(uintptr_t, uint32_t, const Ngs2RackOption*, const Ngs2ContextBufferInfo*, uintptr_t*);
int APS5_VABI sceNgs2RackGetVoiceHandle(uintptr_t, uint32_t, uintptr_t*);
int APS5_VABI sceNgs2VoiceControl(uintptr_t, const Ngs2VoiceParamHeader*);
int APS5_VABI sceNgs2VoiceGetState(uintptr_t, Ngs2VoiceState*, size_t);
int APS5_VABI sceNgs2VoiceGetStateFlags(uintptr_t, uint32_t*);
int APS5_VABI sceAjmInitialize(std::int64_t, std::uint32_t*);
int APS5_VABI sceAjmFinalize(std::uint32_t);
int APS5_VABI sceAjmInstanceCreate(std::uint32_t, std::uint32_t, std::uint64_t, std::uint32_t*);
int APS5_VABI sceAjmInstanceDestroy(std::uint32_t, std::uint32_t);
int APS5_VABI sceAjmBatchInitialize(void*, std::size_t, AjmBatchInfo*);
int APS5_VABI sceAjmBatchJobDecode(AjmBatchInfo*, std::uint32_t, const void*, std::size_t, void*, std::size_t, void*);
int APS5_VABI sceAjmBatchStart(std::uint32_t, const AjmBatchInfo*, int, AjmBatchError*, std::uint32_t*);
int APS5_VABI sceAjmBatchWait(std::uint32_t, std::uint32_t, std::uint32_t, AjmBatchError*);
}

namespace {

constexpr std::uintptr_t TailBase = 0x400000000ull;
constexpr std::uintptr_t CarveoutBase = 0x4000000000ull;
constexpr std::size_t Bytes = 0x400000;
constexpr int CpuGpuReadWrite = 0x33;
constexpr int MapFixed = 0x10;
constexpr std::uint32_t Grain = 64;
constexpr int Grains = 6;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t Faults() {
    std::uint64_t faults = 0;
    GuestArena::GuestDisplacedStats_nid_no_patch(&faults, nullptr);
    return faults;
}

// Bump allocation over a guest range: Place() returns the guest address; the test itself writes through the host
// view (HostAddress), so only the libraries' own accesses could fault.
struct Arena {
    std::uintptr_t base;
    std::size_t used = 0;
    template <typename T = std::uint8_t>
    T* Place(std::size_t bytes) {
        used = (used + 63) & ~std::size_t{63};
        Require(used + bytes <= Bytes, "the test arena is full");
        auto* placed = reinterpret_cast<T*>(base + used);
        used += bytes;
        std::memset(Host(placed), 0, bytes);
        return placed;
    }
    template <typename T>
    static T* Host(T* guest) {
        return reinterpret_cast<T*>(GuestArena::HostAddress(reinterpret_cast<std::uintptr_t>(guest)));
    }
    template <typename T>
    T* Copy(const T& value) {
        auto* placed = Place<T>(sizeof(T));
        std::memcpy(Host(placed), &value, sizeof(T));
        return placed;
    }
};

Ngs2ContextBufferInfo ContextBuffer(Arena& arena, const Ngs2ContextBufferInfo& query) {
    Ngs2ContextBufferInfo info{};
    info.host_buffer = arena.Place(query.host_buffer_size);
    info.host_buffer_size = query.host_buffer_size;
    return info;
}

uintptr_t CreateVoice(Arena& arena, uintptr_t system, std::uint32_t rackId) {
    Ngs2ContextBufferInfo query{};
    Require(sceNgs2RackQueryBufferSize(rackId, nullptr, &query) == SCE_NGS2_OK, "rack buffer size");
    const auto* info = arena.Copy(ContextBuffer(arena, query));
    uintptr_t rack = 0;
    Require(sceNgs2RackCreate(system, rackId, nullptr, info, &rack) == SCE_NGS2_OK && rack != 0, "rack create");
    uintptr_t voice = 0;
    Require(sceNgs2RackGetVoiceHandle(rack, 0, &voice) == SCE_NGS2_OK && voice != 0, "voice handle");
    return voice;
}

template <typename TParam>
void Control(Arena& arena, uintptr_t voice, std::uint32_t id, TParam param) {
    param.header = {static_cast<std::uint16_t>(sizeof(TParam)), 0, id};
    Require(sceNgs2VoiceControl(voice, &arena.Copy(param)->header) == SCE_NGS2_OK, "voice control " + std::to_string(id));
}

// A sampler playing a looped ramp into a stereo mastering voice; Grains renders of Grain samples (I16).
std::vector<std::int16_t> RenderNgs2(Arena& arena) {
    Ngs2ContextBufferInfo query{};
    Require(sceNgs2SystemQueryBufferSize(nullptr, &query) == SCE_NGS2_OK, "system buffer size");
    const auto* systemInfo = arena.Copy(ContextBuffer(arena, query));
    uintptr_t system = 0;
    Require(sceNgs2SystemCreate(nullptr, systemInfo, &system) == SCE_NGS2_OK && system != 0, "system create");
    Require(system == reinterpret_cast<uintptr_t>(Arena::Host(systemInfo)->host_buffer), "the system handle is not the guest address of its buffer");
    Require(sceNgs2SystemSetGrainSamples(system, Grain) == SCE_NGS2_OK, "grain");

    const auto mastering = CreateVoice(arena, system, SCE_NGS2_RACK_ID_MASTERING);
    Control(arena, mastering, SCE_NGS2_MASTERING_VOICE_PARAM_SETUP, Ngs2MasteringVoiceSetupParam{{}, 2, 0});
    Control(arena, mastering, SCE_NGS2_MASTERING_VOICE_PARAM_OUTPUT, Ngs2MasteringVoiceOutputParam{{}, 0, 0});
    Control(arena, mastering, SCE_NGS2_VOICE_PARAM_EVENT, Ngs2VoiceEventParam{{}, SCE_NGS2_VOICE_EVENT_PLAY});

    const auto sampler = CreateVoice(arena, system, SCE_NGS2_RACK_ID_SAMPLER);
    Control(arena, sampler, SCE_NGS2_SAMPLER_VOICE_PARAM_SETUP, Ngs2SamplerVoiceSetupParam{{}, {SCE_NGS2_WAVEFORM_TYPE_PCM_I16L, 2, 48000, 0, 0, 0}});
    constexpr std::uint32_t frames = 100;
    auto* pcm = arena.Place<std::int16_t>(frames * 2 * sizeof(std::int16_t));
    for (std::uint32_t i = 0; i < frames * 2; ++i) Arena::Host(pcm)[i] = static_cast<std::int16_t>((i * 331) % 20000 - 10000);
    const auto* block = arena.Copy(Ngs2WaveformBlock{0, frames * 2 * sizeof(std::int16_t), UINT32_MAX, 0, frames, 0, 0x55});
    Control(arena, sampler, SCE_NGS2_SAMPLER_VOICE_PARAM_ADD_WAVEFORM_BLOCKS, Ngs2SamplerVoiceWaveformBlocksParam{{}, pcm, 0, 1, block});
    Control(arena, sampler, SCE_NGS2_VOICE_PARAM_PATCH, Ngs2VoicePatchParam{{}, 0, 0, mastering});
    Control(arena, sampler, SCE_NGS2_VOICE_PARAM_EVENT, Ngs2VoiceEventParam{{}, SCE_NGS2_VOICE_EVENT_PLAY});

    auto* output = arena.Place<std::int16_t>(Grain * 2 * sizeof(std::int16_t));
    const auto* outputInfo = arena.Copy(Ngs2RenderBufferInfo{output, Grain * 2 * sizeof(std::int16_t), SCE_NGS2_WAVEFORM_TYPE_PCM_I16L, 2});
    auto* state = arena.Place<Ngs2SamplerVoiceState>(sizeof(Ngs2SamplerVoiceState));
    auto* flags = arena.Place<std::uint32_t>(sizeof(std::uint32_t));
    std::vector<std::int16_t> rendered;
    for (int grain = 0; grain < Grains; ++grain) {
        Require(sceNgs2SystemRender(system, outputInfo, 1) == SCE_NGS2_OK, "render");
        rendered.insert(rendered.end(), Arena::Host(output), Arena::Host(output) + Grain * 2);
        Require(sceNgs2VoiceGetState(sampler, &state->voice_state, sizeof(*state)) == SCE_NGS2_OK, "voice state");
        Require(sceNgs2VoiceGetStateFlags(sampler, flags) == SCE_NGS2_OK, "voice state flags");
        const auto& host = *Arena::Host(state);
        const auto data = reinterpret_cast<std::uintptr_t>(host.waveform_data);
        Require(data >= reinterpret_cast<std::uintptr_t>(pcm) && data < reinterpret_cast<std::uintptr_t>(pcm) + frames * 4, "GetState's waveform_data is not a guest address in the waveform");
        Require(host.user_data == 0x55 && (*Arena::Host(flags) & SCE_NGS2_VOICE_STATE_FLAG_PLAYING) != 0, "voice state values");
    }
    Require(sceNgs2SystemDestroy(system, nullptr) == SCE_NGS2_OK, "system destroy");
    return rendered;
}

const std::uint8_t MP3_MONO[] = {
    0xFF, 0xFB, 0x14, 0xC4, 0x00, 0x00, 0x03, 0xB0, 0x21, 0x5E, 0xF4, 0x30, 0x80, 0x30, 0x98, 0x88, 0xA9, 0x83, 0x34, 0x70, 0x00, 0x18, 0x89, 0xCB,
    0x20, 0x00, 0x3B, 0xBB, 0xB9, 0x9F, 0xD7, 0x0E, 0x06, 0x06, 0x06, 0x2C, 0x3E, 0xEF, 0x83, 0xE7, 0xF1, 0x00, 0x63, 0x83, 0xFD, 0x4E, 0xFC, 0xFF,
    0x29, 0xE1, 0xF7, 0xD9, 0x87, 0x7E, 0xFF, 0x65, 0x9E, 0x21, 0xAF, 0x46, 0x58, 0xBB, 0x46, 0x58, 0xBA, 0x64, 0x40, 0x21, 0xB2, 0x6C, 0xFA, 0x6C,
    0xF8, 0x42, 0x0E, 0x7C, 0x1E, 0x8F, 0x0D, 0xBF, 0x1B, 0x02, 0xBC, 0x24, 0x0D, 0x37, 0xCA, 0x9D, 0x52, 0xFD, 0x77, 0xE0, 0x16, 0xBA, 0x02, 0x04,
    0xFF, 0xFB, 0x14, 0xC4, 0x03, 0x83, 0xC4, 0xE0, 0x2B, 0x28, 0x1D, 0xE4, 0x80, 0x20, 0x5D, 0x95, 0x60, 0x41, 0x70, 0x16, 0xC8, 0xD3, 0x02, 0x00,
    0x2F, 0x12, 0x02, 0xD0, 0x40, 0x1D, 0x98, 0x4D, 0x06, 0x99, 0x84, 0xF2, 0x68, 0x98, 0xBF, 0x86, 0xD1, 0x84, 0xC8, 0x18, 0x2A, 0xA9, 0x79, 0x50,
    0x85, 0xDE, 0x14, 0x88, 0x9A, 0xF8, 0x44, 0x02, 0xB0, 0x60, 0x62, 0x5E, 0x11, 0xE9, 0xDC, 0x18, 0x65, 0x8F, 0xFF, 0x7F, 0xFC, 0xB3, 0xF8, 0x58,
    0xEF, 0xFF, 0xFF, 0xE1, 0x95, 0xF8, 0x44, 0x02, 0xB0, 0x60, 0x62, 0x5E, 0x11, 0xE9, 0xDC, 0x18, 0x65, 0x8F, 0xFF, 0x7F, 0xFC, 0xB3, 0xF8, 0x58,
    0xFF, 0xFB, 0x14, 0xC4, 0x09, 0x83, 0xC2, 0xEC, 0xAB, 0x02, 0x0B, 0x80, 0xB6, 0x40, 0x5D, 0x95, 0x60, 0x41, 0x70, 0x16, 0xC8, 0xEF, 0xFF, 0xFF,
    0xE1, 0x9F, 0x08, 0x80, 0x56, 0x0C, 0x0C, 0x4B, 0xC2, 0x3D, 0x3B, 0x83, 0x0C, 0xB1, 0xFF, 0xEF, 0xFF, 0x96, 0x7F, 0x0B, 0x1D, 0xFF, 0xFF, 0xFC,
    0x31, 0xF8, 0x44, 0x02, 0xB0, 0x60, 0x62, 0x5E, 0x11, 0xE9, 0xDC, 0x18, 0x65, 0x8F, 0xFF, 0x7F, 0xFC, 0xB3, 0xF8, 0x58, 0xEF, 0xFF, 0xFF, 0xE1,
    0x9F, 0x08, 0x80, 0x56, 0x0C, 0x0C, 0x4B, 0xC2, 0x3D, 0x3B, 0x83, 0x0C, 0xB1, 0xFF, 0xEF, 0xFF, 0x96, 0x7F, 0x0B, 0x1D, 0xFF, 0xFF, 0xFC, 0x31,
    0xFF, 0xFB, 0x14, 0xC4, 0x17, 0x83, 0xC2, 0xEC, 0xAB, 0x02, 0x0B, 0x80, 0xB6, 0x40, 0x5D, 0x95, 0x60, 0x41, 0x70, 0x16, 0xC8, 0x96, 0x55, 0x8D,
    0xBF, 0xE9, 0xD8, 0x50, 0x01, 0xA6, 0x17, 0x80, 0x34, 0x61, 0x00, 0x17, 0xC6, 0x15, 0x40, 0xD4, 0x61, 0x56, 0x25, 0x66, 0x3E, 0xA4, 0x92, 0x62,
    0xC3, 0xED, 0x26, 0x23, 0xA5, 0x5C, 0x63, 0x10, 0x01, 0xC6, 0x0F, 0x20, 0x9E, 0x60, 0x62, 0x08, 0x26, 0x06, 0xE0, 0x12, 0xD6, 0x22, 0x02, 0x6C,
    0x93, 0xC7, 0x04, 0x3E, 0xBF, 0xFE, 0xBF, 0xFF, 0x72, 0x62, 0x28, 0x22, 0xD1, 0x2F, 0xEF, 0xFD, 0xF1, 0x15, 0x09, 0x83, 0x08, 0x8C, 0x00, 0xFE,
    0xFF, 0xFB, 0x14, 0xC4, 0x25, 0x80, 0x07, 0x74, 0x2D, 0x16, 0x15, 0xE7, 0x80, 0x00, 0xF2, 0x0C, 0xE8, 0x83, 0x36, 0xB0, 0x00, 0xC4, 0x42, 0x82,
    0xC0, 0x86, 0x24, 0x2A, 0x02, 0x11, 0xFF, 0x02, 0x70, 0xF6, 0x07, 0x80, 0x9B, 0xCB, 0xCD, 0x0D, 0x00, 0x7C, 0x49, 0x1D, 0xBF, 0xD3, 0xDE, 0x3B,
    0x52, 0x36, 0xFE, 0xFA, 0xF3, 0x66, 0xB5, 0x13, 0xDF, 0x06, 0x84, 0xA1, 0x2F, 0xCB, 0x05, 0x41, 0x5F, 0xF5, 0x2A, 0x4C, 0x41, 0x4D, 0x45, 0x34,
    0x2E, 0x30, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xFF, 0xFB, 0x14, 0xC4, 0x0E, 0x83, 0xC0, 0x00, 0x01, 0xA4, 0x1C, 0x00, 0x00, 0x20, 0x00, 0x00, 0x34, 0x80, 0x00, 0x00, 0x04, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
    0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
};

struct DecodeSideband {
    std::int32_t result;
    std::int32_t internalResult;
    std::int32_t inputConsumed;
    std::int32_t outputWritten;
    std::uint64_t totalDecodedSamples;
};

// MP3 frames decoded through a batch whose buffer, input, output and sideband live in the arena.
std::vector<std::int16_t> DecodeAjm(Arena& arena, std::uint32_t context) {
    std::uint32_t instance = 0;
    Require(sceAjmInstanceCreate(context, 0, 0, &instance) == 0, "AJM instance");
    auto* batch = arena.Place(0x80);
    auto* input = arena.Place(sizeof(MP3_MONO));
    std::memcpy(Arena::Host(input), MP3_MONO, sizeof(MP3_MONO));
    auto* pcm = arena.Place<std::int16_t>(1152 * sizeof(std::int16_t));
    auto* sideband = arena.Place<DecodeSideband>(sizeof(DecodeSideband));
    auto* info = arena.Place<AjmBatchInfo>(sizeof(AjmBatchInfo));
    auto* error = arena.Place<AjmBatchError>(sizeof(AjmBatchError));
    auto* id = arena.Place<std::uint32_t>(sizeof(std::uint32_t));
    std::vector<std::int16_t> decoded;
    std::size_t offset = 0;
    for (int frame = 0; frame < 6; ++frame) {
        Require(sceAjmBatchInitialize(batch, 0x80, info) == 0, "AJM batch initialize");
        Require(sceAjmBatchJobDecode(info, instance, input + offset, sizeof(MP3_MONO) - offset, pcm, 1152 * sizeof(std::int16_t), sideband) == 0, "AJM decode job");
        Require(sceAjmBatchStart(context, info, 0, error, id) == 0 && sceAjmBatchWait(context, *Arena::Host(id), 0, error) == 0, "AJM batch");
        const auto& result = *Arena::Host(sideband);
        Require(result.result == 0 && result.inputConsumed == 96 && result.outputWritten == 1152 * 2, "AJM sideband");
        decoded.insert(decoded.end(), Arena::Host(pcm), Arena::Host(pcm) + 1152);
        offset += static_cast<std::size_t>(result.inputConsumed);
    }
    Require(sceAjmInstanceDestroy(context, instance) == 0, "AJM instance destroy");
    return decoded;
}

}

int main() {
#if !(defined(__APPLE__) && defined(__aarch64__))
    std::puts("guest displaced audio: no displaced window on this host, skipped");
    return 0;
#else
    try {
        Require(GuestArena::GuestDisplacedActive_nid_no_patch(), "the displaced window is not active on macOS arm64");
        std::uint32_t context = 0;
        Require(sceAjmInitialize(0, &context) == 0 && context != 0, "AJM initialize");
        std::vector<std::uint64_t> ordinaryStorage(Bytes / sizeof(std::uint64_t));
        Arena ordinary{reinterpret_cast<std::uintptr_t>(ordinaryStorage.data())};
        const auto expectedNgs2 = RenderNgs2(ordinary);
        const auto expectedAjm = DecodeAjm(ordinary, context);
        bool audible = false;
        for (const auto sample : expectedNgs2) audible = audible || sample != 0;
        Require(audible, "the ordinary NGS2 render is silent");
        std::int64_t physical = 0;
        Require(sceKernelAllocateDirectMemory(0, 0x100000000ll, Bytes, 0x200000, 3, &physical) == 0, "allocate direct memory");
        for (const std::uintptr_t base : {TailBase, CarveoutBase}) {
            Require(GuestArena::IsDisplaced(base), "not displaced");
            void* fixed = reinterpret_cast<void*>(base);
            Require(sceKernelMapDirectMemory(&fixed, Bytes, CpuGpuReadWrite, MapFixed, physical, 0x200000) == 0 && fixed == reinterpret_cast<void*>(base), "the fixed mapping failed");
            Arena displaced{base};
            auto before = Faults();
            const auto ngs2 = RenderNgs2(displaced);
            const auto ngs2Faults = Faults() - before;
            before = Faults();
            const auto ajm = DecodeAjm(displaced, context);
            const auto ajmFaults = Faults() - before;
            std::printf("guest displaced audio at 0x%llx: NGS2 faults %llu, AJM faults %llu\n", static_cast<unsigned long long>(base),
                static_cast<unsigned long long>(ngs2Faults), static_cast<unsigned long long>(ajmFaults));
            Require(ngs2 == expectedNgs2, "NGS2 rendered different samples from displaced memory");
            Require(ajm == expectedAjm, "AJM decoded different samples into displaced memory");
            Require(ngs2Faults == 0, "NGS2 accessed displaced guest memory through the fault handler");
            Require(ajmFaults == 0, "AJM accessed displaced guest memory through the fault handler");
            Require(sceKernelMunmap(fixed, Bytes) == 0, "munmap");
        }
        Require(sceKernelReleaseDirectMemory(physical, Bytes) == 0, "release direct memory");
        Require(sceAjmFinalize(context) == 0, "AJM finalize");
        std::puts("guest displaced audio tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "guest displaced audio: %s\n", error.what());
        return 1;
    }
#endif
}
