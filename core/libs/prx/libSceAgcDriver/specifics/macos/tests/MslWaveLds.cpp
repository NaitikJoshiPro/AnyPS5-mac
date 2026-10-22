// agc_metal_msl_wave_lds (work package B3): MetalBackend's wave, LDS and atomic emission checked on
// the CPU, without a Metal device. Real RDNA2 programs of the execution tests are recompiled for the
// Metal target and the generated MSL and reflection are checked for the B3 contract:
// - cross-lane reads are simd_shuffle statements (both halves for wave64 at two lanes per invocation);
// - LDS within threadgroup memory is `threadgroup uint aps5_lds[N]`, beyond it the workgroup's slice
//   of the WorkgroupMemory arena with Reflection::workgroupMemoryStrideDwords = N + 1;
// - 64-bit LDS atomics take the LDS lock: lock dword reset before a barrier, elected try-lock per
//   SIMD group per round, a LoopLimit fault after the round limit instead of a hang;
// - every lock loop (LDS, 64-bit device atomics of global_atomics64 and look_back_scan) has
//   MslLockLoop's forward-progress shape (CheckLockLoops), and look-back readers never spin on a lock;
// - ds_append/ds_consume issue one atomic per wave and broadcast its result.
// The GPU behaviour is covered by the execution tests (agc_driver_*); this test keeps the emitted
// shape from regressing in ways those tests would only show as a hang.

#include "Recompiler.hpp"
#include "MslAbi.hpp"
#include "BdaAbi.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <regex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// tests/execution/ReadLane.cpp (ReadLaneCode).
alignas(256) constexpr std::array<std::uint32_t, 21> ReadLaneCode{
    0x34020082, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xbf8c3f70, 0xd7600008, 0x00014b04,
    0x7e280505, 0xd7600009, 0x00002904, 0x7e140208, 0x7e160209, 0x7e180214, 0xe0702000, 0x80010a01,
    0xe0702004, 0x80010b01, 0xe0702008, 0x80010c01, 0xbf810000,
};

// tests/execution/Dpp8.cpp (Wave32Code).
alignas(256) constexpr std::array<std::uint32_t, 67> Dpp8Wave32Code{
    0x34020084, 0x34060086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e1402ff, 0xdeaddead, 0x7e1602ff,
    0xdeaddead, 0x7e1802ff, 0xdeaddead, 0x7e1a02ff, 0xdeaddead, 0x7e1c02ff, 0xdeaddead, 0x7e1e02ff,
    0xdeaddead, 0x7e2002ff, 0xdeaddead, 0x7e2202ff, 0xdeaddead, 0x7e2402ff, 0xdeaddead, 0x7e2602ff,
    0xdeaddead, 0x7e2802ff, 0xdeaddead, 0x7e2a02ff, 0xdeaddead, 0x7e1402e9, 0x78afc504, 0x7e1602ea,
    0x05397704, 0x4a180ae9, 0x00024904, 0x061a0ee9, 0x97705306, 0x7e1c0306, 0x561c0ee9, 0xa5602406,
    0x7d880b04, 0x021e0ae9, 0x78afc504, 0x7e200ae9, 0x97705304, 0x50220ae9, 0x05397704, 0x7e1202ff,
    0x12345678, 0xbefe03ff, 0x5a5af0f3, 0x7e2402e9, 0x78afc504, 0x7e2602ea, 0x78afc504, 0x4c280ae9,
    0xa5602404, 0x7e2a02e9, 0x05397709, 0xbefe03c1, 0xe0781000, 0x80010a03, 0xe0781010, 0x80010e03,
    0xe0781020, 0x80011203, 0xbf810000,
};

// tests/execution/Dpp8.cpp (Wave64Code).
alignas(256) constexpr std::array<std::uint32_t, 69> Dpp8Wave64Code{
    0x34020084, 0x34060086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0x7e1402ff, 0xdeaddead, 0x7e1602ff,
    0xdeaddead, 0x7e1802ff, 0xdeaddead, 0x7e1a02ff, 0xdeaddead, 0x7e1c02ff, 0xdeaddead, 0x7e1e02ff,
    0xdeaddead, 0x7e2002ff, 0xdeaddead, 0x7e2202ff, 0xdeaddead, 0x7e2402ff, 0xdeaddead, 0x7e2602ff,
    0xdeaddead, 0x7e2802ff, 0xdeaddead, 0x7e2a02ff, 0xdeaddead, 0x7e1402e9, 0x78afc504, 0x7e1602ea,
    0x05397704, 0x4a180ae9, 0x00024904, 0x061a0ee9, 0x97705306, 0x7e1c0306, 0x561c0ee9, 0xa5602406,
    0x7d880b04, 0x021e0ae9, 0x78afc504, 0x7e200ae9, 0x97705304, 0x50220ae9, 0x05397704, 0x7e1202ff,
    0x12345678, 0xbefe03ff, 0x5a5af0f3, 0xbeff03ff, 0xc3a5960f, 0x7e2402e9, 0x78afc504, 0x7e2602ea,
    0x78afc504, 0x4c280ae9, 0xa5602404, 0x7e2a02e9, 0x05397709, 0xbefe04c1, 0xe0781000, 0x80010a03,
    0xe0781010, 0x80010e03, 0xe0781020, 0x80011203, 0xbf810000,
};

// tests/execution/LdsBeyondDeviceLimit.cpp (Code).
alignas(256) constexpr std::array<std::uint32_t, 69> LdsBeyondCode{
    0x34020084, 0xe0381000, 0x80000401, 0x7e100208, 0x7e3e0209, 0x343e3e82, 0x4a103f08, 0x7e40020a,
    0x34404083, 0x4a104108, 0x34121090, 0x4a121300, 0x4a1412ff, 0x10000000, 0x7e160280, 0x7e180280,
    0x7e1a0281, 0x7e1e0281, 0x7e200280, 0xbf8c0000, 0xd9340000, 0x00000b06, 0xd834000c, 0x00000b06,
    0xd8340000, 0x00000904, 0xd8340000, 0x00000a05, 0xd8340000, 0x00000907, 0xbf8c0000, 0xbf8a0000,
    0xd880000c, 0x28000d06, 0xd9800000, 0x29000f06, 0x3a260884, 0x3a2a0aff, 0x00000080, 0x363608ff,
    0xfffffff8, 0xbf8c0000, 0xbf8a0000, 0xd8d80000, 0x2b000013, 0xd8d80000, 0x2c000015, 0xd8d80000,
    0x2d000007, 0xd8d8000c, 0x2e000006, 0xd9d80000, 0x2f000006, 0xd9d80000, 0x3100001b, 0xd8d80000,
    0x33000005, 0x343c1086, 0x4a3c3d00, 0x343c3c86, 0xbf8c0000, 0xe0781000, 0x8001281e, 0xe0781010,
    0x80012c1e, 0xe0781020, 0x8001301e, 0xbf8c0000, 0xbf810000,
};

// Constants of tests/execution/DsAppendConsume.cpp its code embeds.
constexpr std::uint32_t AppendStart = 1000;
constexpr std::uint32_t ConsumeStart = 2000;
constexpr std::uint32_t MaskedAppendStart = 3000;
constexpr std::uint32_t MaskedConsumeStart = 4000;
constexpr std::uint32_t BaseShiftedStart = 5000;
constexpr std::uint32_t UntouchedAppend = 0x5eed0001u;
constexpr std::uint32_t UntouchedConsume = 0x5eed0002u;
// tests/execution/DsAppendConsume.cpp (Code).
alignas(256) constexpr std::array<std::uint32_t, 98> AppendConsumeCode{
    0x34060084, 0x7e100280, 0x7e0802ff, AppendStart, 0x7e0a02ff, ConsumeStart, 0x7e0c02ff, MaskedAppendStart,
    0x7e0e02ff, MaskedConsumeStart, 0x7e1c02ff, BaseShiftedStart, 0x7e1e02ff, BaseShiftedStart + 1u, 0x7e2002ff, BaseShiftedStart + 2u,
    0x7e2202ff, BaseShiftedStart + 3u, 0x7e1802ff, UntouchedAppend, 0x7e1a02ff, UntouchedConsume, 0x7da40080, 0xd8340010,
    0x00000408, 0xd8340014, 0x00000508, 0xd8340018, 0x00000608, 0xd834001c, 0x00000708, 0xd8340050,
    0x00000e08, 0xd8340054, 0x00000f08, 0xd8340058, 0x00001008, 0xd834005c, 0x00001108, 0xbefe03c1,
    0xbf8cc07f, 0xbefc03ff, 0x0040ffff, 0xd8f80010, 0x0a000000, 0xd8f40014, 0x0b000000, 0xbefc03ff,
    0x00400000, 0x7da80090, 0xd8f80018, 0x0c000000, 0xd8f4001c, 0x0d000000, 0xbefe03c1, 0xbf8cc07f,
    0xd8d80010, 0x14000008, 0xd8d80014, 0x15000008, 0xd8d80018, 0x16000008, 0xd8d8001c, 0x17000008,
    0xd8d80050, 0x18000008, 0xd8d80054, 0x19000008, 0xd8d80058, 0x1a000008, 0xd8d8005c, 0x1b000008,
    0xbf8cc07f, 0xe0702000, 0x80000a03, 0xe0702004, 0x80000b03, 0xe0702008, 0x80000c03, 0xe070200c,
    0x80000d03, 0xe0702010, 0x80001403, 0xe0702014, 0x80001503, 0xe0702018, 0x80001603, 0xe070201c,
    0x80001703, 0xe0702020, 0x80001803, 0xe0702024, 0x80001903, 0xe0702028, 0x80001a03, 0xe070202c,
    0x80001b03, 0xbf810000,
};

// tests/execution/LdsAtomics64.cpp (Code).
alignas(256) constexpr std::array<std::uint32_t, 475> LdsAtomics64Code{
    0x34020085, 0x34060089, 0x34040083, 0x7ef80280, 0x7efa0280, 0x7efc0280, 0xe0301000, 0x80000401,
    0xe0301004, 0x80000501, 0xe0301008, 0x80000601, 0xe030100c, 0x80000701, 0xe0301010, 0x80000801,
    0xe0301014, 0x80000901, 0xbf8c3f70, 0xd9340000, 0x00000402, 0xd9340100, 0x00000402, 0xd9340200,
    0x00000402, 0xd9340300, 0x00000402, 0xd9340400, 0x00000402, 0xd9340500, 0x00000402, 0xd9340600,
    0x00000402, 0xd9340700, 0x00000402, 0xd9340800, 0x00000402, 0xd9340900, 0x00000402, 0xd9340a00,
    0x00000402, 0xd9340b00, 0x00000402, 0xd9340c00, 0x00000402, 0xd9340d00, 0x00000402, 0xd9340e00,
    0x00000402, 0xd9340f00, 0x00000402, 0xd9341000, 0x00000402, 0xd9341100, 0x00000402, 0xd9341200,
    0x00000402, 0xd9341300, 0x00000402, 0xd9341400, 0x00000402, 0xd9341500, 0x00000402, 0xd9341600,
    0x00000402, 0xd9341700, 0x00000402, 0xd9341800, 0x00000402, 0xd9341900, 0x00000402, 0xd9341a00,
    0x00000402, 0xd9341b00, 0x00000402, 0xd9341c00, 0x00000402, 0xd9341d00, 0x00000402, 0xd9341e00,
    0x00000402, 0xd9341f00, 0x00000402, 0xd9342000, 0x00000402, 0xd9342100, 0x00000402, 0xd9342200,
    0x00000402, 0xd9342300, 0x00007c7e, 0xd9342308, 0x00007c7e, 0xbf8cc07f, 0xbf8a0000, 0xd9000000,
    0x00000602, 0xd9040100, 0x00000602, 0xd9080200, 0x00000602, 0xd90c0300, 0x00000602, 0xd9100400,
    0x00000602, 0xd9140500, 0x00000602, 0xd9180600, 0x00000602, 0xd91c0700, 0x00000602, 0xd9200800,
    0x00000602, 0xd9240900, 0x00000602, 0xd9280a00, 0x00000602, 0xd92c0b00, 0x00000602, 0xd9300c00,
    0x00080602, 0xd9400d00, 0x00080602, 0xd9440e00, 0x00080602, 0xd9480f00, 0x00000602, 0xd94c1000,
    0x00000602, 0xd9801100, 0x0a000602, 0xd9841200, 0x0c000602, 0xd9881300, 0x0e000602, 0xd98c1400,
    0x10000602, 0xd9901500, 0x12000602, 0xd9941600, 0x14000602, 0xd9981700, 0x16000602, 0xd99c1800,
    0x18000602, 0xd9a01900, 0x1a000602, 0xd9a41a00, 0x1c000602, 0xd9a81b00, 0x1e000602, 0xd9ac1c00,
    0x20000602, 0xd9b01d00, 0x22080602, 0xd9b41e00, 0x24000602, 0xd9c01f00, 0x26080602, 0xd9c42000,
    0x28080602, 0xd9c82100, 0x2a000602, 0xd9cc2200, 0x2c000602, 0xd9802300, 0x2e00067e, 0xd9a02308,
    0x3000087e, 0xbf8cc07f, 0xbf8a0000, 0xd9d80000, 0x32000002, 0xd9d80100, 0x34000002, 0xd9d80200,
    0x36000002, 0xd9d80300, 0x38000002, 0xd9d80400, 0x3a000002, 0xd9d80500, 0x3c000002, 0xd9d80600,
    0x3e000002, 0xd9d80700, 0x40000002, 0xd9d80800, 0x42000002, 0xd9d80900, 0x44000002, 0xd9d80a00,
    0x46000002, 0xd9d80b00, 0x48000002, 0xd9d80c00, 0x4a000002, 0xd9d80d00, 0x4c000002, 0xd9d80e00,
    0x4e000002, 0xd9d80f00, 0x50000002, 0xd9d81000, 0x52000002, 0xd9d81100, 0x54000002, 0xd9d81200,
    0x56000002, 0xd9d81300, 0x58000002, 0xd9d81400, 0x5a000002, 0xd9d81500, 0x5c000002, 0xd9d81600,
    0x5e000002, 0xd9d81700, 0x60000002, 0xd9d81800, 0x62000002, 0xd9d81900, 0x64000002, 0xd9d81a00,
    0x66000002, 0xd9d81b00, 0x68000002, 0xd9d81c00, 0x6a000002, 0xd9d81d00, 0x6c000002, 0xd9d81e00,
    0x6e000002, 0xd9d81f00, 0x70000002, 0xd9d82000, 0x72000002, 0xd9d82100, 0x74000002, 0xd9d82200,
    0x76000002, 0xd9d82300, 0x7800007e, 0xd9d82308, 0x7a00007e, 0xbf8cc07f, 0xe0701000, 0x80013203,
    0xe0701004, 0x80013303, 0xe0701008, 0x80013403, 0xe070100c, 0x80013503, 0xe0701010, 0x80013603,
    0xe0701014, 0x80013703, 0xe0701018, 0x80013803, 0xe070101c, 0x80013903, 0xe0701020, 0x80013a03,
    0xe0701024, 0x80013b03, 0xe0701028, 0x80013c03, 0xe070102c, 0x80013d03, 0xe0701030, 0x80013e03,
    0xe0701034, 0x80013f03, 0xe0701038, 0x80014003, 0xe070103c, 0x80014103, 0xe0701040, 0x80014203,
    0xe0701044, 0x80014303, 0xe0701048, 0x80014403, 0xe070104c, 0x80014503, 0xe0701050, 0x80014603,
    0xe0701054, 0x80014703, 0xe0701058, 0x80014803, 0xe070105c, 0x80014903, 0xe0701060, 0x80014a03,
    0xe0701064, 0x80014b03, 0xe0701068, 0x80014c03, 0xe070106c, 0x80014d03, 0xe0701070, 0x80014e03,
    0xe0701074, 0x80014f03, 0xe0701078, 0x80015003, 0xe070107c, 0x80015103, 0xe0701080, 0x80015203,
    0xe0701084, 0x80015303, 0xe0701088, 0x80015403, 0xe070108c, 0x80015503, 0xe0701090, 0x80015603,
    0xe0701094, 0x80015703, 0xe0701098, 0x80015803, 0xe070109c, 0x80015903, 0xe07010a0, 0x80015a03,
    0xe07010a4, 0x80015b03, 0xe07010a8, 0x80015c03, 0xe07010ac, 0x80015d03, 0xe07010b0, 0x80015e03,
    0xe07010b4, 0x80015f03, 0xe07010b8, 0x80016003, 0xe07010bc, 0x80016103, 0xe07010c0, 0x80016203,
    0xe07010c4, 0x80016303, 0xe07010c8, 0x80016403, 0xe07010cc, 0x80016503, 0xe07010d0, 0x80016603,
    0xe07010d4, 0x80016703, 0xe07010d8, 0x80016803, 0xe07010dc, 0x80016903, 0xe07010e0, 0x80016a03,
    0xe07010e4, 0x80016b03, 0xe07010e8, 0x80016c03, 0xe07010ec, 0x80016d03, 0xe07010f0, 0x80016e03,
    0xe07010f4, 0x80016f03, 0xe07010f8, 0x80017003, 0xe07010fc, 0x80017103, 0xe0701100, 0x80017203,
    0xe0701104, 0x80017303, 0xe0701108, 0x80017403, 0xe070110c, 0x80017503, 0xe0701110, 0x80017603,
    0xe0701114, 0x80017703, 0xe0701118, 0x80017803, 0xe070111c, 0x80017903, 0xe0701120, 0x80017a03,
    0xe0701124, 0x80017b03, 0xe0701128, 0x80010a03, 0xe070112c, 0x80010b03, 0xe0701130, 0x80010c03,
    0xe0701134, 0x80010d03, 0xe0701138, 0x80010e03, 0xe070113c, 0x80010f03, 0xe0701140, 0x80011003,
    0xe0701144, 0x80011103, 0xe0701148, 0x80011203, 0xe070114c, 0x80011303, 0xe0701150, 0x80011403,
    0xe0701154, 0x80011503, 0xe0701158, 0x80011603, 0xe070115c, 0x80011703, 0xe0701160, 0x80011803,
    0xe0701164, 0x80011903, 0xe0701168, 0x80011a03, 0xe070116c, 0x80011b03, 0xe0701170, 0x80011c03,
    0xe0701174, 0x80011d03, 0xe0701178, 0x80011e03, 0xe070117c, 0x80011f03, 0xe0701180, 0x80012003,
    0xe0701184, 0x80012103, 0xe0701188, 0x80012203, 0xe070118c, 0x80012303, 0xe0701190, 0x80012403,
    0xe0701194, 0x80012503, 0xe0701198, 0x80012603, 0xe070119c, 0x80012703, 0xe07011a0, 0x80012803,
    0xe07011a4, 0x80012903, 0xe07011a8, 0x80012a03, 0xe07011ac, 0x80012b03, 0xe07011b0, 0x80012c03,
    0xe07011b4, 0x80012d03, 0xe07011b8, 0x80012e03, 0xe07011bc, 0x80012f03, 0xe07011c0, 0x80013003,
    0xe07011c4, 0x80013103, 0xbf810000,
};

// tests/execution/GlobalAtomics64.cpp (GlobalAtomicsCode): eleven 64-bit global atomics.
alignas(256) constexpr std::array<std::uint32_t, 159> GlobalAtomics64Code{
    0x34020083, 0x34040084, 0x340a0083, 0x7e2802ff, 0x05e471e1, 0x7e2a02ff, 0x05e471e1, 0x4a060280,
    0x4a080480, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0a80, 0xdd418000, 0x14080a03, 0xbf8c3f70,
    0xe0741000, 0x80011406, 0x4a0602ff, 0x00000200, 0x4a0804ff, 0x00000400, 0xe0381000, 0x80000a04,
    0xbf8c3f70, 0x4a0c0aff, 0x00000200, 0xdd458000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406,
    0x4a0602ff, 0x00000400, 0x4a0804ff, 0x00000800, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff,
    0x00000400, 0xdd498000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00000600,
    0x4a0804ff, 0x00000c00, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000600, 0xdd4d8000,
    0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00000800, 0x4a0804ff, 0x00001000,
    0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000800, 0xdd558000, 0x14080a03, 0xbf8c3f70,
    0xe0741000, 0x80011406, 0x4a0602ff, 0x00000a00, 0x4a0804ff, 0x00001400, 0xe0381000, 0x80000a04,
    0xbf8c3f70, 0x4a0c0aff, 0x00000a00, 0xdd598000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406,
    0x4a0602ff, 0x00000c00, 0x4a0804ff, 0x00001800, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff,
    0x00000c00, 0xdd5d8000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00000e00,
    0x4a0804ff, 0x00001c00, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00000e00, 0xdd618000,
    0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0x4a0602ff, 0x00001000, 0x4a0804ff, 0x00002000,
    0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff, 0x00001000, 0xdd658000, 0x14080a03, 0xbf8c3f70,
    0xe0741000, 0x80011406, 0x4a0602ff, 0x00001200, 0x4a0804ff, 0x00002400, 0xe0381000, 0x80000a04,
    0xbf8c3f70, 0x4a0c0aff, 0x00001200, 0xdd698000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406,
    0x4a0602ff, 0x00001400, 0x4a0804ff, 0x00002800, 0xe0381000, 0x80000a04, 0xbf8c3f70, 0x4a0c0aff,
    0x00001400, 0xdd6d8000, 0x14080a03, 0xbf8c3f70, 0xe0741000, 0x80011406, 0xbf810000
};

// tests/execution/LookBackScan.cpp (ScanCode): wave64 decoupled look-back; blocks spin on GLC dwordx2
// reads of the pairs their predecessors publish with 64-bit atomics.
alignas(256) constexpr std::array<std::uint32_t, 43> LookBackScanCode{
    0xbefe0481, 0x7e020281, 0x7e000280, 0xe0c85000, 0x80000100, 0xbf8c3f70, 0x7e200501, 0x34040282,
    0xe0301000, 0x80010302, 0x34080283, 0xbf8c3f70, 0xbf068010, 0x85118182, 0x7e0c0211, 0x7e0e0303,
    0xe1401000, 0x80020604, 0x7e100280, 0xbf068010, 0xbf85000f, 0x80928110, 0x8f138312, 0x7e120213,
    0xe0345000, 0x80020a09, 0xbf8c3f70, 0x7e28050a, 0x7e2a050b, 0xbf068014, 0xbf85fff7, 0x4a101015,
    0xbf068214, 0xbf850002, 0x80928112, 0xbf82fff2, 0x4a1a0708, 0x7e180282, 0xe1401000, 0x80020c04,
    0xe0701000, 0x80030802, 0xbf810000,
};

std::vector<std::uint32_t> Backing(512u, 0u);

ShaderRecompiler::SpirvTarget MetalTarget() {
    std::vector<std::uint32_t> capabilities;
    for (const auto capability : {spv::CapabilityShader, spv::CapabilityInt64, spv::CapabilityInt64Atomics, spv::CapabilityInt16, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityGroupNonUniformArithmetic, spv::CapabilitySignedZeroInfNanPreserve, spv::CapabilityImageQuery, spv::CapabilityFloat64}) {
        capabilities.push_back(static_cast<std::uint32_t>(capability));
    }
    std::ranges::sort(capabilities);
    // As the Metal device's Target() on the M3: 32-wide SIMD groups, 32 KiB of threadgroup memory.
    ShaderRecompiler::SpirvTarget target {0u, 0u, 32u, ShaderRecompiler::BdaAbi::Version, capabilities, {}, false, {1024u, 1024u, 1024u}, 1024u, 32768u, {}, {}};
    target.backend = ShaderRecompiler::ShaderBackend::Msl;
    return target;
}

struct Program {
    const char* name;
    std::span<const std::uint32_t> code;
    std::uint32_t threads;
    std::uint32_t waveSize;
    std::uint32_t ldsDwords;
    std::uint32_t userDataCount;
};

ShaderRecompiler::RecompileResult Recompile(const Program& program) {
    std::vector<std::uint32_t> userData(program.userDataCount, 0u);
    const auto address = reinterpret_cast<std::uintptr_t>(Backing.data());
    for (std::uint32_t buffer = 0; buffer + 4u <= program.userDataCount; buffer += 4u) {
        const std::array<std::uint32_t, 4> descriptor {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), static_cast<std::uint32_t>(Backing.size() * 4u), 0x31016facu};
        std::copy(descriptor.begin(), descriptor.end(), userData.begin() + buffer);
    }
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(program.code.data()), std::as_bytes(program.code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{program.threads, 1, 1}, program.ldsDwords, {true, true, true}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(program.code.data()), program.code, 0, {}},
        {program.waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        MetalTarget(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

std::size_t Count(const std::string& text, const std::string& needle) {
    std::size_t count = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) ++count;
    return count;
}

void Has(const Program& program, const std::string& source, const std::string& needle) {
    Require(source.find(needle) != std::string::npos, std::string(program.name) + ": the MSL lacks `" + needle + "`\n" + source);
}

void Lacks(const Program& program, const std::string& source, const std::string& needle) {
    Require(source.find(needle) == std::string::npos, std::string(program.name) + ": the MSL has `" + needle + "`\n" + source);
}

// The text of the brace block that opens at the first '{' at or after `from`.
std::string Block(const std::string& text, std::size_t from) {
    const auto open = text.find('{', from);
    Require(open != std::string::npos, "no block after offset " + std::to_string(from));
    int depth = 0;
    for (auto at = open; at < text.size(); ++at) {
        if (text[at] == '{') ++depth;
        if (text[at] == '}' && --depth == 0) return text.substr(open, at - open + 1);
    }
    throw std::runtime_error("unbalanced block after offset " + std::to_string(from));
}

std::string Squeeze(const std::string& text) {
    std::string out;
    for (const char c : text) {
        if (c != ' ' && c != '\n') out += c;
    }
    return out;
}

// MslLockLoop's shape, which keeps every emulated atomic free of intra-SIMD-group waits (Apple SIMD
// groups have no independent forward progress): each round, inside `if (!done)`, one lane elected by
// simd_is_first() try-locks once; acquire, the critical section, release and the unlocking exchange all
// run in that same `if` and nothing in it loops; the loop leaves through simd_any or the round limit's
// LoopLimit fault. Returns the number of lock loops.
std::size_t CheckLockLoops(const Program& program, const std::string& source) {
    std::size_t loops = 0;
    for (auto at = source.find("for (uint aps5_round"); at != std::string::npos; at = source.find("for (uint aps5_round", at + 1u)) {
        ++loops;
        const auto name = std::string(program.name) + " lock loop " + std::to_string(loops);
        const auto loop = Block(source, at);
        const auto flat = Squeeze(loop);
        const std::regex head(R"(^\{if\(!(aps5_done\w+|aps5_ia64\w+_done)\)\{if\(simd_is_first\(\)\)\{uintaps5_lk\w+=0u;if\(atomic_compare_exchange_weak_explicit\(([^,]+),&aps5_lk\w+,1u,memory_order_relaxed,memory_order_relaxed\)\)\{)");
        std::smatch match;
        Require(std::regex_search(flat, match, head), name + ": the round does not start with the elected try-lock\n" + loop);
        const std::string done = match[1];
        const std::string lock = match[2];
        const auto critical = Block(loop, loop.find("atomic_compare_exchange_weak_explicit("));
        const auto inner = Squeeze(critical);
        Require(inner.find("memory_order_acquire") != std::string::npos || inner.find("memory_order_seq_cst") != std::string::npos, name + ": no acquire fence after the lock is taken");
        const auto unlock = "(void)atomic_exchange_explicit(" + lock + ",0u,memory_order_relaxed);" + done + "=true;}";
        Require(inner.size() >= unlock.size() && inner.compare(inner.size() - unlock.size(), unlock.size(), unlock) == 0, name + ": the critical section does not end with the unlocking exchange (upstream 317a323ee) and " + done + " = true\n" + critical);
        for (const char* forbidden : {"for(", "while(", "simd_any(", "simd_ballot(", "simd_shuffle("}) {
            Require(inner.find(forbidden) == std::string::npos, name + ": `" + forbidden + "` inside the critical section\n" + critical);
        }
        Require(flat.find("if(!simd_any(!" + done + "))break;") != std::string::npos, name + ": the loop does not end once no lane of the SIMD group waits");
        Require(flat.find("==1048576u){if(simd_is_first())aps5_record_fault(aps5_driver.faultBuffer,6u,") != std::string::npos, name + ": no LoopLimit fault at the round limit");
    }
    Lacks(program, source, "while (atomic_exchange");
    Require(std::regex_search(source, std::regex(R"(while \(!?atomic_[a-z_]*\(aps5_lock)")) == false, std::string(program.name) + ": a lane spins on a lock word");
    return loops;
}

std::string Check(const Program& program) {
    const auto result = Recompile(program);
    Require(!result.msl.empty(), std::string(program.name) + ": no MSL for the Metal target");
    Require(result.msl.Reflection().abiVersion == ShaderRecompiler::MslAbi::Version, std::string(program.name) + ": unexpected MslAbi version");
    return std::string(result.msl.Source());
}

}

int main() {
    try {
        // v_readlane_b32 / v_writelane-free program: a shuffle per read; wave64 reads both halves.
        {
            const Program wave32 {"read_lane", ReadLaneCode, 32u, 32u, 0u, 8u};
            const auto source = Check(wave32);
            Has(wave32, source, "simd_shuffle(");
            Lacks(wave32, source, "aps5_lds");
            const Program wave64 {"read_lane_wave64", ReadLaneCode, 64u, 64u, 0u, 8u};
            const auto wide = Check(wave64);
            Require(Count(wide, "simd_shuffle(") >= 2u * Count(source, "simd_shuffle("), "read_lane_wave64: a wave64 lane read must shuffle both lane halves");
            Has(wave64, wide, "& 32u) != 0u ?");
        }
        // DPP8: lane selects within groups of 8, gated by the source lane's EXEC bit.
        for (const Program& program : {Program{"dpp8", Dpp8Wave32Code, 32u, 32u, 0u, 8u}, Program{"dpp8_wave64", Dpp8Wave64Code, 64u, 64u, 0u, 8u}}) {
            const auto source = Check(program);
            Has(program, source, "simd_shuffle(");
            Has(program, source, "& 4294967288u)");
        }
        // LDS beyond 32 KiB: the workgroup's slice of the device arena.
        {
            const Program program {"lds_beyond_device_limit", LdsBeyondCode, 64u, 32u, 16384u, 8u};
            const auto result = Recompile(program);
            const std::string source(result.msl.Source());
            Require(result.msl.Reflection().workgroupMemoryStrideDwords == 16386u, "lds_beyond_device_limit: the arena stride must be the LDS dwords plus the lock dword, rounded up to an even dword count (upstream af0af833c1)");
            Has(program, source, "device uint* aps5_wgm [[buffer(" + std::to_string(ShaderRecompiler::MslAbi::Index(ShaderRecompiler::MslAbi::BufferIndex::WorkgroupMemory)) + ")]]");
            Has(program, source, "device uint* aps5_lds = aps5_wgm + ");
            Has(program, source, "* 16386u;");
            Lacks(program, source, "threadgroup uint aps5_lds[");
            const Program within {"lds_within_device_limit", LdsBeyondCode, 64u, 32u, 1024u, 8u};
            const auto small = Recompile(within);
            Require(small.msl.Reflection().workgroupMemoryStrideDwords == 0u, "lds_within_device_limit: LDS that fits threadgroup memory must not use the arena");
            Has(within, std::string(small.msl.Source()), "threadgroup uint aps5_lds[1025];");
        }
        // ds_append / ds_consume: one atomic per wave on the counter, broadcast to the wave.
        {
            const Program program {"ds_append_consume", AppendConsumeCode, 32u, 32u, 256u, 4u};
            const auto source = Check(program);
            Has(program, source, "atomic_fetch_add_explicit(");
            Has(program, source, "atomic_fetch_sub_explicit(");
            Has(program, source, "popcount(");
            Has(program, source, "simd_shuffle(");
        }
        // 64-bit LDS atomics: every LDS atomic under the lock, which never spins in one invocation.
        {
            const Program program {"lds_atomics64", LdsAtomics64Code, 32u, 32u, 2244u, 8u};
            const auto source = Check(program);
            Has(program, source, "threadgroup uint aps5_lds[2245];");
            Has(program, source, "aps5_lds[2244] = 0u;");
            Has(program, source, "simd_is_first()");
            Has(program, source, "atomic_compare_exchange_weak_explicit(");
            Has(program, source, "if (!simd_any(!");
            Has(program, source, "== 1048576u)");
            Has(program, source, "aps5_record_fault(aps5_driver.faultBuffer, 6u");
            Has(program, source, "memory_order_acquire");
            Has(program, source, "memory_order_release");
            Lacks(program, source, "while (atomic_exchange");
            Require(CheckLockLoops(program, source) > 0u, "lds_atomics64: no lock loop");
        }
        // 64-bit device atomics: every one in the shared lock loop (forward progress, MslLockLoop).
        {
            const Program program {"global_atomics64", GlobalAtomics64Code, 64u, 32u, 0u, 10u};
            const auto source = Check(program);
            Require(CheckLockLoops(program, source) == 11u, "global_atomics64: expected eleven lock loops");
            Has(program, source, "aps5_atomic64_lock(aps5_driver.atomicLocks, ");
            Has(program, source, "atomic_max_explicit(reinterpret_cast<device atomic_ulong*>(");
            Lacks(program, source, "coherent(device) device const ulong*");
        }
        // Look-back scan (wave64, two lanes per invocation): the publishing atomics are lock loops; the
        // spinning reader never takes a lock and its GLC dwordx2 read is a bounded atomic re-read.
        {
            const Program program {"look_back_scan", LookBackScanCode, 64u, 64u, 0u, 16u};
            const auto source = Check(program);
            Require(CheckLockLoops(program, source) >= 2u, "look_back_scan: expected the publishing lock loops");
            Has(program, source, ".read2(");
            Has(program, source, "for (uint attempt = 0u; attempt < 64u; ++attempt) {");
            Lacks(program, source, "coherent(device) device const ulong*");
            Has(program, source, "aps5_loop_limit");
        }
        std::puts("msl wave/lds emission check passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
