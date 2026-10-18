// Guest programs of agc_metal_wave64_lane_local and agc_metal_wave64_lane_local_plan (DECISIONS D62).
// llvm-mc -arch=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64; user data s[0:3] input V#, s[4:7]
// output V#, s8 threads per group, s9 workgroup id, v0 thread id; 8 output dwords per thread.
#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace Wave64LaneLocal {

/*
  s_mul_i32 s12, s9, s8
  v_add_nc_u32 v1, s12, v0
  v_lshlrev_b32 v2, 2, v1
  buffer_load_dword v3, v2, s[0:3], 0 offen
  v_mul_u32_u24 v20, 32, v1
  s_waitcnt vmcnt(0)
  v_mov_b32 v10, 0
  v_mov_b32 v11, 0
  v_mov_b32 v12, 0
  v_mov_b32 v13, 0
  // if (x & 1) A = x * 3 else A = x + 7   (if/else with execz skips)
  v_and_b32 v4, 1, v3
  v_cmp_ne_u32 vcc, 0, v4
  s_and_saveexec_b64 s[14:15], vcc
  s_xor_b64 s[14:15], exec, s[14:15]
  s_cbranch_execz L_else
  v_mul_lo_u32 v10, v3, 3
  buffer_store_dword v3, v20, s[4:7], 0 offen offset:20
L_else:
  s_or_saveexec_b64 s[14:15], s[14:15]
  s_xor_b64 exec, exec, s[14:15]
  s_cbranch_execz L_endif
  v_add_nc_u32 v10, 7, v3
L_endif:
  s_or_b64 exec, exec, s[14:15]
  // loop: i < (x & 7): accumulate, store i inside the loop
  v_and_b32 v5, 7, v3
  s_mov_b64 s[16:17], exec
L_loop:
  v_cmp_lt_u32 vcc, v11, v5
  s_and_b64 exec, exec, vcc
  s_cbranch_execz L_loop_end
  v_add_nc_u32 v11, 1, v11
  v_add_nc_u32 v12, v12, v3
  v_xor_b32 v12, v12, v11
  buffer_store_dword v11, v20, s[4:7], 0 offen offset:24
  s_branch L_loop
L_loop_end:
  s_mov_b64 exec, s[16:17]
  // own-bit mask: lanes with x % 3 == 0
  v_mul_hi_u32 v6, v3, 0xaaaaaaab
  v_lshrrev_b32 v6, 1, v6
  v_mul_lo_u32 v6, v6, 3
  v_sub_nc_u32 v6, v3, v6
  v_cmp_eq_u32 s[18:19], 0, v6
  v_cndmask_b32_e64 v13, 5, 9, s[18:19]
  // nested: if (x > 1000) { if (x & 2) B = 1 else B = 2 }
  v_mov_b32 v14, 0
  v_cmp_lt_u32 vcc, 1000, v3
  s_and_saveexec_b64 s[20:21], vcc
  s_cbranch_execz L_outer_end
  v_and_b32 v7, 2, v3
  v_cmp_ne_u32 vcc, 0, v7
  s_and_saveexec_b64 s[22:23], vcc
  s_cbranch_execz L_inner_end
  v_mov_b32 v14, 1
L_inner_end:
  s_or_b64 exec, exec, s[22:23]
  v_add_nc_u32 v14, 2, v14
L_outer_end:
  s_or_b64 exec, exec, s[20:21]
  // do-while: v16 counts to ((x >> 3) & 3) + 1 (execnz back edge)
  v_bfe_u32 v8, v3, 3, 2
  v_add_nc_u32 v8, 1, v8
  v_mov_b32 v16, 0
  s_mov_b64 s[24:25], exec
L_do:
  v_add_nc_u32 v16, 1, v16
  v_cmp_lt_u32 vcc, v16, v8
  s_and_b64 exec, exec, vcc
  s_cbranch_execnz L_do
  s_mov_b64 exec, s[24:25]
  buffer_store_dword v16, v20, s[4:7], 0 offen offset:28
  buffer_store_dword v10, v20, s[4:7], 0 offen
  buffer_store_dword v11, v20, s[4:7], 0 offen offset:4
  buffer_store_dword v12, v20, s[4:7], 0 offen offset:8
  buffer_store_dword v13, v20, s[4:7], 0 offen offset:12
  v_lshlrev_b32 v14, 0, v14
  buffer_store_dword v14, v20, s[4:7], 0 offen offset:16
  s_endpgm
*/
alignas(256) constexpr std::array<std::uint32_t, 85> BranchesCode{
    0x930c0809u, 0x4a02000cu, 0x34040282u, 0xe0301000u, 0x80000302u, 0x162802a0u, 0xbf8c3f70u, 0x7e140280u,
    0x7e160280u, 0x7e180280u, 0x7e1a0280u, 0x36080681u, 0x7d8a0880u, 0xbe8e246au, 0x898e0e7eu, 0xbf880004u,
    0xd569000au, 0x02010703u, 0xe0701014u, 0x80010314u, 0xbe8e250eu, 0x89fe0e7eu, 0xbf880001u, 0x4a140687u,
    0x88fe0e7eu, 0x360a0687u, 0xbe90047eu, 0x7d820b0bu, 0x87fe6a7eu, 0xbf880006u, 0x4a161681u, 0x4a18070cu,
    0x3a18170cu, 0xe0701018u, 0x80010b14u, 0xbf82fff7u, 0xbefe0410u, 0xd56a0006u, 0x0201ff03u, 0xaaaaaaabu,
    0x2c0c0c81u, 0xd5690006u, 0x02010706u, 0x4c0c0d03u, 0xd4c20012u, 0x02020c80u, 0xd501000du, 0x00491285u,
    0x7e1c0280u, 0x7d8206ffu, 0x000003e8u, 0xbe94246au, 0xbf880007u, 0x360e0682u, 0x7d8a0e80u, 0xbe96246au,
    0xbf880001u, 0x7e1c0281u, 0x88fe167eu, 0x4a1c1c82u, 0x88fe147eu, 0xd5480008u, 0x02090703u, 0x4a101081u,
    0x7e200280u, 0xbe98047eu, 0x4a202081u, 0x7d821110u, 0x87fe6a7eu, 0xbf89fffcu, 0xbefe0418u, 0xe070101cu,
    0x80011014u, 0xe0701000u, 0x80010a14u, 0xe0701004u, 0x80010b14u, 0xe0701008u, 0x80010c14u, 0xe070100cu,
    0x80010d14u, 0x341c1c80u, 0xe0701010u, 0x80010e14u, 0xbf810000u
};

/*
  s_mul_i32 s12, s9, s8
  v_add_nc_u32 v1, s12, v0
  v_lshlrev_b32 v2, 2, v1
  buffer_load_dword v3, v2, s[0:3], 0 offen
  v_mul_u32_u24 v20, 32, v1
  s_waitcnt vmcnt(0)
  v_mov_b32 v10, 0
  // single wave: LDS exchange without s_barrier (lockstep), lane ^ 33 then lane ^ 1
  v_lshlrev_b32 v6, 2, v0
  ds_write_b32 v6, v3
  v_xor_b32 v7, 33, v0
  v_lshlrev_b32 v7, 2, v7
  s_waitcnt lgkmcnt(0)
  ds_read_b32 v15, v7
  s_waitcnt lgkmcnt(0)
  v_add_nc_u32 v16, 1, v15
  ds_write_b32 v6, v16
  v_xor_b32 v8, 1, v0
  v_lshlrev_b32 v8, 2, v8
  s_waitcnt lgkmcnt(0)
  ds_read_b32 v17, v8
  s_waitcnt lgkmcnt(0)
  buffer_store_dword v15, v20, s[4:7], 0 offen
  buffer_store_dword v17, v20, s[4:7], 0 offen offset:4
  s_endpgm
*/
alignas(256) constexpr std::array<std::uint32_t, 31> WaveLdsCode{
    0x930c0809u, 0x4a02000cu, 0x34040282u, 0xe0301000u, 0x80000302u, 0x162802a0u, 0xbf8c3f70u, 0x7e140280u,
    0x340c0082u, 0xd8340000u, 0x00000306u, 0x3a0e00a1u, 0x340e0e82u, 0xbf8cc07fu, 0xd8d80000u, 0x0f000007u,
    0xbf8cc07fu, 0x4a201e81u, 0xd8340000u, 0x00001006u, 0x3a100081u, 0x34101082u, 0xbf8cc07fu, 0xd8d80000u,
    0x11000008u, 0xbf8cc07fu, 0xe0701000u, 0x80010f14u, 0xe0701004u, 0x80011114u, 0xbf810000u
};

/*
  s_mul_i32 s12, s9, s8
  v_add_nc_u32 v1, s12, v0
  v_lshlrev_b32 v2, 2, v1
  buffer_load_dword v3, v2, s[0:3], 0 offen
  v_mul_u32_u24 v20, 32, v1
  s_waitcnt vmcnt(0)
  v_mov_b32 v10, 0
  // two waves: LDS exchange with the other wave through s_barrier (lane ^ 64)
  v_lshlrev_b32 v6, 2, v0
  ds_write_b32 v6, v3
  v_xor_b32 v7, 64, v0
  v_lshlrev_b32 v7, 2, v7
  s_waitcnt lgkmcnt(0)
  s_barrier
  ds_read_b32 v15, v7
  s_waitcnt lgkmcnt(0)
  v_and_b32 v4, 1, v15
  v_cmp_ne_u32 vcc, 0, v4
  s_and_saveexec_b64 s[14:15], vcc
  s_cbranch_execz L_end
  v_mul_lo_u32 v15, v15, 5
L_end:
  s_or_b64 exec, exec, s[14:15]
  buffer_store_dword v15, v20, s[4:7], 0 offen
  s_endpgm
*/
alignas(256) constexpr std::array<std::uint32_t, 28> TwoWaveLdsCode{
    0x930c0809u, 0x4a02000cu, 0x34040282u, 0xe0301000u, 0x80000302u, 0x162802a0u, 0xbf8c3f70u, 0x7e140280u,
    0x340c0082u, 0xd8340000u, 0x00000306u, 0x3a0e00c0u, 0x340e0e82u, 0xbf8cc07fu, 0xbf8a0000u, 0xd8d80000u,
    0x0f000007u, 0xbf8cc07fu, 0x36081e81u, 0x7d8a0880u, 0xbe8e246au, 0xbf880002u, 0xd569000fu, 0x02010b0fu,
    0x88fe0e7eu, 0xe0701000u, 0x80010f14u, 0xbf810000u
};

/*
  s_mul_i32 s12, s9, s8
  v_add_nc_u32 v1, s12, v0
  v_lshlrev_b32 v2, 2, v1
  buffer_load_dword v3, v2, s[0:3], 0 offen
  v_mul_u32_u24 v20, 32, v1
  s_waitcnt vmcnt(0)
  v_mov_b32 v10, 0
  // negative: s_bcnt1 of EXEC inside an if (needs the other lanes' bits)
  v_and_b32 v4, 1, v3
  v_cmp_ne_u32 vcc, 0, v4
  s_and_saveexec_b64 s[14:15], vcc
  s_cbranch_execz L_end
  s_bcnt1_i32_b64 s16, exec
  v_mov_b32 v10, s16
L_end:
  s_or_b64 exec, exec, s[14:15]
  buffer_store_dword v10, v20, s[4:7], 0 offen
  s_endpgm
*/
alignas(256) constexpr std::array<std::uint32_t, 18> ExecCountCode{
    0x930c0809u, 0x4a02000cu, 0x34040282u, 0xe0301000u, 0x80000302u, 0x162802a0u, 0xbf8c3f70u, 0x7e140280u,
    0x36080681u, 0x7d8a0880u, 0xbe8e246au, 0xbf880002u, 0xbe90107eu, 0x7e140210u, 0x88fe0e7eu, 0xe0701000u,
    0x80010a14u, 0xbf810000u
};

/*
  s_mul_i32 s12, s9, s8
  v_add_nc_u32 v1, s12, v0
  v_lshlrev_b32 v2, 2, v1
  buffer_load_dword v3, v2, s[0:3], 0 offen
  v_mul_u32_u24 v20, 32, v1
  s_waitcnt vmcnt(0)
  v_mov_b32 v10, 0
  // negative: s_cbranch_vccz over code that runs for every EXEC lane
  v_cmp_gt_u32 vcc, 100, v3
  s_cbranch_vccz L_end
  v_mov_b32 v10, 1
L_end:
  buffer_store_dword v10, v20, s[4:7], 0 offen
  s_endpgm
*/
alignas(256) constexpr std::array<std::uint32_t, 15> VcczCode{
    0x930c0809u, 0x4a02000cu, 0x34040282u, 0xe0301000u, 0x80000302u, 0x162802a0u, 0xbf8c3f70u, 0x7e140280u,
    0x7d8806ffu, 0x00000064u, 0xbf860001u, 0x7e140281u, 0xe0701000u, 0x80010a14u, 0xbf810000u
};

/*
  s_mul_i32 s12, s9, s8
  v_add_nc_u32 v1, s12, v0
  v_lshlrev_b32 v2, 2, v1
  buffer_load_dword v3, v2, s[0:3], 0 offen
  v_mul_u32_u24 v20, 32, v1
  s_waitcnt vmcnt(0)
  v_mov_b32 v10, 0
  v_mov_b32 v11, 0
  v_mov_b32 v12, 0
  // row prefix sums (DPP), the other row's lane 15 (permlanex16), the wave sum from lanes 31 and 63
  v_mov_b32 v4, v3
  v_add_nc_u32_dpp v4, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_nc_u32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_nc_u32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_nc_u32_dpp v4, v4, v4 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_permlanex16_b32 v5, v4, -1, -1
  v_add_nc_u32 v6, v4, v5
  v_readlane_b32 s16, v6, 31
  v_readlane_b32 s17, v6, 63
  s_add_u32 s18, s16, s17
  v_add_nc_u32 v10, s18, v6
  // ds_swizzle (swap 16 within 32) and ds_bpermute (lane ^ 5 within 32)
  ds_swizzle_b32 v13, v3 offset:swizzle(SWAP,16)
  v_xor_b32 v7, 5, v0
  v_lshlrev_b32 v7, 2, v7
  ds_bpermute_b32 v14, v7, v3
  s_waitcnt lgkmcnt(0)
  v_xor_b32 v10, v10, v13
  v_add_nc_u32 v10, v10, v14
  // readfirstlane of the whole wave and of the odd lanes only
  v_readfirstlane_b32 s19, v3
  v_mov_b32 v11, s19
  v_and_b32 v4, 1, v3
  v_cmp_ne_u32 vcc, 0, v4
  s_and_saveexec_b64 s[14:15], vcc
  s_cbranch_execz L_end
  v_readfirstlane_b32 s20, v3
  v_mov_b32 v12, s20
L_end:
  s_or_b64 exec, exec, s[14:15]
  buffer_store_dword v10, v20, s[4:7], 0 offen
  buffer_store_dword v11, v20, s[4:7], 0 offen offset:4
  buffer_store_dword v12, v20, s[4:7], 0 offen offset:8
  s_endpgm
*/
alignas(256) constexpr std::array<std::uint32_t, 53> CrossLaneCode{
    0x930c0809u, 0x4a02000cu, 0x34040282u, 0xe0301000u, 0x80000302u, 0x162802a0u, 0xbf8c3f70u, 0x7e140280u,
    0x7e160280u, 0x7e180280u, 0x7e080303u, 0x4a0808fau, 0xff091104u, 0x4a0808fau, 0xff091204u, 0x4a0808fau,
    0xff091404u, 0x4a0808fau, 0xff091804u, 0xd7780005u, 0x03058304u, 0x4a0c0b04u, 0xd7600010u, 0x02013f06u,
    0xd7600011u, 0x02017f06u, 0x80121110u, 0x4a140c12u, 0xd8d4401fu, 0x0d000003u, 0x3a0e0085u, 0x340e0e82u,
    0xdacc0000u, 0x0e000307u, 0xbf8cc07fu, 0x3a141b0au, 0x4a141d0au, 0x7e260503u, 0x7e160213u, 0x36080681u,
    0x7d8a0880u, 0xbe8e246au, 0xbf880002u, 0x7e280503u, 0x7e180214u, 0x88fe0e7eu, 0xe0701000u, 0x80010a14u,
    0xe0701004u, 0x80010b14u, 0xe0701008u, 0x80010c14u, 0xbf810000u
};

struct Case {
    const char* name;
    std::span<const std::uint32_t> code;
    std::uint32_t threads;
    std::uint32_t ldsDwords;
    // Expected at one lane per invocation (else two lanes per invocation).
    bool laneLocal;
};

inline const std::array<Case, 13> Cases{{
    {"Branches/64", BranchesCode, 64u, 0u, true},
    {"Branches/48", BranchesCode, 48u, 0u, true},
    {"Branches/128", BranchesCode, 128u, 0u, true},
    {"Branches/256", BranchesCode, 256u, 0u, true},
    {"WaveLds/64", WaveLdsCode, 64u, 64u, true},
    {"TwoWaveLds/128", TwoWaveLdsCode, 128u, 128u, true},
    // One-wave threadgroups exchange through threadgroup memory: s_bcnt1 of EXEC (exact ballots,
    // voted branches), vccz over all EXEC lanes (a voted branch), DPP/permlanex16/swizzle/bpermute
    // within the half plus v_readlane/v_readfirstlane across it.
    {"ExecCount/64", ExecCountCode, 64u, 0u, true},
    {"Vccz/64", VcczCode, 64u, 0u, true},
    {"CrossLane/64", CrossLaneCode, 64u, 0u, true},
    // Several waves per threadgroup cannot exchange: two lanes per invocation.
    {"ExecCount/128", ExecCountCode, 128u, 0u, false},
    {"Vccz/128", VcczCode, 128u, 0u, false},
    {"CrossLane/128", CrossLaneCode, 128u, 0u, false},
    {"CrossLane/48", CrossLaneCode, 48u, 0u, false},
}};

}
