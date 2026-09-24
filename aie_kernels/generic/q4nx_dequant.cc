//===- q4nx_dequant.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dequantize one q4nx block and emit it as bfp16ebs8 in the order
// iron.operators.flm.GEMM reads its B operand. Geometry arrives as -D flags;
// the Q4NX_ prefix avoids R, S and T, which collide with template parameters
// in the aie2p built-in headers.
#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#if !defined(Q4NX_M_TILE) || !defined(Q4NX_K_TILE) || !defined(Q4NX_GROUP) ||  \
    !defined(Q4NX_CT_K) || !defined(Q4NX_S) || !defined(Q4NX_T)
#error "q4nx_dequant.cc needs its geometry -D defined"
#endif

namespace {

constexpr int M_TILE = Q4NX_M_TILE;
constexpr int K_TILE = Q4NX_K_TILE;
constexpr int GROUP = Q4NX_GROUP;
constexpr int CT_K = Q4NX_CT_K;
constexpr int SS = Q4NX_S;
constexpr int TT = Q4NX_T;

// n rows in one vector: 128 nibbles is one 512-bit load and the file stores 16
// n per k, so a load spans 8 k.
constexpr int PR = 16;

constexpr int SCALES = M_TILE * K_TILE / GROUP;
constexpr int TILE_VALUES = SS * TT;
constexpr int BLOCKS_PER_RUN = (CT_K / SS) * TT;
constexpr int BLOCKS_PER_KSLICE = (M_TILE / TT) * BLOCKS_PER_RUN;
// bfp16ebs8 pointer arithmetic counts bytes, not blocks (llvm-aie#1232).
constexpr int BLOCK_BYTES = TT + 1;

static_assert(TT == 8, "bfp16ebs8 shares one exponent across 8 values");
static_assert(SS == 8, "the column unroll and the interleave tree are 8 wide");
static_assert(PR * SS == 128, "one 512-bit load holds 128 nibbles");
static_assert(M_TILE % PR == 0, "rows must divide into PR-row groups");
static_assert(K_TILE % CT_K == 0, "k tile must be a multiple of the k slice");
static_assert(CT_K % SS == 0, "k slice must be a multiple of s");
// One scale and min are loaded per i, so the s columns of a step must not
// straddle a quantization group.
static_assert(GROUP % SS == 0, "a group must not split a k step");
static_assert(K_TILE % GROUP == 0, "k tile must hold whole groups");

// The quantization group of each SS-wide k step. GROUP need not be a power of
// two, and a constant divide that is not becomes a magic multiply, which on a
// 32-bit target needs the 64-bit __muldi3 -- a libcall in the inner loop, and
// a vectorization barrier. The step index is already a linear function of the
// loop counters, so the quotients are just tabulated once.
constexpr int K_STEPS = K_TILE / SS;

struct GroupOfStep {
  uint8_t v[K_STEPS];
};

constexpr GroupOfStep group_of_step() {
  GroupOfStep t{};
  for (int s = 0; s < K_STEPS; s++)
    t.v[s] = (uint8_t)(s * SS / GROUP);
  return t;
}

constexpr GroupOfStep GRP = group_of_step();
static_assert(K_TILE / GROUP <= 256, "group index must fit in a byte");

} // namespace

extern "C" {

void q4nx_dequant_bfp(const uint8_t *__restrict qw, bfp16ebs8 *__restrict out) {
  event0();
  // The mode is core-global and sticky, so a kernel that converts must pick
  // one. Floor is what a core powers up in, and matches the truncation the
  // operator's byte-exact reference models.
  aie::rounding_mode saved_rounding =
      aie::swap_rounding(aie::rounding_mode::floor);

  const bfloat16 *scales = (const bfloat16 *)qw;
  const bfloat16 *mins = scales + SCALES;
  // uint4 is `unsigned _BitInt(4)`, whose sizeof is 1, so a uint4 pointer
  // steps bytes. Walk bytes and cast at the load.
  const uint8_t *qs = (const uint8_t *)(mins + SCALES);
  static_assert(sizeof(uint4) == 1,
                "uint4 pointer arithmetic is assumed to step bytes");

  for (int ks = 0; ks < K_TILE / CT_K; ks++) {
    for (int row = 0; row < M_TILE; row += PR) {
      // Two runs per 16-row group, one per 8 n, each contiguous in the
      // dense buffer.
      bfp16ebs8 *lo = out + BLOCK_BYTES * (ks * BLOCKS_PER_KSLICE +
                                           (row / TT) * BLOCKS_PER_RUN);
      aie::block_vector_output_buffer_stream<bfp16ebs8, TILE_VALUES> s_lo(lo);
      aie::block_vector_output_buffer_stream<bfp16ebs8, TILE_VALUES> s_hi(
          lo + BLOCK_BYTES * BLOCKS_PER_RUN);

      const uint8_t *q_it =
          qs + ((row / PR) * K_TILE * PR + ks * CT_K * PR) / 2;

      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_RANGE(CT_K / SS, CT_K / SS)
      for (int i = 0; i < CT_K / SS; i++) {
        const int grp = GRP.v[ks * (CT_K / SS) + i];

        aie::vector<uint4, PR * SS> q =
            aie::load_v<PR * SS>((const uint4 *)q_it);
        q_it += PR * SS / 2;
        aie::accum<accfloat, PR * SS> qf;
        qf.from_vector(aie::to_float(q, 0));
        aie::vector<bfloat16, PR * SS> qb = qf.template to_vector<bfloat16>();

        aie::vector<bfloat16, PR> sc =
            aie::load_v<PR>(scales + grp * M_TILE + row);
        aie::accum<accfloat, PR> mn;
        mn.from_vector(aie::load_v<PR>(mins + grp * M_TILE + row));

        // min + scale * quant, with the min seeded into the accumulator
        // so each column costs one mac.
        aie::vector<bfloat16, PR> col[SS];
#pragma clang loop unroll(full)
        for (int c = 0; c < SS; c++) {
          aie::accum<accfloat, PR> acc = mn;
          acc = aie::mac(acc, sc, qb.template extract<PR>(c));
          col[c] = acc.template to_vector<bfloat16>();
        }

        auto z01 = aie::interleave_zip(col[0], col[1], TT);
        auto z23 = aie::interleave_zip(col[2], col[3], TT);
        auto z45 = aie::interleave_zip(col[4], col[5], TT);
        auto z67 = aie::interleave_zip(col[6], col[7], TT);
        // Indexed k*8 + n. The transpose makes it n*8 + k, so the 8
        // values sharing an exponent are 8 k for one n. It must precede
        // the conversion, which fuses them.
        auto up = aie::concat(z01.first, z23.first, z45.first, z67.first);
        auto dn = aie::concat(z01.second, z23.second, z45.second, z67.second);

        aie::accum<accfloat, TILE_VALUES> a_lo(aie::transpose(up, TT, SS));
        aie::accum<accfloat, TILE_VALUES> a_hi(aie::transpose(dn, TT, SS));
        s_lo << a_lo.template to_vector<bfp16ebs8>();
        s_hi << a_hi.template to_vector<bfp16ebs8>();
      }
    }
  }

  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
