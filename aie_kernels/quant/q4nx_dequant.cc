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
constexpr int STEPS = CT_K / SS;
constexpr int N8 = M_TILE / TT;

static_assert(TT == 8, "bfp16ebs8 shares one exponent across 8 values");
static_assert(SS == 8, "one step is an 8x8 tile, transposed in place");
static_assert(PR * SS == 128, "one 512-bit load holds 128 nibbles");
static_assert(M_TILE % PR == 0, "rows must divide into PR-row groups");
static_assert(K_TILE % CT_K == 0, "k tile must be a multiple of the k slice");
static_assert(CT_K % SS == 0, "k slice must be a multiple of s");
// One scale and min are loaded per step, so the s columns of a step must not
// straddle a quantization group.
static_assert(GROUP % SS == 0, "a group must not split a k step");
static_assert(K_TILE % GROUP == 0, "k tile must hold whole groups");

// The byte offset of each SS-wide k step's scale row, tabulated once: GROUP
// need not be a power of two, and dividing by it in the inner loop would be a
// __muldi3 libcall.
constexpr int K_STEPS = K_TILE / SS;

struct GroupRow {
  uint16_t v[K_STEPS];
};

constexpr GroupRow group_row() {
  GroupRow t{};
  for (int s = 0; s < K_STEPS; s++)
    t.v[s] = (uint16_t)(s * SS / GROUP * M_TILE * sizeof(bfloat16));
  return t;
}

constexpr GroupRow GROW = group_row();
static_assert(SCALES * sizeof(bfloat16) <= 65536,
              "scale row offset must fit in 16 bits");

// A load holds 8 k of 16 n, with n 0-7 in the even words and n 8-15 in the
// odd ones. The shuffle mode that gathers one half is a runtime operand, so
// both halves share one loop body.
struct HalfMode {
  uint8_t v[N8];
};

constexpr HalfMode half_mode() {
  HalfMode t{};
  for (int n8 = 0; n8 < N8; n8++)
    t.v[n8] = (uint8_t)(T32_16x2_lo + (n8 & 1));
  return t;
}

constexpr HalfMode HMODE = half_mode();

// 8 bf16 repeated to fill 32 lanes. Two 128-bit shuffles, where a concat
// lowers to a chain of vshifts.
aie::vector<bfloat16, 32> rep8(const bfloat16 *p) {
  aie::vector<uint32_t, 16> v =
      aie::load_v<TT>(p).template grow<32>().template cast_to<uint32_t>();
  v = ::shuffle(v, v, T128_2x4_lo);
  v = ::shuffle(v, v, T128_2x4_lo);
  return v.template cast_to<bfloat16>();
}

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

  // The output is contiguous in (ks, n8, step) order, so one stream writes
  // it all; a second stream would spill the shared sf register every step.
  aie::block_vector_output_buffer_stream<bfp16ebs8, TILE_VALUES> s_out(out);

  for (int ks = 0; ks < K_TILE / CT_K; ks++) {
    const uint8_t *q_it = qs + ks * CT_K * PR / 2;
    const uint8_t *s_it = (const uint8_t *)scales;
    const uint16_t *row_it = GROW.v + ks * STEPS;
    const uint8_t *mode_it = HMODE.v;
    // The (n8, step) loop is flattened, since CT_K = 16 leaves only two steps
    // per n8, and the address unit walks every cursor so the scalar ALU
    // stays free. Both halves of a 16-n group read the same nibble bytes.
    dims_3d_t q_dims =
        dims_3d_from_steps(STEPS, SS * PR / 2, 2, 0, K_TILE * PR / 2);
    dims_2d_t s_dims = dims_2d_from_steps(STEPS, 0, TT * sizeof(bfloat16));
    dims_2d_t row_dims = dims_2d_from_steps(STEPS, sizeof(uint16_t), 0);
    dims_2d_t mode_dims = dims_2d_from_steps(STEPS, 0, 1);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(N8 * STEPS, N8 * STEPS)
    for (int j = 0; j < N8 * STEPS; j++) {
      aie::vector<uint32_t, 16> w = aie::load_v<16>((const uint32_t *)q_it);
      aie::vector<uint32_t, 16> half = ::shuffle(w, w, *mode_it);
      aie::vector<uint4, TT * SS> q =
          half.template extract<8>(0).template cast_to<uint4>();
      aie::accum<accfloat, TT * SS> qf;
      qf.from_vector(aie::to_float(q, 0));
      aie::vector<bfloat16, TT * SS> qb = qf.template to_vector<bfloat16>();

      // Indexed k*8 + n, so the 8 scales of this n8 repeat across k.
      const bfloat16 *sp = (const bfloat16 *)(s_it + *row_it);
      aie::vector<bfloat16, 32> sc = rep8(sp);
      aie::vector<bfloat16, 32> mn = rep8(sp + SCALES);
      // min + scale * quant, with the min seeded into the accumulator.
      aie::accum<accfloat, TT * SS> acc(aie::concat(mn, mn));
      acc = aie::mac(acc, aie::concat(sc, sc), qb);
      // The transpose makes it n*8 + k, so the 8 values sharing an exponent
      // are 8 k for one n. It must precede the conversion, which fuses them.
      aie::vector<bfloat16, TT * SS> kn = acc.template to_vector<bfloat16>();
      aie::accum<accfloat, TT * SS> nk(aie::transpose(kn, SS, TT));
      s_out << nk.template to_vector<bfp16ebs8>();

      q_it = add_3d_byte(q_it, q_dims);
      s_it = add_2d_byte(s_it, s_dims);
      row_it = add_2d_byte(row_it, row_dims);
      mode_it = add_2d_byte(mode_it, mode_dims);
    }
  }

  aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C"
