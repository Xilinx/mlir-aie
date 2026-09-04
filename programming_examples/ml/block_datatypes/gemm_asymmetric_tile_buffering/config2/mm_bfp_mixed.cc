//===- mm_bfp_mixed.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ATB microkernel, all-bfp16ebs8 variant: A, B and C tiles are all
// v8bfp16ebs8. One flat post-RA-pipelined loop over all 12 (16x16) C blocks
// of the per-call stripe, cross-block pipelined: accumulators init from the
// previous iteration's prefetched C vectors, run 16 expanded k-steps,
// prefetch the next block's C, push the results. Measured II=77 (= RecMII),
// NS=2, ZOL (m=192, k=128, n=96, rho=6). Requires -mllvm
// -aie-postpipeliner-maxii=120 -aie-postpipeliner-maxtry-ii=50 (via
// compile_flags in n32_core.py): the loop's ResMII=68 exceeds the default
// max II of 60.

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

#ifndef DIM_RHO
#define DIM_RHO 6
#endif

// Per-core L1 C tile is (m, n); the L1 A sub-tile is (m/rho, k). The kernel
// is invoked rho times per (A-sub-tile, B-tile) pair and rotates over the
// rho C stripes via g_counter.
constexpr int m = DIM_M;
constexpr int k = DIM_K;
constexpr int n = DIM_N;
constexpr int rho = DIM_RHO;
constexpr int m_a = m / rho;
constexpr int nbc = n / 16;
constexpr int nblk = (m_a / 16) * nbc;

static_assert(m % rho == 0 && m_a % 16 == 0,
              "m must be a multiple of 16*rho (A sub-tile rows)");
static_assert(k == 128,
              "k must be 128: the pipelined flat loop carries 16 expanded "
              "k-steps");
static_assert(n % 16 == 0,
              "n must be a multiple of 16 (2 column blocks of 8 in the MAC "
              "loop)");

extern "C" {

// MATMUL_ONLY / ZERO_ONLY gates — distinct ExternalFunction .o builds of
// this TU emit exactly one symbol. Without any macro, both are emitted.
#if !defined(MATMUL_ONLY) && !defined(ZERO_ONLY)
#define MATMUL_ONLY
#define ZERO_ONLY
#endif

static int g_counter = 0;

#ifdef MATMUL_ONLY
void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA_in,
                             bfp16ebs8 *__restrict pB_in,
                             bfp16ebs8 *__restrict pC_in) {
  event0();
  // Round-to-nearest-even (Peano defaults to floor on converts).
  aie::set_rounding(aie::rounding_mode::conv_even);

  // sizeof(bfp16ebs8) == 1 on Peano: pointer arithmetic below is byte
  // arithmetic (64-element block = 72 bytes; C stripe = m_a*n/8*9 bytes).
  const bfp16ebs8 __aie_dm_resource_a *__restrict pA =
      (const bfp16ebs8 __aie_dm_resource_a *)pA_in;
  const bfp16ebs8 __aie_dm_resource_b *__restrict pB =
      (const bfp16ebs8 __aie_dm_resource_b *)pB_in;
  bfp16ebs8 __aie_dm_resource_c *__restrict pC =
      (bfp16ebs8 __aie_dm_resource_c *)pC_in + g_counter * (m_a * n / 8 * 9);
  g_counter = (g_counter == rho - 1) ? 0 : g_counter + 1;

  using AStream = aie::block_vector_restrict_input_buffer_stream<
      bfp16ebs8, 64, aie_dm_resource::a>;
  using BStream = aie::block_vector_restrict_input_buffer_stream<
      bfp16ebs8, 64, aie_dm_resource::b>;
  using CInStream = aie::block_vector_restrict_input_buffer_stream<
      bfp16ebs8, 64, aie_dm_resource::c>;
  using COutStream =
      aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c>;
  using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;

  // Persistent C streams over the stripe in linear block order. The prefetch
  // pops run one block ahead: the last iteration reads 288 bytes past the
  // stripe (unused, stays in tile data memory).
  CInStream cin(pC);
  COutStream cout(pC);

  // Prologue: prefetch C-in vectors for block 0.
  aie::block_vector<bfp16ebs8, 64> c0 = cin.pop();
  aie::block_vector<bfp16ebs8, 64> c1 = cin.pop();
  aie::block_vector<bfp16ebs8, 64> c2 = cin.pop();
  aie::block_vector<bfp16ebs8, 64> c3 = cin.pop();

  int br = 0, bc = 0;
  // aie-gpr-realloc: II 79 -> 77. pipeline(disable) hands the loop directly
  // to the post-RA pipeliner: II 89 -> 77.
  AIE_LOOP_GPR_REALLOC
  AIE_PREPARE_FOR_POSTPIPELINING
  for (int blk = 0; blk < nblk; blk++) {
    // Fresh A/B streams per block at computed bases (no seek fifo ops).
    AStream a_stream(pA + br * (2 * k / 8) * 72);
    BStream b_stream(pB + bc * (2 * k / 8) * 72);

    MMUL acc0((aie::accum<accfloat, 64>(c0)));
    MMUL acc1((aie::accum<accfloat, 64>(c1)));
    MMUL acc2((aie::accum<accfloat, 64>(c2)));
    MMUL acc3((aie::accum<accfloat, 64>(c3)));

    // One k-step: 2 A pops + 2 B pops + 4 8x8x8 MACs. On fill steps (B pops
    // 0 and 8 of each 16-pop buffer) B pops come first so the fifo_ld_fill
    // emits ahead of the A-stream ops.
    auto kstep = [&]<bool BFILL>() __attribute__((always_inline)) {
      aie::block_vector<bfp16ebs8, 64> A0, A1, B0, B1;
      if constexpr (BFILL) {
        B0 = b_stream.pop();
        B1 = b_stream.pop();
        A0 = a_stream.pop();
        A1 = a_stream.pop();
      } else {
        A0 = a_stream.pop();
        A1 = a_stream.pop();
        B0 = b_stream.pop();
        B1 = b_stream.pop();
      }
      acc0.mac(A0, aie::op_transpose(B0));
      acc1.mac(A0, aie::op_transpose(B1));
      acc2.mac(A1, aie::op_transpose(B0));
      acc3.mac(A1, aie::op_transpose(B1));
    };
    kstep.operator()<true>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<true>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();
    kstep.operator()<false>();

    // Prefetch next block's C (read region is never written by this call).
    c0 = cin.pop();
    c1 = cin.pop();
    c2 = cin.pop();
    c3 = cin.pop();

    cout.push(acc0.template to_vector<bfp16ebs8>());
    cout.push(acc1.template to_vector<bfp16ebs8>());
    cout.push(acc2.template to_vector<bfp16ebs8>());
    cout.push(acc3.template to_vector<bfp16ebs8>());

    bc = (bc == nbc - 1) ? 0 : bc + 1;
    br += (bc == 0) ? 1 : 0;
  }

  event1();
}
#endif

#ifdef ZERO_ONLY
void zero_kernel(bfp16ebs8 *__restrict cOut) {
  const aie::accum<accfloat, 64> acc = aie::zeros<accfloat, 64>();
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c>
      out_stream(cOut);
  for (int i = 0; i < m * n / 64; i++) {
    out_stream.push(acc.template to_vector<bfp16ebs8>());
  }
}
#endif
}
