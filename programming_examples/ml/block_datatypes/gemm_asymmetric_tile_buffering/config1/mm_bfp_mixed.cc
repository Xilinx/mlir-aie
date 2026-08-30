//===- mm_bfp_mixed.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ATB microkernel, bf16 A / bfp16ebs8 B / bf16 C variant. A is converted to
// bfp16ebs8 in a stack staging buffer first (sizeof(bfp16ebs8) == 1 on Peano,
// so all bfp16ebs8* arithmetic below is byte arithmetic), then the MAC loop
// streams A and B as 576-bit blocks while C is plain bf16 load/store.
// Register placement uses DM-bank annotations: A -> bank A, B -> bank B,
// C -> bank C, raw A input -> bank D.
//
// Compute structure: per call, 2 block rows x 8 blocks of 16x16; each block
// runs 8 expanded k-steps (2 A pops + 2 B pops + 4 8x8x8 MACs, B-first at
// fill steps 0 and 4) in a post-RA-pipelined blk loop with next-block C
// prefetch. Measured II=42, NS=2.

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

#ifndef DIM_RHO
#define DIM_RHO 4
#endif

// Per-core L1 C tile is (m, n); the L1 A sub-tile is (m/rho, k). The kernel
// is invoked rho times per (A-sub-tile, B-tile) pair and rotates over the
// rho C stripes (m/rho rows each) via g_counter.
constexpr int rho = DIM_RHO;
constexpr int m = 128;
constexpr int k = 64;
constexpr int n = 128;
constexpr int m_a = m / rho;

static_assert(m * rho == 512 && k == 64 && n == 128 && rho == 4);

extern "C" {

// MATMUL_ONLY / ZERO_ONLY gates — distinct ExternalFunction .o builds of
// this TU emit exactly one symbol. Without any macro, both are emitted.
#if !defined(MATMUL_ONLY) && !defined(ZERO_ONLY)
#define MATMUL_ONLY
#define ZERO_ONLY
#endif

static int g_counter = 0;

#ifdef MATMUL_ONLY
void matmul_vectorized_different_datatypes(bfloat16 *__restrict pA_in,
                                           bfp16ebs8 *__restrict pB_in,
                                           bfloat16 *__restrict pC_curtile) {
  event0();
  // Round-to-nearest-even (Peano defaults to floor on converts).
  aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *__restrict pC_tile = pC_curtile + g_counter * (m_a * n);
  g_counter = (g_counter == rho - 1) ? 0 : g_counter + 1;

  uint8_t converted_A[m_a * k / 8 * 9] __attribute__((aligned(64)));

  aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::a>
      a_out((bfp16ebs8 *)converted_A);
  const bfloat16 __aie_dm_resource_d *__restrict pA =
      (const bfloat16 __aie_dm_resource_d *)pA_in;

  for (int rp = 0; rp < m_a / 16; rp++) {
    const bfloat16 __aie_dm_resource_d *__restrict q = pA + rp * (16 * k);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int cb = 0; cb < k / 8; cb++) {
      aie::vector<bfloat16, 64> v0 = aie::load_v<64>(q);
      aie::vector<bfloat16, 64> v1 = aie::load_v<64>(q + 8 * k);
      a_out.push(aie::accum<accfloat, 64>(v0).template to_vector<bfp16ebs8>());
      a_out.push(aie::accum<accfloat, 64>(v1).template to_vector<bfp16ebs8>());
      q += 64;
    }
  }

  using AStream = aie::block_vector_restrict_input_buffer_stream<
      bfp16ebs8, 64, aie_dm_resource::a>;
  using BStream = aie::block_vector_restrict_input_buffer_stream<
      bfp16ebs8, 64, aie_dm_resource::b>;
  using CPtr = bfloat16 __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;

  const bfp16ebs8 *pA_base = (const bfp16ebs8 *)converted_A;
  CPtr const pC_base = (CPtr)pC_tile;

  for (int br = 0; br < m_a / 16; br++) {
    // Persistent sliding B stream per block row.
    BStream b_stream(pB_in);
    {
      CPtr pC = pC_base + br * (n * 16);

      aie::vector<bfloat16, 64> c0 = aie::load_v<64>(pC);
      aie::vector<bfloat16, 64> c1 = aie::load_v<64>(pC + 64);
      for (int blk = 0; blk < n / 16; blk++) {
        CPtr pCb = pC + blk * 128;
        CPtr pNext = (blk < n / 16 - 1) ? pCb + 128 : pCb;

        MMUL acc0((aie::accum<accfloat, 64>(c0)));
        MMUL acc1((aie::accum<accfloat, 64>(c1)));
        MMUL acc2((aie::accum<accfloat, 64>(aie::load_v<64>(pCb + n * 8))));
        MMUL acc3((aie::accum<accfloat, 64>(aie::load_v<64>(pCb + 64 + n * 8))));

        // Prefetch next block's first C pair while this block computes.
        c0 = aie::load_v<64>(pNext);
        c1 = aie::load_v<64>(pNext + 64);

        // Fresh A stream per block: re-reads this block row's band (fills
        // constant-fold before pops 0 and 8).
        AStream a_stream(pA_base);
        a_stream.seek(br * (k / 4));

        // One k-step: 2 A pops + 2 B pops + 4 8x8x8 MACs. On fill steps
        // (B pops 0 and 8 of each 16-pop stream buffer) the B pops come
        // first so the fifo_ld_fill emits ahead of the A-stream ops.
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
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();

        aie::store_v(pCb, acc0.template to_vector<bfloat16>());
        aie::store_v(pCb + 64, acc1.template to_vector<bfloat16>());
        aie::store_v(pCb + n * 8, acc2.template to_vector<bfloat16>());
        aie::store_v(pCb + 64 + n * 8, acc3.template to_vector<bfloat16>());
      }
    }
  }

  event1();
}
#endif

#ifdef ZERO_ONLY
void zero_kernel_bf16(bfloat16 *__restrict cOut) {
  const aie::vector<bfloat16, 64> zeros = aie::zeros<bfloat16, 64>();
  bfloat16 __aie_dm_resource_c *__restrict p =
      (bfloat16 __aie_dm_resource_c *)cOut;
  for (int i = 0; i < m * n / 64; i++)
    aie::store_v(p + i * 64, zeros);
}
#endif
}
