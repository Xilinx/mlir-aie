//===- mm_bfp_mixed.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ATB microkernel; C is accumulated in bf16: the kernel rotates over 4
// stripes (stripes 0/1 alias the first 2*m*n bf16 of the bfp16 C tile,
// stripes 2/3 live in the TU-local staging buffer) and converts both halves
// to bfp16ebs8 in place after (K_Problemsize/k)*DIV calls.
//
// Compute structure (config1-style): per call, 2 block rows x 8 blocks of
// 16x16; each block runs 8 expanded k-steps (2 A pops + 2 B pops + 4 8x8x8
// MACs, B-first at fill steps 0 and 4) in a post-RA-pipelined blk loop with
// next-block C prefetch. Measured II=41, NS=2.

#include <aie_api/aie.hpp>

#ifndef DIM_RHO
#define DIM_RHO 4
#endif

// The bf16 -> bfp16 flush happens once per output tile, after exactly
// (K_Problemsize / k) * DIV matmul calls; this kernel only supports
// K = K_Problemsize.
constexpr int K_Problemsize = 4096;
constexpr int rho = DIM_RHO;
constexpr int m = 128 / rho;
constexpr int k = 64;
constexpr int n = 128;

static_assert(m * rho == 128 && k == 64 && n == 128 && rho == 4);

// bf16 staging for C stripes 2 and 3. matmul and zero share this buffer, so
// both ExternalFunctions must point at the same .o (see n32_core.py).
alignas(aie::vector_decl_align) bfloat16 c_bf16_2nd_half[m * rho * n / 2];

extern "C" {

static int g_counter = 0;
static int k_counter = 0;

void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA_in,
                             bfp16ebs8 *__restrict pB_in,
                             bfp16ebs8 *__restrict pC_bfp16) {
  event0();
  // Round-to-nearest-even (Peano defaults to floor on converts).
  aie::set_rounding(aie::rounding_mode::conv_even);

  using AStream = aie::block_vector_restrict_input_buffer_stream<
      bfp16ebs8, 64, aie_dm_resource::a>;
  using BStream = aie::block_vector_restrict_input_buffer_stream<
      bfp16ebs8, 64, aie_dm_resource::b>;
  using CPtr = bfloat16 __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<8, 8, 8, bfp16ebs8>;

  const bfp16ebs8 *pA_base = (const bfp16ebs8 *)pA_in;

  bfloat16 *const pC_sel = (g_counter < 2)
                               ? (bfloat16 *)pC_bfp16 + g_counter * (m * n)
                               : c_bf16_2nd_half + (g_counter - 2) * (m * n);
  g_counter = (g_counter == rho - 1) ? 0 : g_counter + 1;
  CPtr const pC_base = (CPtr)pC_sel;

  for (int br = 0; br < m / 16; br++) {
    // Persistent sliding B stream per block row (16 pops per block).
    BStream b_stream(pB_in);
    {
      CPtr pC = pC_base + br * (n / 16) * 256;

      aie::vector<bfloat16, 64> c0 = aie::load_v<64>(pC);
      aie::vector<bfloat16, 64> c1 = aie::load_v<64>(pC + 64);
      for (int blk = 0; blk < n / 16; blk++) {
        // C blocks are stored block-linearly (256 bf16 per 16x16 block).
        CPtr pCb = pC + blk * 256;
        CPtr pNext = (blk < n / 16 - 1) ? pCb + 256 : pCb;

        MMUL acc0((aie::accum<accfloat, 64>(c0)));
        MMUL acc1((aie::accum<accfloat, 64>(c1)));
        MMUL acc2((aie::accum<accfloat, 64>(aie::load_v<64>(pCb + 128))));
        MMUL acc3((aie::accum<accfloat, 64>(aie::load_v<64>(pCb + 192))));

        // Prefetch next block's first C pair while this block computes.
        c0 = aie::load_v<64>(pNext);
        c1 = aie::load_v<64>(pNext + 64);

        // Fresh A stream per block: re-reads this block row's 16-block band
        // (fills constant-fold before pops 0 and 8).
        AStream a_stream(pA_base);
        a_stream.seek(br * (k / 4));

        // One k-step: 2 A pops + 2 B pops + 4 8x8x8 MACs. On fill steps
        // (B pops 0 and 8 of each 16-pop buffer) B pops come first so the
        // fifo_ld_fill emits ahead of the A-stream ops.
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
        aie::store_v(pCb + 128, acc2.template to_vector<bfloat16>());
        aie::store_v(pCb + 192, acc3.template to_vector<bfloat16>());
      }
    }
  }

  // Full accumulation done: convert both bf16 halves to bfp16ebs8 blocks in
  // place (reads stay ahead of writes: 128B/iter read vs 72B/iter written).
  if (k_counter == (K_Problemsize / k) * rho - 1) {
    k_counter = 0;
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64, aie_dm_resource::c>
        out_stream(pC_bfp16);

    CPtr pC_bf16 = (CPtr)pC_bfp16;
    for (int i = 0; i < m * rho * n / 2 / 64; i++) {
      aie::accum<accfloat, 64> acc_data(aie::load_v<64>(pC_bf16 + i * 64));
      out_stream.push(acc_data.template to_vector<bfp16ebs8>());
    }
    CPtr pC_2nd = (CPtr)c_bf16_2nd_half;
    for (int i = 0; i < m * rho * n / 2 / 64; i++) {
      aie::accum<accfloat, 64> acc_data(aie::load_v<64>(pC_2nd + i * 64));
      out_stream.push(acc_data.template to_vector<bfp16ebs8>());
    }
  } else {
    k_counter += 1;
  }
  event1();
}

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  const aie::vector<bfloat16, 64> zeros = aie::zeros<bfloat16, 64>();
  bfloat16 __aie_dm_resource_c *__restrict p0 =
      (bfloat16 __aie_dm_resource_c *)cOut;
  for (int i = 0; i < m * rho * n / 64 / 2; i++)
    aie::store_v(p0 + i * 64, zeros);
  bfloat16 *__restrict p1 = c_bf16_2nd_half;
  for (int i = 0; i < m * rho * n / 64 / 2; i++)
    aie::store_v(p1 + i * 64, zeros);
}
}
