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

template <typename T, unsigned N, aie_dm_resource R>
using InBufStream = aie::block_vector_restrict_input_buffer_stream<T, N, R>;
template <typename T, unsigned N, aie_dm_resource R>
using OutBufStream = aie::block_vector_output_buffer_stream<T, N, R>;

// The bf16 -> bfp16 flush happens once per output tile, after exactly
// (K_Problemsize / k) * DIV matmul calls; this kernel only supports
// K = K_Problemsize.
constexpr int K_Problemsize = 4096;
constexpr int rho = 4;
constexpr int m = 128 / rho;
constexpr int k = 64;
constexpr int n = 128;

// aie::mmul sub-tile shape (the only shape supported for bfp16ebs8 on
// AIE2P): per MAC, an r x s A panel times an s x t B panel accumulates
// into r x t of C. C blocks are stored block-linearly: each (2r)x(2t)
// block is 4 consecutive r x t panels (offsets 0, rt, 2rt, 3rt).
constexpr int r = 8; // MAC rows (A/C)
constexpr int s = 8; // MAC reduction step (k-columns per MAC)
constexpr int t = 8; // MAC columns (B/C)

// bf16 vector width for the flush/zero passes.
constexpr int vec = 64;

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

  using AStream = InBufStream<bfp16ebs8, r * s, aie_dm_resource::a>;
  using BStream = InBufStream<bfp16ebs8, s * t, aie_dm_resource::b>;
  using CPtr = bfloat16 __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<r, s, t, bfp16ebs8>;

  const bfp16ebs8 *pA_base = (const bfp16ebs8 *)pA_in;

  bfloat16 *const pC_sel = (g_counter < 2)
                               ? (bfloat16 *)pC_bfp16 + g_counter * (m * n)
                               : c_bf16_2nd_half + (g_counter - 2) * (m * n);
  g_counter = (g_counter == rho - 1) ? 0 : g_counter + 1;
  CPtr const pC_base = (CPtr)pC_sel;

  for (int br = 0; br < m / (2 * r); br++) {
    // Persistent sliding B stream per block row (16 pops per block).
    BStream b_stream(pB_in);
    {
      CPtr pC = pC_base + br * (n / (2 * t)) * (4 * r * t);

      aie::vector<bfloat16, r * t> c0 = aie::load_v<r * t>(pC);
      aie::vector<bfloat16, r * t> c1 = aie::load_v<r * t>(pC + r * t);
      for (int blk = 0; blk < n / (2 * t); blk++) {
        // C blocks are stored block-linearly (4*r*t bf16 per 2rx2t block).
        CPtr pCb = pC + blk * (4 * r * t);
        CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 4 * r * t : pCb;

        MMUL acc0((aie::accum<accfloat, r * t>(c0)));
        MMUL acc1((aie::accum<accfloat, r * t>(c1)));
        MMUL acc2(
            (aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + 2 * r * t))));
        MMUL acc3(
            (aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + 3 * r * t))));

        // Prefetch next block's first C pair while this block computes.
        c0 = aie::load_v<r * t>(pNext);
        c1 = aie::load_v<r * t>(pNext + r * t);

        // Fresh A stream per block: re-reads this block row's 16-block band
        // (fills constant-fold before pops 0 and 8).
        AStream a_stream(pA_base);
        a_stream.seek(br * (2 * k / s));

        // One k-step: 2 A pops + 2 B pops + 4 8x8x8 MACs. On fill steps
        // (B pops 0 and 8 of each 16-pop buffer) B pops come first so the
        // fifo_ld_fill emits ahead of the A-stream ops.
        auto kstep = [&]<bool BFILL>() __attribute__((always_inline)) {
          aie::block_vector<bfp16ebs8, r * s> A0, A1; // r x s A panels
          aie::block_vector<bfp16ebs8, s * t> B0, B1; // s x t B panels
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
        aie::store_v(pCb + r * t, acc1.template to_vector<bfloat16>());
        aie::store_v(pCb + 2 * r * t, acc2.template to_vector<bfloat16>());
        aie::store_v(pCb + 3 * r * t, acc3.template to_vector<bfloat16>());
      }
    }
  }

  // Full accumulation done: convert both bf16 halves to bfp16ebs8 blocks in
  // place (reads stay ahead of writes: 128B/iter read vs 72B/iter written).
  if (k_counter == (K_Problemsize / k) * rho - 1) {
    k_counter = 0;
    OutBufStream<bfp16ebs8, r * t, aie_dm_resource::c> out_stream(pC_bfp16);

    CPtr pC_bf16 = (CPtr)pC_bfp16;
    for (int i = 0; i < m * rho * n / 2 / vec; i++) {
      aie::accum<accfloat, vec> acc_data(aie::load_v<vec>(pC_bf16 + i * vec));
      out_stream.push(acc_data.template to_vector<bfp16ebs8>());
    }
    CPtr pC_2nd = (CPtr)c_bf16_2nd_half;
    for (int i = 0; i < m * rho * n / 2 / vec; i++) {
      aie::accum<accfloat, vec> acc_data(aie::load_v<vec>(pC_2nd + i * vec));
      out_stream.push(acc_data.template to_vector<bfp16ebs8>());
    }
  } else {
    k_counter += 1;
  }
  event1();
}

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  const aie::vector<bfloat16, vec> zeros = aie::zeros<bfloat16, vec>();
  bfloat16 __aie_dm_resource_c *__restrict p0 =
      (bfloat16 __aie_dm_resource_c *)cOut;
  for (int i = 0; i < m * rho * n / vec / 2; i++)
    aie::store_v(p0 + i * vec, zeros);
  bfloat16 *__restrict p1 = c_bf16_2nd_half;
  for (int i = 0; i < m * rho * n / vec / 2; i++)
    aie::store_v(p1 + i * vec, zeros);
}
}
