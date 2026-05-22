//===- mm_bfp_mixed.cc ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// The kernel's `c_bf16_2nd_half` BF16 staging buffer is flushed back to the
// BFP-encoded C output once per output tile, after exactly
// (K_Problemsize / k) * DIV matmul calls. The flush count is fixed at compile
// time, so this kernel only supports K = K_Problemsize at runtime. Both this
// example's Makefile default and the lit test pin K to this value; running
// at a different K via CLI override is not supported by this kernel.
constexpr int K_Problemsize = 4096;
constexpr int DIV = 4;
constexpr int M = 128 / 4;
constexpr int K = 64;
constexpr int N = 128;
constexpr int m = 128 / 4;
constexpr int k = 64;
constexpr int n = 128;
constexpr int r = 8;
constexpr int s = 8;
constexpr int t = 8;

alignas(aie::vector_decl_align) bfloat16 c_bf16_2nd_half[m * DIV * n / 2];

template <int Mz, int Nz>
void zero_vectorized_v64bfp16ebs8(bfp16ebs8 *__restrict cOut) {

  bfloat16 *c_ptr_bf16 = (bfloat16 *)cOut;
  const aie::vector<bfloat16, 64> zeros = aie::zeros<bfloat16, 64>();
  for (int i = 0; i < m * DIV * n / 64 / 2; i++) {
    aie::store_v(c_ptr_bf16 + i * 64, zeros);
  }

  bfloat16 *c_bf16_2nd_half_ptr = c_bf16_2nd_half;
  for (int i = 0; i < m * DIV * n / 64 / 2; i++) {
    aie::store_v(c_bf16_2nd_half_ptr + i * 64, zeros);
  }
}

extern "C" {

static int g_counter = 0;

static int k_counter = 0;

void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB,
                             bfp16ebs8 *__restrict pC_bfp16) {

  bfloat16 *pC;
  if (g_counter == 0) {
    pC = (bfloat16 *)pC_bfp16;
  } else if (g_counter == 1) {
    pC = (bfloat16 *)pC_bfp16;
    pC += 1 * m * n;
  } else if (g_counter == 2) {
    pC = c_bf16_2nd_half;
  } else {
    pC = c_bf16_2nd_half + 1 * m * n;
  }

  if (g_counter == DIV - 1) {
    g_counter = 0;
  } else {
    g_counter = g_counter + 1;
  }
  int run_num = 1;
  for (int run = 0; run < run_num; run++) {
    for (int i = 0; i < m * n / (16 * 16); i = i + 4) {
      int block_row = i / (n / 16);
      int block_col = i % (n / 16);

      int A_stream_index = block_row * 2 * k / 8;
      int B_stream_index = block_col * (k / 8) * 2;

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB_stream(pB);
      pB_stream.seek(B_stream_index);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream(pA);
      pA_stream.seek(A_stream_index);

      aie::accum<accfloat, 64> chess_storage(dm1)
          acc0_data(aie::load_v<64>(pC));
      aie::accum<accfloat, 64> chess_storage(dm2)
          acc1_data(aie::load_v<64>(pC + 64));
      aie::accum<accfloat, 64> chess_storage(dm3)
          acc2_data(aie::load_v<64>(pC + 128));
      aie::accum<accfloat, 64> chess_storage(dm4)
          acc3_data(aie::load_v<64>(pC + 192));

      aie::block_vector<bfp16ebs8, 64> chess_storage(ex0) A0_data_bfp;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex1) A1_data_bfp;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex2) B0_data_bfp;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex3) B1_data_bfp;

      aie::block_vector<bfp16ebs8, 64> chess_storage(ex4) A0_data_bfp_pong;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex6) A1_data_bfp_pong;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex5) B0_data_bfp_pong;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex7) B1_data_bfp_pong;

      aie::block_vector<bfp16ebs8, 64> chess_storage(ex8) A0_data_bfp_2;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex9) A1_data_bfp_2;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex10) B0_data_bfp_2;
      aie::block_vector<bfp16ebs8, 64> chess_storage(ex11) B1_data_bfp_2;

      A0_data_bfp = pA_stream.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream.pop();
      B1_data_bfp = pB_stream.pop();

      A0_data_bfp_pong = pA_stream.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream2(pA);
      pA_stream2.seek(A_stream_index);

      A0_data_bfp_2 = pA_stream2.pop();
      B0_data_bfp_2 = pB_stream.pop();
      A1_data_bfp_2 = pA_stream2.pop();
      B1_data_bfp_2 = pB_stream.pop();

      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream2.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream2.pop();
      B1_data_bfp_pong = pB_stream.pop();

      aie::store_v(pC, acc0_data.template to_vector<bfloat16>());
      acc0_data = aie::load_v<64>(pC + 256);
      aie::store_v(pC + 64, acc1_data.template to_vector<bfloat16>());
      acc1_data = aie::load_v<64>(pC + 320);
      aie::store_v(pC + 128, acc2_data.template to_vector<bfloat16>());
      acc2_data = aie::load_v<64>(pC + 384);
      aie::store_v(pC + 192, acc3_data.template to_vector<bfloat16>());
      acc3_data = aie::load_v<64>(pC + 448);

      acc0_data = mac_8x8_8x8T(A0_data_bfp_2, B0_data_bfp_2, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_2, B1_data_bfp_2, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_2, B0_data_bfp_2, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_2, B1_data_bfp_2, acc3_data);
      A0_data_bfp_2 = pA_stream2.pop();
      B0_data_bfp_2 = pB_stream.pop();
      A1_data_bfp_2 = pA_stream2.pop();
      B1_data_bfp_2 = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream2.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream2.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_2, B0_data_bfp_2, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_2, B1_data_bfp_2, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_2, B0_data_bfp_2, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_2, B1_data_bfp_2, acc3_data);
      A0_data_bfp_2 = pA_stream2.pop();
      B0_data_bfp_2 = pB_stream.pop();
      A1_data_bfp_2 = pA_stream2.pop();
      B1_data_bfp_2 = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream2.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream2.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_2, B0_data_bfp_2, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_2, B1_data_bfp_2, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_2, B0_data_bfp_2, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_2, B1_data_bfp_2, acc3_data);
      A0_data_bfp_2 = pA_stream2.pop();
      B0_data_bfp_2 = pB_stream.pop();
      A1_data_bfp_2 = pA_stream2.pop();
      B1_data_bfp_2 = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream2.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream2.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_2, B0_data_bfp_2, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_2, B1_data_bfp_2, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_2, B0_data_bfp_2, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_2, B1_data_bfp_2, acc3_data);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream3(pA);
      pA_stream3.seek(A_stream_index);

      A0_data_bfp = pA_stream3.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream3.pop();
      B1_data_bfp = pB_stream.pop();

      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream3.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream3.pop();
      B1_data_bfp_pong = pB_stream.pop();

      aie::store_v(pC + 256, acc0_data.template to_vector<bfloat16>());
      acc0_data = aie::load_v<64>(pC + 512);
      aie::store_v(pC + 320, acc1_data.template to_vector<bfloat16>());
      acc1_data = aie::load_v<64>(pC + 576);
      aie::store_v(pC + 384, acc2_data.template to_vector<bfloat16>());
      acc2_data = aie::load_v<64>(pC + 640);
      aie::store_v(pC + 448, acc3_data.template to_vector<bfloat16>());
      acc3_data = aie::load_v<64>(pC + 704);

      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream3.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream3.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream3.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream3.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream3.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream3.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream3.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream3.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream3.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream3.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream3.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream3.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream4(pA);
      pA_stream4.seek(A_stream_index);

      A0_data_bfp = pA_stream4.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream4.pop();
      B1_data_bfp = pB_stream.pop();

      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream4.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream4.pop();
      B1_data_bfp_pong = pB_stream.pop();

      aie::store_v(pC + 512, acc0_data.template to_vector<bfloat16>());
      acc0_data = aie::load_v<64>(pC + 768);
      aie::store_v(pC + 576, acc1_data.template to_vector<bfloat16>());
      acc1_data = aie::load_v<64>(pC + 832);
      aie::store_v(pC + 640, acc2_data.template to_vector<bfloat16>());
      acc2_data = aie::load_v<64>(pC + 896);
      aie::store_v(pC + 704, acc3_data.template to_vector<bfloat16>());
      acc3_data = aie::load_v<64>(pC + 960);

      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream4.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream4.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream4.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream4.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream4.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream4.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream4.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream4.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);
      A0_data_bfp = pA_stream4.pop();
      B0_data_bfp = pB_stream.pop();
      A1_data_bfp = pA_stream4.pop();
      B1_data_bfp = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      A0_data_bfp_pong = pA_stream4.pop();
      B0_data_bfp_pong = pB_stream.pop();
      A1_data_bfp_pong = pA_stream4.pop();
      B1_data_bfp_pong = pB_stream.pop();
      acc0_data = mac_8x8_8x8T(A0_data_bfp, B0_data_bfp, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp, B1_data_bfp, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp, B0_data_bfp, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp, B1_data_bfp, acc3_data);

      acc0_data = mac_8x8_8x8T(A0_data_bfp_pong, B0_data_bfp_pong, acc0_data);
      acc1_data = mac_8x8_8x8T(A0_data_bfp_pong, B1_data_bfp_pong, acc1_data);
      acc2_data = mac_8x8_8x8T(A1_data_bfp_pong, B0_data_bfp_pong, acc2_data);
      acc3_data = mac_8x8_8x8T(A1_data_bfp_pong, B1_data_bfp_pong, acc3_data);

      aie::store_v(pC + 768, acc0_data.template to_vector<bfloat16>());
      aie::store_v(pC + 832, acc1_data.template to_vector<bfloat16>());
      aie::store_v(pC + 896, acc2_data.template to_vector<bfloat16>());
      aie::store_v(pC + 960, acc3_data.template to_vector<bfloat16>());

      pC += 256 * 4;
    }
  }

  if (k_counter == (K_Problemsize / k) * DIV - 1) { // // K / k  *  DIV
    k_counter = 0;
    // inplace convert the first half BF16 to pC_bfp16
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pCOut_stream_first(
        pC_bfp16);
    pCOut_stream_first.seek(0);

    bfloat16 *pC_bf16;
    pC_bf16 = (bfloat16 *)pC_bfp16;
    for (int i = 0; i < m * DIV * n / 2 / 64; i++) {
      aie::accum<accfloat, 64> acc_data(aie::load_v<64>(pC_bf16));
      pC_bf16 += 64;
      pCOut_stream_first.push(acc_data.template to_vector<bfp16ebs8>());
    }

    bfloat16 *c_bf16_2nd_half_ptr = c_bf16_2nd_half;
    for (int i = 0; i < m * DIV * n / 2 / 64; i++) {
      aie::accum<accfloat, 64> acc_data(aie::load_v<64>(c_bf16_2nd_half_ptr));
      c_bf16_2nd_half_ptr += 64;
      pCOut_stream_first.push(acc_data.template to_vector<bfp16ebs8>());
    }
  } else {
    k_counter += 1;
  }
}

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  zero_vectorized_v64bfp16ebs8<DIM_M, DIM_N>(cOut);
}
}