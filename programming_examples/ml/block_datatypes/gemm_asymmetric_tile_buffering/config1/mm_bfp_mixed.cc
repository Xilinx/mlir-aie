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

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8);
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}
constexpr int M = 128 / 4;
constexpr int K = 64;
constexpr int N = 128;
constexpr int m = 128 / 4;
constexpr int k = 64;
constexpr int n = 128;
constexpr int r = 8;
constexpr int s = 8;
constexpr int t = 8;

extern "C" {

static int g_counter = 0;

void matmul_vectorized_different_datatypes(bfloat16 *__restrict pA,
                                           bfp16ebs8 *__restrict pB,
                                           bfloat16 *__restrict pC_curtile) {

  event0();
  pC_curtile += g_counter * m * n;
  if (g_counter == 3) {
    g_counter = 0;
  } else {
    g_counter = g_counter + 1;
  }
  // convert pA from bfloat16 to bfp16ebs8
  alignas(aie::vector_decl_align) bfp16ebs8 converted_A[M * K / 8];
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pA_bfp16_stream(
      converted_A);
  pA_bfp16_stream.seek(0);
  aie::vector<bfloat16, 64> A_temp;
  for (int row = 0; row < M / 8 / 2; row++) {
    for (int col = 0; col < K / 8; col++) {
      for (int innerid = 0; innerid < 2; innerid++) {
        A_temp = aie::load_v<64>(pA + (row * 2 + innerid) * 8 * 64 + col * 64);
        aie::accum<accfloat, 64> A0_data_float;
        A0_data_float = A_temp;
        pA_bfp16_stream.push(A0_data_float.template to_vector<bfp16ebs8>());
      }
    }
  }

  for (int i = 0; i < m * n / (16 * 16); i = i + 4) {
    int block_row = i / (n / 16);
    int block_col = i % (n / 16);

    bfloat16 *pC = pC_curtile + block_row * N * 16 +
                   block_col * 2 *
                       64; // 64 elements per 8x8 innermost block, 2x2 per block

    int A_stream_index = block_row * 2 * k / 8;
    int B_stream_index = block_col * (k / 8) * 2;

    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB_stream(pB);
    pB_stream.seek(B_stream_index);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream(converted_A);
    pA_stream.seek(A_stream_index);

    aie::accum<accfloat, 64> chess_storage(dm1) acc0_data(aie::load_v<64>(pC));
    aie::accum<accfloat, 64> chess_storage(dm2)
        acc1_data(aie::load_v<64>(pC + 64));
    aie::accum<accfloat, 64> chess_storage(dm3)
        acc2_data(aie::load_v<64>(pC + 0 + N * 8));
    aie::accum<accfloat, 64> chess_storage(dm4)
        acc3_data(aie::load_v<64>(pC + 64 + N * 8));

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

    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream2(
        converted_A);
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
    acc0_data = aie::load_v<64>(pC + 128);
    aie::store_v(pC + 64, acc1_data.template to_vector<bfloat16>());
    acc1_data = aie::load_v<64>(pC + 192);
    aie::store_v(pC + 0 + N * 8, acc2_data.template to_vector<bfloat16>());
    acc2_data = aie::load_v<64>(pC + 128 + N * 8);
    aie::store_v(pC + 64 + N * 8, acc3_data.template to_vector<bfloat16>());
    acc3_data = aie::load_v<64>(pC + 192 + N * 8);

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

    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream3(
        converted_A);
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

    aie::store_v(pC + 128, acc0_data.template to_vector<bfloat16>());
    acc0_data = aie::load_v<64>(pC + 256);
    aie::store_v(pC + 192, acc1_data.template to_vector<bfloat16>());
    acc1_data = aie::load_v<64>(pC + 320);
    aie::store_v(pC + 128 + N * 8, acc2_data.template to_vector<bfloat16>());
    acc2_data = aie::load_v<64>(pC + 256 + N * 8);
    aie::store_v(pC + 192 + N * 8, acc3_data.template to_vector<bfloat16>());
    acc3_data = aie::load_v<64>(pC + 320 + N * 8);

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

    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA_stream4(
        converted_A);
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

    aie::store_v(pC + 256, acc0_data.template to_vector<bfloat16>());
    acc0_data = aie::load_v<64>(pC + 384);
    aie::store_v(pC + 320, acc1_data.template to_vector<bfloat16>());
    acc1_data = aie::load_v<64>(pC + 448);
    aie::store_v(pC + 256 + N * 8, acc2_data.template to_vector<bfloat16>());
    acc2_data = aie::load_v<64>(pC + 384 + N * 8);
    aie::store_v(pC + 320 + N * 8, acc3_data.template to_vector<bfloat16>());
    acc3_data = aie::load_v<64>(pC + 448 + N * 8);

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

    aie::store_v(pC + 384, acc0_data.template to_vector<bfloat16>());
    aie::store_v(pC + 448, acc1_data.template to_vector<bfloat16>());
    aie::store_v(pC + 384 + N * 8, acc2_data.template to_vector<bfloat16>());
    aie::store_v(pC + 448 + N * 8, acc3_data.template to_vector<bfloat16>());
  }

  event1();
}

void zero_kernel_bf16(bfloat16 *__restrict cOut) {
  zero_vectorized<bfloat16, DIM_M, DIM_N>(cOut);
}
}
