// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

// This kernel uses the higher-level AIE API to perform a transpsoe.
// This higher-level API uses VSHUFFLE intrinsics internally.
// See programming_examples/basic/shuffle_transpsoe for a transpose
// example at that level.

#include <aie_api/aie.hpp>
#include <cstdint>
#include <cassert>

#if !defined(DIM_m) || !defined(DIM_n) || !defined(DIM_s)
#error Please specify matrix sizes m, n, s at kernel compile time using e.g., -DDIM_m=32 -DDIM_n=32 -DDIM_s=8.
#endif

#define DTYPE uint8_t

constexpr size_t VECTOR_SIZE = (DIM_m * sizeof(DTYPE) < 512 ? DIM_m * sizeof(DTYPE) : 512);
constexpr size_t DIM_r = DIM_m;

static_assert(DIM_m % DIM_r == 0);
static_assert(DIM_n % DIM_s == 0);
static_assert(DIM_m * sizeof(DTYPE) <= 512);
static_assert(DIM_r == DIM_m);

constexpr size_t OUTER_SIZE = DIM_m * DIM_n;


extern "C" {

void copy(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  constexpr size_t INNER_SIZE = 256;
  DTYPE * const in_ptr_end = in_ptr + OUTER_SIZE;
  for(; in_ptr < in_ptr_end; in_ptr += INNER_SIZE, out_ptr += INNER_SIZE) {
    aie::vector<DTYPE, INNER_SIZE> data = aie::load_v<INNER_SIZE>(in_ptr);
    aie::store_v(out_ptr, data);
  }
}

void transpose(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  constexpr size_t VECTOR_SIZE = 16;
  DTYPE * const in_ptr_end = in_ptr + OUTER_SIZE;
  while(in_ptr < in_ptr_end) {
    aie::vector<DTYPE, VECTOR_SIZE> row_0 = aie::load_v<VECTOR_SIZE>(in_ptr);
    in_ptr += VECTOR_SIZE;
    aie::vector<DTYPE, VECTOR_SIZE> row_1 = aie::load_v<VECTOR_SIZE>(in_ptr);
    in_ptr += VECTOR_SIZE;
    aie::vector<DTYPE, VECTOR_SIZE> row_2 = aie::load_v<VECTOR_SIZE>(in_ptr);
    in_ptr += VECTOR_SIZE;
    aie::vector<DTYPE, VECTOR_SIZE> row_3 = aie::load_v<VECTOR_SIZE>(in_ptr);
    in_ptr += VECTOR_SIZE;

    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> zip_2_0 =
      aie::interleave_zip(row_0, row_2, 2);
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> zip_2_1 =
      aie::interleave_zip(row_1, row_3, 2);
    auto [zip_2_00, zip_2_01] = zip_2_0;
    auto [zip_2_10, zip_2_11] = zip_2_1;

    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> zip_1_0 =
      aie::interleave_zip(zip_2_00, zip_2_10, 1);
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> zip_1_1 =
      aie::interleave_zip(zip_2_01, zip_2_11, 1);
    auto [zip_1_00, zip_1_01] = zip_1_0;
    auto [zip_1_10, zip_1_11] = zip_1_1;

    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> out_0 =
      aie::interleave_unzip(zip_1_00, zip_1_01, 2);
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> out_1 =
      aie::interleave_unzip(zip_1_10, zip_1_11, 2);
    
    auto [out_0_0, out_0_1] = out_0;
    auto [out_1_0, out_1_1] = out_1;

    aie::store_v(out_ptr, out_0_0);
    out_ptr += VECTOR_SIZE;
    aie::store_v(out_ptr, out_0_1);
    out_ptr += VECTOR_SIZE;
    aie::store_v(out_ptr, out_1_0);
    out_ptr += VECTOR_SIZE;
    aie::store_v(out_ptr, out_1_1);
    out_ptr += VECTOR_SIZE;
  }
}

}
