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
  constexpr size_t INNER_SIZE = 64;
  static_assert(DIM_m * DIM_n % INNER_SIZE == 0);
  DTYPE * const in_ptr_end = in_ptr + OUTER_SIZE;
  for(; in_ptr < in_ptr_end; in_ptr += INNER_SIZE, out_ptr += INNER_SIZE) {
    aie::vector<DTYPE, INNER_SIZE> data = aie::load_v<INNER_SIZE>(in_ptr);
    aie::store_v(out_ptr, data);
  }
}

void transpose(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  constexpr size_t VECTOR_SIZE = 16;
  static_assert(DIM_m * DIM_n % VECTOR_SIZE == 0);
  assert(s == 4);
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

    // Interleave all rows one element at a time.
    // Result: zipped_0, zipped_1, zipped_2, zipped_3
    // row_0[0], row_1[0], row_2[0], row_3[0], row_0[1], row_1[1], ...
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> zip_one_0 =
      aie::interleave_zip(row_0, row_1, 1);
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> zip_one_1 =
      aie::interleave_zip(row_2, row_3, 1);
    auto [zip_one_00, zip_one_01] = zip_one_0;
    auto [zip_one_10, zip_one_11] = zip_one_1;
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> zip_two_0 =
      aie::interleave_zip(zip_one_00, zip_one_10, 2);
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> zip_two_1 =
      aie::interleave_zip(zip_one_01, zip_one_11, 2);
    auto [zipped_0, zipped_1] = zip_two_0;
    auto [zipped_2, zipped_3] = zip_two_1;

    // Extract four elements at a time
    // Result: unzipped_0, unzipped_1, unzipped_2, unzipped_3
    // unzipped_0 = zipped_0[0..4], zipped_0[16..20],  ..., zipped_1[0..4]
    // unzipped_1 = zipped_0[4..8],  zipped_0[20..24], ..., zipped_1[4..8]
    // unzipped_2 = zipped_0[8..12], zipped_0[24..28], ..., zipped_1[8..12]
    // unzipped_3 = zipped_0[12..16], zipped_0[28..32], ..., zipped_1[12..16]
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> unzip_eight_0 =
      aie::interleave_unzip(zipped_0, zipped_1, 8);
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> unzip_eight_1 =
      aie::interleave_unzip(zipped_2, zipped_3, 8);
    auto [unzip_eight_00, unzip_eight_10] = unzip_eight_0;
    auto [unzip_eight_01, unzip_eight_11] = unzip_eight_1;
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> unzip_four_0 =
      aie::interleave_unzip(unzip_eight_00, unzip_eight_01, 4);
    std::pair<aie::vector<DTYPE, VECTOR_SIZE>, aie::vector<DTYPE, VECTOR_SIZE>> unzip_four_1 =
      aie::interleave_unzip(unzip_eight_10, unzip_eight_11, 4);
    auto [unzipped_0, unzipped_1] = unzip_four_0;
    auto [unzipped_2, unzipped_3] = unzip_four_1;

    aie::store_v(out_ptr, unzipped_0);
    out_ptr += VECTOR_SIZE;
    aie::store_v(out_ptr, unzipped_1);
    out_ptr += VECTOR_SIZE;
    aie::store_v(out_ptr, unzipped_2);
    out_ptr += VECTOR_SIZE;
    aie::store_v(out_ptr, unzipped_3);
    out_ptr += VECTOR_SIZE;
  }
}

}
