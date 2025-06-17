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

#if !defined(DIM_m) || !defined(DIM_n) 
#error Please specify matrix sizes m, n at kernel compile time using e.g., -DDIM_m=32 -DDIM_n=32.
#endif

#if !defined(DTYPE_i8) && !defined(DTYPE_i16) && !defined(DTYPE_i32) 
#error Please specify data type at kernel compile time using e.g., -DDTYPE_i8 or -DDTYPE_i16 or -DDTYPE_i32.
#endif

#if defined(DTYPE_i8)
#define DTYPE uint8_t
#endif
#if defined(DTYPE_i16)
#define DTYPE uint16_t
#endif
#if defined(DTYPE_i32)
#define DTYPE uint32_t
#endif

constexpr size_t OUTER_SIZE = DIM_m * DIM_n;
constexpr size_t VECTOR_SIZE = (DIM_m * sizeof(DTYPE) < 512 ? DIM_m * sizeof(DTYPE) : 512) / sizeof(DTYPE);

static_assert(OUTER_SIZE % VECTOR_SIZE == 0);

extern "C" {

void copy(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  DTYPE * const in_ptr_end = in_ptr + OUTER_SIZE;
  for(; in_ptr < in_ptr_end; in_ptr += VECTOR_SIZE, out_ptr += VECTOR_SIZE) {
    aie::vector<DTYPE, VECTOR_SIZE> data = aie::load_v<VECTOR_SIZE>(in_ptr);
    aie::store_v(out_ptr, data);
  }
}

/* Individually transposes 4x4-sized subtiles in in_ptr (a matrix of size 
   DIM_n*DIM_m). Note that in principle, this would also work in-place
   (i.e. in_ptr == out_ptr). */
void transpose_4(DTYPE *__restrict__ in_ptr, DTYPE *__restrict__ out_ptr) {
  for(unsigned col = 0; col < DIM_m; col += VECTOR_SIZE) {
    for(unsigned row = 0; row < DIM_n; row += 4) {
      aie::vector<DTYPE, VECTOR_SIZE> row_0 = aie::load_v<VECTOR_SIZE>(in_ptr + col + row * DIM_m);
      aie::vector<DTYPE, VECTOR_SIZE> row_1 = aie::load_v<VECTOR_SIZE>(in_ptr + col + (row + 1) * DIM_m);
      aie::vector<DTYPE, VECTOR_SIZE> row_2 = aie::load_v<VECTOR_SIZE>(in_ptr + col + (row + 2) * DIM_m);
      aie::vector<DTYPE, VECTOR_SIZE> row_3 = aie::load_v<VECTOR_SIZE>(in_ptr + col + (row + 3) * DIM_m);

      // Interleave all rows one element at a time.
      // Result: zipped_0, zipped_1, zipped_2, zipped_3
      // zipped_0 = row_0[0], row_1[0], row_2[0], row_3[0], row_0[1], row_1[1], ...
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

      aie::store_v(out_ptr + col + row * DIM_m, unzipped_0);
      aie::store_v(out_ptr + col + (row + 1) * DIM_m, unzipped_1);
      aie::store_v(out_ptr + col + (row + 2) * DIM_m, unzipped_2);
      aie::store_v(out_ptr + col + (row + 3) * DIM_m, unzipped_3);
    }
  }
}

}
