//===- argmax.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <stdint.h>
#include <string.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// Index of the largest element of a tile, as the (partial, combine) pair that
// aie2/reduce_max.cc already uses for a distributed max: every core runs
// _argmax_* over its own slice and a tree merges the records with
// _argmax_combine.
//
// A record is 8 bytes, written through an int32 output tile so that one
// objectFIFO carries value and index together:
//   out[0]  the winning value -- int32 as itself, bfloat16 widened to float and
//           bit-cast, so the combine step can compare without the input tile
//   out[1]  its index, already global: the caller passes the slice's
//           index_offset, which makes combine order-independent
//
// Ties resolve to the lowest index, matching numpy.argmax. A NaN never compares
// greater, so NaN inputs are skipped rather than returned -- numpy.argmax
// returns the first NaN instead.

template <typename T>
using argmax_value_t =
    std::conditional_t<std::is_same_v<T, int32_t>, int32_t, float>;

template <typename T>
static inline void _argmax_store(int32_t *restrict out, T value,
                                 int32_t index) {
  const argmax_value_t<T> widened = value;
  memcpy(out, &widened, sizeof(widened));
  out[1] = index;
}

// One streaming pass. Lane j only ever sees positions j, j+N, j+2N, ..., and a
// strict `>` keeps the earliest of equal values within a lane, so the global
// first-occurrence index is min(offset[j] + j) over the lanes still holding the
// maximum -- resolved once, after the loop, not per step.
//
// The per-lane offsets are int16. That bounds a call at 32767 elements, which
// no input tile can reach: 32767 bfloat16 is 64 KB, the whole of a core's data
// memory, and the tail loop below takes any remainder.
template <typename T, typename V>
void _argmax_vector(T *restrict in, int32_t *restrict out,
                    const int32_t input_size, const int32_t index_offset) {
  event0();
  constexpr int32_t N = V::size();
  using Idx = aie::vector<int16_t, N>;
  assert(input_size <= INT16_MAX);

  alignas(64) int16_t lane_init[N];
  for (int32_t k = 0; k < N; k++)
    lane_init[k] = (int16_t)k;
  const Idx lane = aie::load_v<N>(lane_init);

  V running_max = aie::broadcast<T, N>(std::numeric_limits<T>::lowest());
  Idx running_off = aie::zeros<int16_t, N>();
  Idx offset = aie::zeros<int16_t, N>();
  const Idx step = aie::broadcast<int16_t, N>((int16_t)N);

  int32_t i = 0;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (; i + N <= input_size; i += N) {
    V next = aie::load_v<N>(in + i);
    auto improved = aie::gt(next, running_max);
    running_max = aie::select(running_max, next, improved);
    running_off = aie::select(running_off, offset, improved);
    offset = aie::add(offset, step);
  }

  T best = aie::reduce_max(running_max);
  const Idx candidates =
      aie::select(aie::broadcast<int16_t, N>(INT16_MAX),
                  aie::add(running_off, lane), aie::eq(running_max, best));
  int32_t best_index = (int32_t)aie::reduce_min(candidates);

  for (; i < input_size;
       i++) { // remainder: input_size need not be a multiple of N
    if (in[i] > best) {
      best = in[i];
      best_index = i;
    }
  }

  _argmax_store<T>(out, best, index_offset + best_index);
  event1();
}

template <typename T>
void _argmax_scalar(T *restrict in, int32_t *restrict out,
                    const int32_t input_size, const int32_t index_offset) {
  event0();
  T best = std::numeric_limits<T>::lowest();
  int32_t best_index = 0;
  for (int32_t i = 0; i < input_size; i++) {
    if (in[i] > best) { // strict >, so the first of equal values wins
      best = in[i];
      best_index = i;
    }
  }
  _argmax_store<T>(out, best, index_offset + best_index);
  event1();
}

template <typename TValue>
void _argmax_combine(int32_t *restrict in1, int32_t *restrict in2,
                     int32_t *restrict out) {
  event0();
  TValue v1, v2;
  memcpy(&v1, in1, sizeof(v1));
  memcpy(&v2, in2, sizeof(v2));
  const bool take2 = (v2 > v1) || (v2 == v1 && in2[1] < in1[1]);
  out[0] = take2 ? in2[0] : in1[0];
  out[1] = take2 ? in2[1] : in1[1];
  event1();
}

extern "C" {

void argmax_vector_bfloat16(bfloat16 *a_in, int32_t *c_out, int32_t input_size,
                            int32_t index_offset) {
  _argmax_vector<bfloat16, aie::vector<bfloat16, 32>>(a_in, c_out, input_size,
                                                      index_offset);
}

void argmax_scalar_bfloat16(bfloat16 *a_in, int32_t *c_out, int32_t input_size,
                            int32_t index_offset) {
  _argmax_scalar<bfloat16>(a_in, c_out, input_size, index_offset);
}

void argmax_combine_bfloat16(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  _argmax_combine<float>(a_in, b_in, c_out);
}

void argmax_vector(int32_t *a_in, int32_t *c_out, int32_t input_size,
                   int32_t index_offset) {
  _argmax_vector<int32_t, aie::vector<int32_t, 16>>(a_in, c_out, input_size,
                                                    index_offset);
}

void argmax_scalar(int32_t *a_in, int32_t *c_out, int32_t input_size,
                   int32_t index_offset) {
  _argmax_scalar<int32_t>(a_in, c_out, input_size, index_offset);
}

void argmax_combine(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  _argmax_combine<int32_t>(a_in, b_in, c_out);
}

} // extern "C"
