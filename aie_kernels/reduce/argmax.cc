//===- argmax.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifdef ARGMAX_ELEMS
// The per-lane offsets are int16.
static_assert(ARGMAX_ELEMS > 0 && ARGMAX_ELEMS <= INT16_MAX);
#else
#define ARGMAX_ELEMS input_size
#endif

// A record is two int32s, so one objectFIFO carries value and index together:
//   out[0]  the value: int32 as is, bfloat16 widened to float and bit-cast
//   out[1]  its index plus the caller's index_offset, so indices are global
//           and argmax_combine's merge order does not matter
// Ties go to the lowest index, as in numpy.argmax. A NaN never compares
// greater, so it reads as -inf; numpy.argmax returns the first NaN instead.
// A -0 reads as +0: the bf16 compare orders -0 below +0, but its equality
// does not, so canonicalizing is what keeps -0 and +0 a tie.

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

// -inf rather than lowest(), so a tile of -inf (or of lowest()) still
// resolves to its first element.
template <typename T>
static inline T _argmax_seed() {
  if constexpr (std::numeric_limits<T>::has_infinity)
    return -std::numeric_limits<T>::infinity();
  else
    return std::numeric_limits<T>::lowest();
}

template <typename T>
static inline T _argmax_load(const T *in) {
  T value = in[0];
  if constexpr (std::is_same_v<T, bfloat16>) {
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if (bits == 0x8000)
      value = T(0);
  }
  return value;
}

template <typename T, typename V>
static inline V _argmax_load_v(const T *in) {
  V value = aie::load_v<V::size()>(in);
  if constexpr (std::is_same_v<T, bfloat16>) {
    const auto bits = value.template cast_to<int16_t>();
    // -0 is INT16_MIN. lt, because eq against it crashes Peano's legalizer.
    value = aie::select(bits, aie::zeros<int16_t, V::size()>(),
                        aie::lt(bits, (int16_t)(INT16_MIN + 1)))
                .template cast_to<bfloat16>();
  }
  return value;
}

// One streaming pass. Lane j only ever sees positions j, j+N, j+2N, ..., and a
// strict `>` keeps the earliest of equal values within a lane, so the global
// first-occurrence index is min(offset[j] + j) over the lanes still holding the
// maximum -- resolved once, after the loop, not per step.
template <typename T, typename V>
void _argmax_vector(T *restrict in, int32_t *restrict out,
                    const int32_t input_size, const int32_t index_offset) {
  event0();
  constexpr int32_t N = V::size();
  using Idx = aie::vector<int16_t, N>;

  alignas(64) int16_t lane_init[N];
  for (int32_t k = 0; k < N; k++)
    lane_init[k] = (int16_t)k;
  const Idx lane = aie::load_v<N>(lane_init);

  V running_max = aie::broadcast<T, N>(_argmax_seed<T>());
  Idx running_off = aie::zeros<int16_t, N>();
  Idx offset = aie::zeros<int16_t, N>();
  const Idx step = aie::broadcast<int16_t, N>((int16_t)N);

  int32_t i = 0;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_UNROLL(4)
  for (; i + N <= ARGMAX_ELEMS; i += N) {
    V next = _argmax_load_v<T, V>(in + i);
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

  for (; i < ARGMAX_ELEMS; i++) { // the tile need not be a multiple of N
    const T value = _argmax_load(in + i);
    if (value > best) {
      best = value;
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
  T best = _argmax_seed<T>();
  int32_t best_index = 0;
  for (int32_t i = 0; i < ARGMAX_ELEMS; i++) {
    const T value = _argmax_load(in + i);
    if (value > best) { // strict >, so the first of equal values wins
      best = value;
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
