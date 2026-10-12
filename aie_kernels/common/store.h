//===- store.h --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_STORE_H
#define AIE_KERNELS_COMMON_STORE_H

#include <aie_api/aie.hpp>
#include <stdint.h>

// Stores v at p as aie::store_unaligned_v does, by rewriting the two 32-byte
// words from the one holding p[0]. That rewrites the word after v when v ends
// on a word boundary, which may belong to another buffer, so where `end` says v
// ends its buffer this rewrites the two words ending with v's last byte
// instead. Those start before the buffer only if v ends in its first word.
template <typename T, unsigned N>
inline void store_unaligned_bounded(T *p, aie::vector<T, N> v, bool end) {
  constexpr int bytes = sizeof(T) * N;
  static_assert(bytes == 16 || bytes == 32);
  const int off = end ? bytes - 33 : 0;
  // Left unrounded, as in aie_api: the hardware rounds a 256-bit access down
  // to 32 bytes.
  v32int8 *q = (v32int8 *)((int8_t *)p + off);
  const unsigned at = (((uintptr_t)p + off) & 31) - off;
  const v64int8 x = ::shift_bytes(
      ::undef_v64int8(), ::set_v64int8(0, v.template cast_to<int8>()), 64 - at);
  v64int8 y = ::set_v64int8(0, q[0]);
  y = ::insert(y, 1, q[1]);
  y = ::sel(y, x, ((1ull << bytes) - 1) << at);
  q[0] = ::extract_v32int8(y, 0);
  q[1] = ::extract_v32int8(y, 1);
}

#endif
