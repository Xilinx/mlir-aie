//===- reduce_identity.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_REDUCE_REDUCE_IDENTITY_H
#define AIE_KERNELS_REDUCE_REDUCE_IDENTITY_H

#include <aie_api/aie.hpp>
#include <limits>

// The seed of a running max: -inf rather than lowest(), so a tile of -inf
// reduces to -inf and argmax still takes its first element.
template <typename T>
static inline T max_identity() {
  if constexpr (std::numeric_limits<T>::has_infinity)
    return -std::numeric_limits<T>::infinity();
  else
    return std::numeric_limits<T>::lowest();
}

#endif
