//===- AIEVecShuffleModes.h - Semantics of aievec.shuffle modes -*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// What each aievec.shuffle mode does, derived from the mode name.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECSHUFFLEMODES_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECSHUFFLEMODES_H

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

#include <array>
#include <cstdint>

namespace xilinx::aievec {

// The name of a mode, t<elemBits>_<rows>x<cols>[_lo|_hi|_flip], read as a
// matrix of rows x cols elements.
struct ShuffleModeShape {
  unsigned elemBits;
  unsigned rows;
  unsigned cols;
  // Reverses each group of `cols` elements instead of transposing.
  bool flip;
  // Returns the upper 512 bits of the 1024-bit result.
  bool hi;

  unsigned blockBits() const { return elemBits * rows * cols; }
  // The matrix spans lhs:rhs, so the mode reads both operands.
  bool isTwoInput() const { return blockBits() == 1024; }
};

ShuffleModeShape getShuffleModeShape(ShuffleMode mode);

// Whether the mode exists on AIE2P only.
bool isAIE2POnlyShuffleMode(ShuffleMode mode);

// Byte i of the 512-bit result is byte perm[i] of the 128-byte lhs:rhs. The
// matrix is transposed as out[j*rows+i] = in[i*cols+j]. A matrix smaller than
// 512 bits is transposed in each block of lhs of its size.
std::array<uint8_t, 64> getShuffleModeBytePermutation(ShuffleMode mode);

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECSHUFFLEMODES_H
