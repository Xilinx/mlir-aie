//===- AIEVecMatMul.h - Matmul recipes and tile layouts ---------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The tile multiplies that `aievec.matmul` (AIE2) and `aievec.matmul_aie2p`
// (AIE2P) lower to, and the operand layouts each of them reads. The verifier
// of the matmul ops, the lowering of `vector.contract` and the lowering to
// LLVM all choose from this one table.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECMATMUL_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECMATMUL_H

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace xilinx::aievec {

// One way to multiply an m x k lhs tile by a k x n rhs tile into an m x n
// accumulator. Names give the operand types and m x k x n.
enum class MatMulRecipeKind {
  // AIE2: one `mac.conf` each.
  AIE2_I8xI4_4x16x8,
  AIE2_I8xI8_4x8x8,
  AIE2_I16xI8_4x4x8,
  AIE2_I16xI16_4x2x8,
  AIE2_I16xI8_2x8x8_Acc64,
  AIE2_I16xI8_4x8x4_Acc64,
  AIE2_I16xI16_2x4x8_Acc64,
  AIE2_I16xI16_4x4x4_Acc64,
  AIE2_I32xI16_4x2x4_Acc64,
  AIE2_BF16_4x8x4,
  // AIE2P integer matrix modes.
  AIE2P_I8_8x8x8,
  AIE2P_I16_8x2x8,
  // AIE2P bf16 through bfp16 (block floating point, one exponent per 8 values
  // along k), the `8x8_8x8T` matrix mode.
  AIE2P_BFP16_8x8x8,
  AIE2P_BFP16_4x8x8,
  // AIE2P bf16 through elementwise multiplies of broadcast rows and columns.
  AIE2P_BF16Elem_8x8x4,
  AIE2P_BF16Elem_4x8x4,
  AIE2P_BF16Outer_8x1x8,
};

struct MatMulRecipe {
  MatMulRecipeKind kind;
  // The matmul ops take lhs as [m, k] and rhs as [k, n]. These say whether
  // the recipe reads lhs as [k, m] and rhs as [n, k] instead, so that an
  // operand stored that way needs no shuffle.
  bool lhsKM;
  bool rhsNK;
};

// The recipe for an m x k x n tile with these element types, if there is one.
// Integer types are matched by width; their signedness is a separate MAC
// setting.
std::optional<MatMulRecipe> getMatMulRecipe(bool aie2p, mlir::Type lhsElemTy,
                                            mlir::Type rhsElemTy,
                                            mlir::Type accElemTy, int64_t m,
                                            int64_t k, int64_t n);

// The shuffle modes that transpose a rows x cols tile of `elemTy` held
// row-major. Empty when a dim is 1, as the tile then needs no data movement.
// One mode when the tile fills 512 bits, or 256 bits on AIE2P (the tile then
// sits in the low half of a 512-bit register). Two modes, the low and high
// halves of the result, for an 8x8 tile of 16-bit values on AIE2P. Nullopt
// for any other tile.
std::optional<llvm::SmallVector<ShuffleMode, 2>>
getTileTransposeModes(bool aie2p, mlir::Type elemTy, int64_t rows,
                      int64_t cols);

// Checks that lhs times rhs into acc is a tile some recipe multiplies. lhs is
// stored [m, k], or [k, m] when `lhsTransposed`; rhs is stored [k, n], or
// [n, k] when `rhsTransposed`; acc is [m, n]. Each stored layout must be the
// one the recipe reads, or one that getTileTransposeModes turns into it.
// Returns the recipe. On failure, reports why through `emitError` unless it
// is null.
std::optional<MatMulRecipe> checkMatMulTile(
    bool aie2p, mlir::VectorType lhsTy, mlir::VectorType rhsTy,
    mlir::VectorType accTy, bool lhsTransposed, bool rhsTransposed,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError = nullptr);

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECMATMUL_H
