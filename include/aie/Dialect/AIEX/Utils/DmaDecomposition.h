//===- DmaDecomposition.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decomposition of oversized non-contiguous aiex.npu.dma_memcpy_nd access
// patterns into one or more hardware-legal ND patterns. Shared by the
// aie-decompose-large-dma-bd pass and (after integration) the op verifier.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEX_UTILS_DMADECOMPOSITION_H
#define AIE_DIALECT_AIEX_UTILS_DMADECOMPOSITION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::AIE {
class AIETargetModel;
} // namespace xilinx::AIE

namespace xilinx::AIEX {

/// Number of ND addressing dimensions in the NPU DMA instruction encoding
/// (d0..d2 wrap/stride + d3 iteration/repeat).
static constexpr unsigned kNdDmaDims = 4;

/// Innermost-first ND access pattern (d0..d3 / repeat), matching the
/// convention used by verifyStridesWraps and NpuDmaMemcpyNdOp verification.
///
/// A runtime-sequence task BD may give more than kNdDmaDims dimensions, all of
/// the ones past d2 iteration dimensions; decomposeNdDmaPattern reduces it to
/// kNdDmaDims-dimension patterns.
struct NdDmaPattern {
  llvm::SmallVector<int64_t, kNdDmaDims> offsets;
  llvm::SmallVector<int64_t, kNdDmaDims> sizes;
  llvm::SmallVector<int64_t, kNdDmaDims> strides;
  /// Elements added to the address the offsets give. Carries the position of
  /// a pattern peeled off the dimensions past kNdDmaDims, which no dimension
  /// of the peeled pattern can express.
  int64_t baseOffset = 0;
};

/// Returns true when the pattern passes verifyStridesWraps for the given tile
/// and memref element type (skipTransformationChecks=false). Diagnostics are
/// suppressed so callers can probe legality without emitting errors.
bool patternPassesVerification(mlir::Operation *forOp,
                               mlir::BaseMemRefType referencedBufType,
                               const xilinx::AIE::AIETargetModel &tm,
                               int tileCol, int tileRow,
                               const NdDmaPattern &pattern);

/// True when the pattern is fully constant, non-contiguous, currently illegal,
/// and decomposeNdDmaPattern would produce at least one legal replacement.
bool isDecomposableNdDmaPattern(mlir::Operation *forOp,
                                mlir::BaseMemRefType referencedBufType,
                                const xilinx::AIE::AIETargetModel &targetModel,
                                int tileCol, int tileRow,
                                llvm::ArrayRef<int64_t> offsetsInnermostFirst,
                                llvm::ArrayRef<int64_t> sizesInnermostFirst,
                                llvm::ArrayRef<int64_t> stridesInnermostFirst);

/// Decompose an illegal pattern into one or more legal sub-patterns that move
/// the same data. Prefers dimension factoring (single-op result when possible);
/// falls back to slicing (multiple ops). Returns failure when no legal
/// decomposition exists.
///
/// A pattern with more than kNdDmaDims dimensions first has its iteration
/// dimensions (d3 and up) merged where one continues the next, then, if more
/// than one remains, is peeled into one pattern per index of the dimensions
/// past d3, outermost slowest, each decomposed in turn. One execution of the
/// original visits one index of the iteration dimensions, so the result keeps
/// the order of executions as well as of elements, but a peeled pattern
/// iterates only over d3: its executions are a share of the original's.
mlir::FailureOr<llvm::SmallVector<NdDmaPattern>> decomposeNdDmaPattern(
    mlir::Operation *forOp, mlir::BaseMemRefType referencedBufType,
    const NdDmaPattern &pattern, const xilinx::AIE::AIETargetModel &targetModel,
    int tileCol, int tileRow);

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_DMADECOMPOSITION_H
