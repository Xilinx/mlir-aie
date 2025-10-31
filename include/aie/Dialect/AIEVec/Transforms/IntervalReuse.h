//===- IntervalReuse.h - AIE Vector Data Reuse Computation ------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// Class to compute potential data reuse in AIE vectors
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_INTERVALREUSE_H
#define AIE_DIALECT_AIEVEC_TRANSFORMS_INTERVALREUSE_H

#include "aie/Dialect/AIEVec/AIEVecUtils.h"

#include <utility>

namespace xilinx::aievec {

// The IntervalReuse class.
// This class captures the potential data reuse in the AIE vector. Each load
// from memory (coming from transfer_read op) will exclusively belong to one of
// the IntervalReuse object. Note that in AIE, the minimum vector size is
// 128-bit, and they are aligned to the vector size boundary.
// Let A and B be NxN arrays of 32-bit values. We assume no aliasing. Consider
// the following three cases, where we want to determine which IntervalReuse
// object the two read op accesses within the same loop nest will be mapped to:
// 1. Accesses A[i][j:j+8] and A[i][j+1:j+9]: they exhibit data reuse if we load
// the range  A[j:j+16] from memory into a 512-bit vector. Therefore, these two
// read ops will belong to the same IntervalReuse object.
// 2. Accesses A[i][j:j+8] and A[i+1][j:j+8]: there is no non-trivial
// vector-level reuse possible for these accesses, and they must belong to
// different IntervalReuse objects.
// 3. Accesses A[i][j:j+8] and B[i][j:j+8]: there is no possible reuse, since
// the read ops access different arrays. Therefore, these two read ops must
// belong to different IntervalReuse objects.
// We want to correctly map read ops to IntervalReuse object in all these three
// cases.
//
// The class contains 'memref' and 'base' to disambiguate reuse. 'memref'
// differentiates different arrays. For example, 'memref' for arrays A and B
// will be different. For accesses coming from the same array, 'base' helps in
// disambiguation. 'base' is just the linearized base expression of the access.
// The linearized expression for A[i][j] is i*N+j. We decompose it into base
// (i*N+j), and offset 0. In contrast, the linearized expression for
// A[i+1][j+2] is (i+1)*N+j+2, and we decompose it into base (i+1)*N+j, and
// offset 2. Basically, we abstract away the constant offset from the
// linearized access expression to form the base.
// Given a vector of IntervalReuse objects, we just search for an object with
// matching 'memref' and 'base' to group the read ops that can potentially
// reuse data in vector.
//
// 'extentMap' stores the extent of data that an op reads. We store the extent
// in bits. For example, the extent for operation reading A[i][j:j+8] is
// [0,256].
// The 'read access extent' corresponds to the aligned chunk of data that an
// operation loads. For example, an 8-lane, 32-bit vector load from A[i+7:i+15]
// would have read access extent [0:512], whereas under same conditions, the
// vector load from A[i+9:i+17] would have read access extent [512:768]. Note
// how the extents are aligned to vector size boundary.
//
// 'intervals' merges overlapping intervals to give the view of actual AIE
// vectors that need to be created. Since AIE only allows aligned vector loads,
// each interval is aligned to vector size. Continuing with the previous
// example, the two extents [0,256] and [128,512] overlap. Therefore these will
// be merged together to form a single interval [0,512]. The entries into
// 'intervals' are sorted, so given an op, we can find its interval by doing
// binary search with the op's extent as key (e.g, searching for [128,256] in
// {[0,512],[512,1024]}).

class IntervalReuse {
  // differentiate arrays (e.g., A vs. B)
  mlir::Value memref;
  // differentiate accesses coming from the same array, but with different base
  // expression along the non-vectorized dimension (e.g., A[i+1][j:j+8] vs.
  // A[i][j:j+8];
  mlir::AffineExpr base;
  // A map from each read operation to the extent of bits it reads (aligned to
  // vector size).
  llvm::DenseMap<mlir::Operation *, std::pair<int32_t, int32_t>> extentMap;
  // Disjoint intervals of all the data accesses (i.e., read bits). Each
  // interval entry corresponds to memory load into an AIE vec.
  llvm::SmallVector<std::pair<int32_t, int32_t>, 8> intervals;
  // Identify all the vectors that are only used as LHS operands of mul/mac op.
  // The LHS operand of mul/mac ops have specific size requirement.
  llvm::SmallVector<bool, 8> vecIsLHSOperand;

  // Return true if this array access comes from the same array
  bool sameMemRef(mlir::Value m) { return memref == m; }
  // Return true if this array access has the same invariant base
  // expression.
  bool sameInvariantIndices(mlir::AffineExpr b) { return base == b; }
  // Return true if this array access is enclosed within the same loop nest as
  // other accesses belonging to the same IntervalReuse object.
  bool sameEnclosingLoops(
      mlir::Operation *op,
      llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Operation *, 8>>
          &blockToEnclosingLoops);
  // For an operation, get the index into intervals that subsumes the
  // operation's access extent.
  size_t getIntervalIndex(mlir::Operation *op);

public:
  // Return true if this read operation has a potential data reuse with other
  // read operations in this IntervalReuse.
  bool potentialReuse(
      mlir::vector::TransferReadOp readOp, mlir::AffineExpr invariantBase,
      llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Operation *, 8>>
          &blockToEnclosingLoops);
  // Insert the access extent of this read operation into intervals
  void insertInterval(mlir::vector::TransferReadOp readOp,
                      llvm::DenseMap<mlir::Operation *, IntervalReuse *>
                          &dataAccessToIntervalMap,
                      int32_t offset, int32_t forLoopStepSize,
                      bool isSplat = false,
                      unsigned minVecSize = 128 /*min AIE vec size*/);
  // For a read operation, return the width of the interval its access extent
  // belongs to. The interval width corresponds to the size of the vector that
  // will hold the load from read operation.
  int32_t getIntervalWidth(mlir::Operation *op);
  // Get the read access extent of this read operation. The returned value
  // indicates the start and end offsets of the access from the base (in bits).
  std::pair<int32_t, int32_t> getAccessExtent(mlir::Operation *op);
  // Set the read access extent of this read operation.
  void setAccessExtent(mlir::Operation *op,
                       std::pair<int32_t, int32_t> &extent);
  // Get the interval that contains this read operation's extent
  std::pair<int32_t, int32_t> getInterval(mlir::Operation *op);
  // Given that the read operation 'op' is only LHS operand of some mul/mac
  // op, mark the vector that will load its access extent.
  void markLHSOperandVec(mlir::Operation *op);
  // If the interval corresponds to a vector that is marked as the exclusive
  // LHS operand of some mul/mac op, and if its size is <= 256, try to coalesce
  // it with the next interval.
  void coalesceIntervals();
  // Constructors
  IntervalReuse(mlir::vector::TransferReadOp readOp, mlir::AffineExpr b)
      : memref(readOp.getBase()), base(b) {}
  IntervalReuse() : memref(nullptr), base(nullptr) {}
};

} // namespace xilinx::aievec
// end namespace xilinx

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_INTERVALREUSE_H
