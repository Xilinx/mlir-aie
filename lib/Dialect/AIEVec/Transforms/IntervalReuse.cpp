//===-IntervalReuse.cpp - Interval Analysis for AIE Vectors -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file implements the interval abstraction for AIE vectors. The
// abstraction is essential to generate UPD instructions, and compute the
// start/offset in AIE MAC/MUL intrinsic.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/Transforms/IntervalReuse.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"

#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace xilinx::aievec;

#define DEBUG_TYPE "aie-reuse"

// Return true if the read operation is enclosed with the same loop nests as
// the other read operations belonging to this IntervalReuse object.
bool IntervalReuse::sameEnclosingLoops(
    Operation *op, mlir::DenseMap<Block *, SmallVector<Operation *, 8>>
                       &blockToEnclosingLoops) {
  // Assert that there are some existing read operations in the interval
  assert(!extentMap.empty() &&
         "interval must have at least one read operation");
  // Get any of the previous read operation belonging to this interval as
  // reference.
  Operation *ref = extentMap.begin()->first;

  // Assert that we have computed the enclosing loops for the reference and
  // current read op.
  assert(blockToEnclosingLoops.count(op->getBlock()) &&
         "block to enclosing loop mapping not computed");
  assert(blockToEnclosingLoops.count(ref->getBlock()) &&
         "block to enclosing loop mapping not computed");
  // return true if both reference and op are enclosed in the same loop nest
  return blockToEnclosingLoops[op->getBlock()] ==
         blockToEnclosingLoops[ref->getBlock()];
}

// Given a read operation, return the index of the interval that completely
// subsumes this operation's access extent.
size_t IntervalReuse::getIntervalIndex(Operation *op) {
  // Get the access extent for this operation
  auto bound = getAccessExtent(op);
  // Do a binary search to find the interval that subsumes the bound. Note
  // that the intervals are sorted and disjoint.
  auto lb = std::lower_bound(intervals.begin(), intervals.end(), bound);
  // If the intervals are [0,512],[512,1024], and we want to find [256,512],
  // then the lower_bound will index into [512,1024]. We need to check the
  // previous entry to find the correct bound.
  if (lb != intervals.begin()) {
    auto prev = std::prev(lb);
    if (prev->first <= bound.first && prev->second >= bound.second)
      lb = prev;
  }
  assert(lb != intervals.end() &&
         "Failed to find correct interval for read operation");
  // Assert that we found the right interval
  assert(lb->first <= bound.first && lb->second >= bound.second &&
         "Failed to find correct interval for read operation");
  size_t pos = std::distance(intervals.begin(), lb);
  return pos;
}

// Given a read operation belonging to this IntervalReuse object, return its
// access extent.
std::pair<int32_t, int32_t> IntervalReuse::getAccessExtent(Operation *op) {
  assert(extentMap.find(op) != extentMap.end() &&
         "Could not find the bounds of operator in map");
  return extentMap[op];
}

// Set the access extent of a preexisting read operation in this IntervalReuse
// object.
void IntervalReuse::setAccessExtent(Operation *op,
                                    std::pair<int32_t, int32_t> &extent) {
  assert(extentMap.find(op) != extentMap.end() &&
         "operation does not belong to this reuse interval");
  extentMap[op] = extent;
}

// For a read operation belonging to this IntervalReuse object, get the
// interval that subsumes its read access extent.
std::pair<int32_t, int32_t> IntervalReuse::getInterval(Operation *op) {
  size_t pos = getIntervalIndex(op);
  return intervals[pos];
}

// This read operation is only the LHS operand of a mul/mac op. So tag the
// vector corresponding to its access interval. The tagging helps in coalescing
// vectors for i8xi8 scheme.
void IntervalReuse::markLHSOperandVec(Operation *op) {
  // If vecIsLHSOperand is empty, initialize its size to the size of intervals
  if (vecIsLHSOperand.empty())
    vecIsLHSOperand.resize(intervals.size(), false);
  // Get the position of this operation's access in the interval
  size_t pos = getIntervalIndex(op);
  assert(pos < vecIsLHSOperand.size());
  // Set the corresponding index in vecIsLHSOperand to true
  vecIsLHSOperand[pos] = true;
}

// For a read operation belonging to this reuse interval, return the interval
// width. The interval size corresponds to the size of the vector that this
// read op will get the data from.
int32_t IntervalReuse::getIntervalWidth(Operation *op) {
  auto interval = getInterval(op);
  return interval.second - interval.first;
}

// Function to detect potential reuse in vector among different read ops. First
// check if the same array is read by the read ops. Then check if their base
// expr is the same. Finally, the accesses must belong to the same loop nest.
// If all these conditions are met, there is a potential data reuse.
bool IntervalReuse::potentialReuse(
    vector::TransferReadOp readOp, AffineExpr invariantBase,
    mlir::DenseMap<Block *, SmallVector<Operation *, 8>>
        &blockToEnclosingLoops) {
  return sameMemRef(readOp.getSource()) &&
         sameInvariantIndices(invariantBase) &&
         sameEnclosingLoops(readOp, blockToEnclosingLoops);
}

// For the given vector read operation, compute the access bounds. For example,
// if the vector size is 1x8, and the read is A[N:N+7], then the bound is [N,
// N+7]. The returned bounds are vector size aligned. This means that if the
// access is A[N+1:N+8], and the vector size is 256 bits, then the returned
// bound is [N:N+15]. We assume that each row of the array is properly aligned.
static std::pair<int32_t, int32_t>
computeAccessExtent(vector::TransferReadOp readOp, int32_t offset,
                    int32_t loopStepSize, bool isSplat, unsigned minVecSize) {
  VectorType vType = cast<VectorType>(readOp.getResult().getType());
  unsigned vecSize = getVectorLaneSize(vType);
  int32_t elementSizeInBits = getElementSizeInBits(vType);
  // Create chunks greater in size than minVecSize
  int32_t vecSizeInBits = std::max(minVecSize, vecSize * elementSizeInBits);

  assert(isPowerOfTwo(vecSizeInBits) &&
         "Current support is only for power-of-two vector sizes");

  // Below computation is base on the assumption that vectorization factor is
  // always a power of 2.
  int32_t lb = (offset * elementSizeInBits) & ~(vecSizeInBits - 1);
  int32_t ub = (isSplat ? (offset & ~(vecSize - 1)) + vecSize
                        : offset + loopStepSize * vecSize) *
               elementSizeInBits;
  // Adjust to the nearest multiple of vecSizeInBits
  ub = (ub + vecSizeInBits - 1) & ~(vecSizeInBits - 1);

  return std::make_pair(lb, ub);
}

// Insert a new interval for the read operation into already existing
// intervals. This gives us an interval reuse graph, which can then be used to
// determine the AIE vector sizes. The minimum vector size for AIE is 128 bits,
// so align the access extents to at least 128-bit boundary.
void IntervalReuse::insertInterval(
    vector::TransferReadOp readOp,
    mlir::DenseMap<Operation *, IntervalReuse *> &opToIntervalMap,
    int32_t offset, int32_t loopStepSize, bool isSplat, unsigned minVecSize) {
  // Get the vector-size-aligned lower and upper bounds for the vector read
  std::pair<int32_t, int32_t> bound =
      computeAccessExtent(readOp, offset, loopStepSize, isSplat, minVecSize);

  // Make an entry into extentMap
  extentMap[readOp] = bound;
  // Make an entry into readOp->interval map
  opToIntervalMap[readOp] = this;

  LLVM_DEBUG(llvm::dbgs() << "\n\nInserting access extent [" << bound.first
                          << "," << bound.second << "] for read op " << readOp);

  // If the interval already exists, we're done
  if (std::find(intervals.begin(), intervals.end(), bound) != intervals.end()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "\n\tPre-existing interval already subsumes the access extent");
    return;
  }
  // Insert bound into the current interval set. We do so by using O(N) space,
  // creating a new mergedIntervals, and replacing intervals with it.
  int32_t lb = bound.first, ub = bound.second;
  SmallVector<std::pair<int32_t, int32_t>, 8> mergedIntervals;
  bool inserted = false;

  // Iterate over all the existing intervals, and merge them with the bound.
  // There are three cases: (1) the interval falls before the bound; (2) the
  // interval intersects with bound, and (3) the interval falls after the
  // bound.  The idea is to take care of (2), while just blindly inserting
  // intervals that don't intersect with bounds.
  for (auto iv : intervals) {
    // No interference, interval i is definitely before new interval
    if (iv.second <= lb)
      mergedIntervals.push_back(iv);
    else if (iv.first >= ub) {
      // Interference finished at this point. Resolve if necessary
      if (!inserted) {
        mergedIntervals.push_back(std::make_pair(lb, ub));
        inserted = true;
      }
      mergedIntervals.push_back(iv);
    } else {
      // Interference continues. Compute the overlap
      lb = std::min(lb, iv.first);
      ub = std::max(ub, iv.second);
    }
  }
  if (!inserted)
    mergedIntervals.push_back(std::make_pair(lb, ub));

  intervals.clear();
  intervals = std::move(mergedIntervals);

  // Verify that each interval is within the AIE vector size limit
  assert([&] {
    for (auto iv : intervals) {
      int32_t width = iv.second - iv.first;
      if (width > 1024) {
        printf("Vector width > 1024 currently not supported");
        return false;
      }
    }
    return true;
  }());

  // Print out the merged intervals
  LLVM_DEBUG(llvm::dbgs() << "\n\tAfter inserting access extent, intervals: ");
#ifndef NDEBUG
  for (auto iv : intervals)
    LLVM_DEBUG(llvm::dbgs() << "[" << iv.first << "," << iv.second << "] ");
#endif
}

// If a vector corresponding to any interval was tagged as exclusively being
// the LHS operand of a mul/fma op, run a vector coalescing loop. The loop is
// pretty simple. It tries to identify two vectors that (1) are consecutive,
// (2) tagged, and (3) the combined size is 512 bits, and coalesces them into a
// single vector.
void IntervalReuse::coalesceIntervals() {
  // Only proceed if any vector was tagged as exclusive LHS operand of mul/mac
  // op.
  if (vecIsLHSOperand.empty())
    return;

  // First check to see if we can coalesce any vector. The vector should be
  // tagged, and its size should be <= 256 bits.
  bool canCoalesce = false;
  for (size_t i = 0, e = intervals.size(); i < e; ++i) {
    canCoalesce |=
        vecIsLHSOperand[i] && intervals[i].second - intervals[i].first <= 256;
  }
  if (!canCoalesce)
    return;

  // Now we try to coalesce
  SmallVector<std::pair<int32_t, int32_t>, 8> coalescedIntervals;
  for (size_t i = 0, e = intervals.size(); i < e;) {
    // Check 1. Two consecutive vectors that can be fused
    if (vecIsLHSOperand[i] && i < intervals.size() - 1 &&
        vecIsLHSOperand[i + 1]) {
      // Get vector sizes
      int32_t v1size = intervals[i].second - intervals[i].first;
      int32_t v2size = intervals[i + 1].second - intervals[i + 1].first;
      // Check 2. Both vectors must be <= 256 bits, so that their sum is
      // <= 512 bits.
      if (v1size <= 256 && v2size <= 256) {
        coalescedIntervals.push_back(
            std::make_pair(intervals[i].first, intervals[i + 1].second));
        i += 2;
        continue;
      }
    }
    coalescedIntervals.push_back(intervals[i]);
    ++i;
  }
  // We are done with the use of the tagging vector. Erase it.
  vecIsLHSOperand.clear();
  // Replace intervals with the coalesced intervals
  intervals.clear();
  intervals = std::move(coalescedIntervals);

  // Print out coalesced intervals
  LLVM_DEBUG(llvm::dbgs() << "\n\nAfter coalescing for "
                          << "i8xi8 scheme, intervals: ");
#ifndef NDEBUG
  for (auto iv : intervals)
    LLVM_DEBUG(llvm::dbgs() << "[" << iv.first << "," << iv.second << "] ");
#endif
}
