//===- AIE2PShufflePlanner.h - Static shuffles on AIE2P ---------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Plans a static vector shuffle as a short sequence of AIE2P vshuffle modes
// and splats, and lowers vector.shuffle and vector.transpose through it.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_AIE2PSHUFFLEPLANNER_H
#define AIE_DIALECT_AIEVEC_TRANSFORMS_AIE2PSHUFFLEPLANNER_H

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <optional>

namespace xilinx::aievec {

// A static shuffle as a gather over labelled lanes. Source k contributes the
// labels offset(k) .. offset(k) + lanes(k) - 1, or the single label offset(k)
// when it is a splat. Lane i of the result must hold label target[i]; a
// negative target lane may hold anything.
struct ShuffleGather {
  unsigned elemBits = 0;
  // Splats can be built with aievec.broadcast_scalar for this element type.
  bool canSplat = false;
  llvm::SmallVector<int64_t> sourceLanes;
  llvm::SmallVector<bool> sourceIsSplat;
  llvm::SmallVector<int32_t> target;

  int32_t sourceOffset(unsigned k) const;
  int32_t numLabels() const { return sourceOffset(sourceLanes.size()); }
  std::string key() const;
};

// A gather to lower: lane i of a `resultType` value is lane mask[i] of
// v1:v2, or poison for a negative mask entry. `user` is the op that reads v1
// and v2; a vector.shuffle feeding only it may be folded into the gather.
struct ShuffleRequest {
  mlir::Value v1, v2;
  llvm::SmallVector<int64_t> mask;
  mlir::VectorType resultType;
  mlir::Operation *user;
};

// The registers a plan builds, in an order where operands come first, and the
// register that holds each 512-bit part of the result.
struct ShufflePlan {
  struct Node {
    enum Kind { Source, Splat, Shuffle } kind;
    // Source: the 512-bit part `chunk` of source `source`, padded when the
    // source is smaller. Splat: every lane holds `label`. Shuffle: `mode`
    // applied to nodes `lhs` and `rhs` (equal for one-input modes).
    unsigned source = 0, chunk = 0;
    int32_t label = -1;
    ShuffleMode mode = ShuffleMode::T512_1X2_LO;
    int lhs = -1, rhs = -1;
  };
  llvm::SmallVector<Node> nodes;
  // -1 for a part with no lane to fill.
  llvm::SmallVector<int> results;
  // Vector instructions: shuffles and splats.
  unsigned cost = 0;

  // The distinct modes of the shuffles.
  llvm::SmallVector<ShuffleMode> modes() const;
};

// Plans and caches the lowering of shuffles. A pattern and the legality
// callback of the same conversion share one planner, so each distinct gather
// is searched once. The planner also remembers the modes of the plans it has
// lowered, so later plans prefer modes already loaded into registers.
class AIE2PShufflePlanner {
public:
  // Plans for the gather, cheapest first, each the cheapest found for the
  // modes it needs. Deterministic and bounded.
  const llvm::SmallVector<ShufflePlan, 2> &
  candidates(const ShuffleGather &gather);

  // The plan to lower the request with, when one is cheaper than what LLVM
  // makes of it. `gather` and `leaves` receive the gather the plan is for and
  // the value behind each of its sources.
  std::optional<ShufflePlan>
  planFor(const ShuffleRequest &request, ShuffleGather *gather = nullptr,
          llvm::SmallVector<mlir::Value> *leaves = nullptr);

  // Records that `plan` has been emitted.
  void noteUsed(const ShufflePlan &plan);

private:
  llvm::StringMap<llvm::SmallVector<ShufflePlan, 2>> cache;
  llvm::SmallVector<ShuffleMode> usedModes;
};

// Estimated vector instructions LLVM spends on the gather: nothing for moves
// of whole 256-bit halves, one per lane splat or lane-wise select, and about
// one per lane when it falls back to inserting lanes one at a time.
unsigned estimateLLVMShuffleCost(const ShuffleGather &gather);

// Checks the plan by running it on lane labels with the byte permutation of
// each mode. Returns false if any result lane differs from the target.
bool simulateShufflePlan(const ShuffleGather &gather, const ShufflePlan &plan);

void populateAIE2PShufflePlannerPatterns(
    mlir::RewritePatternSet &patterns,
    std::shared_ptr<AIE2PShufflePlanner> planner);

// vector.shuffle, and rank-1 vector.interleave and vector.deinterleave, stay
// legal unless the planner lowers them.
void configureAIE2PShufflePlannerLegality(
    mlir::ConversionTarget &target,
    std::shared_ptr<AIE2PShufflePlanner> planner);

} // namespace xilinx::aievec

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_AIE2PSHUFFLEPLANNER_H
