//===- AIEStructuralRank.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TRANSFORMS_AIESTRUCTURALRANK_H
#define AIE_TRANSFORMS_AIESTRUCTURALRANK_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <utility>

namespace xilinx::AIE {

/// Ops and the labelled links between them, for ranking them by structure
/// where a pass would otherwise fall back on op order.
struct StructuralGraph {
  llvm::SmallVector<mlir::Operation *> nodes;
  /// What each node is, before anything it links to is considered.
  llvm::DenseMap<mlir::Operation *, std::string> features;
  llvm::DenseMap<mlir::Operation *,
                 llvm::SmallVector<std::pair<std::string, mlir::Operation *>>>
      links;

  /// Add `op` as a node whose features are its name and attributes, symbol
  /// names and references left out. Returns the features to extend.
  std::string &add(mlir::Operation *op);

  /// Link `from` to `to` under `label`, and `to` back to `from`.
  void link(mlir::Operation *from, mlir::Operation *to, llvm::StringRef label);

  /// Rank the nodes by their features and then, a round at a time until no
  /// class splits, by the ranks of the nodes they link to. Neither op order
  /// nor symbol names play a part, so a permuted or renamed design ranks
  /// alike, and nodes left with one rank cannot be told apart.
  llvm::DenseMap<mlir::Operation *, unsigned> rank() const;

  /// The ops directly in `device` but its runtime sequences, each with the
  /// ops nested in it, linked to the ops whose results or symbols they use.
  /// The runtime sequence is left out, so a device ranks the same whatever
  /// sequence drives it.
  static StructuralGraph of(DeviceOp device);
};

} // namespace xilinx::AIE

#endif // AIE_TRANSFORMS_AIESTRUCTURALRANK_H
