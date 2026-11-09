//===- AIEShimSharing.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIESHIMSHARING_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIESHIMSHARING_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <tuple>

namespace xilinx::AIE {

/// Shim endpoint lifetimes, including calls to self-contained runtime
/// sequences. MM2S transfers conservatively remain live until their sequence
/// returns.
class ShimTransferSpans {
public:
  /// A shim end a runtime sequence transfers through.
  struct End {
    mlir::StringAttr fifo;
    /// An MM2S end, sending into the array.
    bool sends;
  };

  /// `endOf` maps a symbol a runtime sequence transfers through to the
  /// objectFIFO end it belongs to, or nothing when it belongs to none.
  ShimTransferSpans(
      DeviceOp device,
      llvm::function_ref<std::optional<End>(mlir::StringAttr)> endOf);

  ShimTransferSpans(
      mlir::ModuleOp module,
      llvm::function_ref<std::optional<End>(DeviceOp, mlir::StringAttr)> endOf);

  /// Whether both ends permit sharing, explicitly or by MM2S lifetime analysis.
  bool apart(mlir::StringAttr a, mlir::StringAttr b) const;

  bool isUsed(mlir::StringAttr end) const;

  /// Splits `ends` into groups whose members are pairwise apart, each group
  /// taking one channel: first fit, ordered by the ends' spans in flight.
  /// Returns indices into `ends`.
  llvm::SmallVector<llvm::SmallVector<unsigned>>
  groups(llvm::ArrayRef<mlir::StringAttr> ends) const;

private:
  void analyze(
      mlir::Operation *root,
      llvm::function_ref<std::optional<End>(DeviceOp, mlir::StringAttr)> endOf);

  llvm::DenseMap<mlir::StringAttr,
                 llvm::SmallVector<std::tuple<unsigned, int64_t, int64_t>>>
      lifetimes;
  llvm::DenseSet<mlir::StringAttr> used;
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::StringAttr>>
      permissions;
  llvm::DenseSet<std::pair<mlir::StringAttr, mlir::StringAttr>> overlaps;
  bool unknown = false;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIESHIMSHARING_H
