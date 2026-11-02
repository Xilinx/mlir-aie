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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace xilinx::AIE {

/// When the runtime sequences have each shim end's transfers in flight, so
/// that ends never in flight together can take turns on one shim channel.
/// Different runtime sequences never overlap (a dispatch finishes its
/// transfers before the next starts). Within one, a receiving end's transfers
/// must all be awaited or freed before the other's first, and a loop spans
/// its body; a sending end stays in flight to the end of its sequence, as a
/// shim MM2S task reports itself complete before its words leave the shim.
/// Ends are known by their objectFIFO's name, before or after split.
/// The placer and objectFIFO allocation both decide by this, so they agree.
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

  /// Whether the ends of `a` and `b` are never in flight together.
  bool apart(mlir::StringAttr a, mlir::StringAttr b) const;

  /// Splits `ends` into groups whose members are pairwise apart, each group
  /// taking one channel: first fit, in name order. Returns indices into `ends`.
  llvm::SmallVector<llvm::SmallVector<unsigned>>
  groups(llvm::ArrayRef<mlir::StringAttr> ends) const;

private:
  struct Span {
    int64_t first = std::numeric_limits<int64_t>::max();
    int64_t last = -1;
    int64_t lastIssue = -1;
    int64_t lastDone = -1;
    bool sends = false;
  };
  llvm::SmallVector<llvm::DenseMap<mlir::StringAttr, Span>> spans;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIESHIMSHARING_H
