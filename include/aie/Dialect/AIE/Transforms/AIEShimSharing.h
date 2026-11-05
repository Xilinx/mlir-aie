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
/// A receiving channel writes whatever arrives into the task it runs, so the
/// array must not send a receiving end's words outside its turn: its senders
/// wait on a gate the host fills only after the end before it is awaited,
/// and each sequence drains all that gate lets them send.
/// Ends are known by their objectFIFO's name, before or after split.
class ShimTransferSpans {
public:
  /// A shim end a runtime sequence transfers through.
  struct End {
    mlir::StringAttr fifo;
    /// An MM2S end, sending into the array.
    bool sends;
  };

  /// A shim input gating a receiving end's senders, per its
  /// `#aie.consumed_by`.
  struct Gate {
    mlir::StringAttr input;
    int64_t inputBytes;
    int64_t outputBytes;
  };

  /// `endOf` maps a symbol a runtime sequence transfers through to the
  /// objectFIFO end it belongs to, or nothing when it belongs to none.
  /// `gates` holds each receiving end's gates, by objectFIFO name; without
  /// them, ends are counted as the runtime order alone allows, which is all a
  /// caller running before the proofs can know.
  ShimTransferSpans(
      DeviceOp device,
      llvm::function_ref<std::optional<End>(mlir::StringAttr)> endOf,
      std::optional<llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<Gate>>>
          gates = std::nullopt);

  /// Each receiving end's gates, by objectFIFO name: the shim inputs whose
  /// consumption its progress proves, from the `#aie.consumed_by` proofs
  /// aie-objectfifo-verify leaves on split route endpoints.
  static llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<Gate>>
  provenGates(DeviceOp device);

  /// Whether the ends of `a` and `b` are never in flight together.
  bool apart(mlir::StringAttr a, mlir::StringAttr b) const;

  /// Splits `ends` into groups whose members are pairwise apart, each group
  /// taking one channel: first fit, ordered by the ends' spans in flight.
  /// Returns indices into `ends`.
  llvm::SmallVector<llvm::SmallVector<unsigned>>
  groups(llvm::ArrayRef<mlir::StringAttr> ends) const;

private:
  struct Span {
    int64_t first = std::numeric_limits<int64_t>::max();
    int64_t last = -1;
    int64_t lastIssue = -1;
    int64_t lastDone = -1;
    /// As lastDone, by an await alone: a free does not wait for the task.
    int64_t lastAwaited = -1;
    bool sends = false;
    /// Bytes the sequence moves through the end; nullopt when not static.
    std::optional<int64_t> bytes = 0;
  };

  /// The gates `end` may share by: all of a receiving end's that `contained`
  /// holds for, or one null gate for an end that needs none.
  llvm::SmallVector<const Gate *> usableGates(mlir::StringAttr end) const;
  /// Whether, in one sequence, `x` is done before `y` starts and, when `gate`
  /// is set, awaited before `gate` is first filled.
  bool before(const llvm::DenseMap<mlir::StringAttr, Span> &byFifo,
              const Span &x, const Span &y, const Gate *gate) const;
  /// Whether `gate` is filled only in sequences that drain the receiving
  /// `end`, each awaiting every byte the gate lets its senders send.
  bool contained(mlir::StringAttr end, const Gate &gate) const;

  llvm::SmallVector<llvm::DenseMap<mlir::StringAttr, Span>> spans;
  std::optional<llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<Gate>>>
      gates;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIESHIMSHARING_H
