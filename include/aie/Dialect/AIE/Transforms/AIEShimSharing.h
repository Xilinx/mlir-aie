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
/// returns. A receiving channel writes whatever arrives into the task it runs,
/// so a receiving end shares only by explicit permission or where the array
/// cannot send its words outside its turn: its senders wait on a gate the host
/// fills only after the end before it is awaited, and each sequence drains all
/// that gate lets them send.
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
    Gate(mlir::StringAttr input, int64_t inputBytes, int64_t outputBytes,
         int64_t granuleBytes)
        : input(input), inputBytes(inputBytes), outputBytes(outputBytes),
          granuleBytes(granuleBytes) {}
    mlir::StringAttr input;
    int64_t inputBytes;
    int64_t outputBytes;
    /// The proof counts drained bytes only in whole multiples of this, when
    /// nonzero.
    int64_t granuleBytes;
  };

  using Gates = llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<Gate>>;

  /// `endOf` maps a symbol a runtime sequence transfers through to the
  /// objectFIFO end it belongs to, or nothing when it belongs to none.
  /// `gates` holds each receiving end's gates, keyed as `endOf` keys ends;
  /// without them, a receiving end shares only by explicit permission, which
  /// is all a caller running before the proofs can know.
  ShimTransferSpans(
      DeviceOp device,
      llvm::function_ref<std::optional<End>(mlir::StringAttr)> endOf,
      std::optional<Gates> gates = std::nullopt);

  ShimTransferSpans(
      mlir::ModuleOp module,
      llvm::function_ref<std::optional<End>(DeviceOp, mlir::StringAttr)> endOf,
      std::optional<Gates> gates = std::nullopt);

  /// Each receiving end's gates, keyed by `keyOf`: the shim inputs whose
  /// consumption its progress proves, from the `#aie.consumed_by` proofs
  /// aie-objectfifo-verify leaves on split route endpoints. Endpoints `keyOf`
  /// maps to null are left out.
  static Gates
  provenGates(DeviceOp device,
              llvm::function_ref<mlir::StringAttr(RouteEndpointOp)> keyOf);

  /// Whether both ends permit sharing: explicitly, by MM2S lifetime analysis,
  /// or, for a receiving end, by its gates.
  bool apart(mlir::StringAttr a, mlir::StringAttr b) const;

  bool isUsed(mlir::StringAttr end) const;

  /// Splits `ends` into groups whose members are pairwise apart, each group
  /// taking one channel: first fit, ordered by the ends' spans in flight.
  /// Returns indices into `ends`.
  llvm::SmallVector<llvm::SmallVector<unsigned>>
  groups(llvm::ArrayRef<mlir::StringAttr> ends) const;

private:
  /// An end's transfers in one runtime sequence, as the gates are checked.
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

  void analyze(
      mlir::Operation *root,
      llvm::function_ref<std::optional<End>(DeviceOp, mlir::StringAttr)> endOf);
  void analyzeGates(
      mlir::Operation *root,
      llvm::function_ref<std::optional<End>(DeviceOp, mlir::StringAttr)> endOf);

  /// Whether the receiving ends `a` and `b` take turns by their gates.
  bool gated(mlir::StringAttr a, mlir::StringAttr b) const;
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

  llvm::DenseMap<mlir::StringAttr,
                 llvm::SmallVector<std::tuple<unsigned, int64_t, int64_t>>>
      lifetimes;
  llvm::DenseSet<mlir::StringAttr> used;
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::StringAttr>>
      permissions;
  /// Receiving ends no explicit list covers.
  llvm::DenseSet<mlir::StringAttr> receivers;
  llvm::DenseSet<std::pair<mlir::StringAttr, mlir::StringAttr>> overlaps;
  bool unknown = false;
  llvm::SmallVector<llvm::DenseMap<mlir::StringAttr, Span>> spans;
  std::optional<Gates> gates;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIESHIMSHARING_H
