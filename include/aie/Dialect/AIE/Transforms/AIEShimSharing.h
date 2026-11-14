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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <tuple>

namespace xilinx::AIE {

/// The shim MM2S channels `device`'s overlay weighs, for
/// aie-generate-column-control-overlay. A route asking for no packet header
/// counts as packet switched with `packetSwitched`. Defined in
/// AIEGenerateColumnControlOverlay.cpp.
struct ShimMM2SUses {
  /// Channels a circuit-switched route end, flow, allocation or shim mux
  /// holds, per shim tile op.
  llvm::DenseMap<mlir::Operation *, llvm::SmallSet<int, 2>> circuits;
  /// Packet-switched route ends, by name, per (tile, channel).
  llvm::DenseMap<std::pair<mlir::Operation *, int>,
                 llvm::SmallVector<mlir::StringAttr>>
      packetEnds;
  /// Packet flows per (tile, channel).
  llvm::DenseMap<std::pair<mlir::Operation *, int>, unsigned> packetFlows;
};
ShimMM2SUses shimMM2SUses(DeviceOp device, bool packetSwitched);

/// Shim endpoint lifetimes, including calls to self-contained runtime
/// sequences. MM2S transfers conservatively remain live until their sequence
/// returns; a callee's return and a dispatch boundary are themselves taken as
/// the end of its sends, by the aiex.run and hardware-context contracts, not
/// by anything this analysis computes. A receiving channel writes whatever
/// arrives into the task it runs, so a receiving end shares only by explicit
/// permission or where the array cannot send its words outside its turn: its
/// senders wait on a gate the host fills only after the end before it is
/// awaited, and each sequence drains all that gate lets them send.
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
    Gate(mlir::StringAttr input, ConsumedByAttr proof)
        : input(input), proof(proof) {}
    mlir::StringAttr input;
    ConsumedByAttr proof;
  };

  using Gates = llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<Gate>>;

  /// How ends and gates name a shim route end: by its objectFIFO, else by its
  /// own symbol.
  static mlir::StringAttr keyOf(RouteEndpointOp endpoint);
  /// The end a runtime sequence of `device` transfers through when it names
  /// `symbol`: a shim DMA route end, keyed by keyOf, or, before
  /// aie-objectfifo-split, an objectFIFO, sending when a shim tile produces
  /// it.
  static std::optional<End> endOf(DeviceOp device, mlir::StringAttr symbol);

  /// `endOf` maps a symbol a runtime sequence transfers through to the
  /// objectFIFO end it belongs to, or nothing when it belongs to none.
  /// `gates` holds each receiving end's gates, keyed as `endOf` keys ends;
  /// without them, a receiving end shares only by explicit permission, which
  /// is all a caller running before the proofs can know. With
  /// `plannedOverlay`, the control overlay's channel on each column's shim
  /// tile is an end too, overlayEnd(), before
  /// aie-generate-column-control-overlay makes it: it carries the control
  /// packets to its column and every configuration load, as the overlay's
  /// own channel does once made.
  ShimTransferSpans(
      DeviceOp device,
      llvm::function_ref<std::optional<End>(mlir::StringAttr)> endOf,
      std::optional<Gates> gates = std::nullopt, bool plannedOverlay = false);

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

  /// The planned control overlay channel's end on column `column`'s shim tile.
  mlir::StringAttr overlayEnd(int column) const;

  /// The number of runtime sequences `bytes` indexes into.
  unsigned sequences() const { return spans.size(); }
  /// The bytes runtime sequence `seq` moves through `end`, nullopt when not
  /// static or when `end` is not one of its ends.
  std::optional<int64_t> bytes(unsigned seq, mlir::StringAttr end) const {
    auto it = spans[seq].find(end);
    return it == spans[seq].end() ? std::nullopt : it->second.bytes;
  }

  /// Splits `ends` into groups whose members are pairwise apart, each group
  /// taking one channel: first fit, ordered by the ends' spans in flight.
  /// Returns indices into `ends`.
  llvm::SmallVector<llvm::SmallVector<unsigned>>
  groups(llvm::ArrayRef<mlir::StringAttr> ends) const;

  /// A shim allocation's channel the ends its `aie.shareable_with` lists may
  /// take turns on.
  struct SharedChannel {
    SharedChannel(llvm::SmallVector<mlir::StringAttr> listed, bool pinned)
        : listed(std::move(listed)), pinned(pinned) {}
    llvm::SmallVector<mlir::StringAttr> listed;
    /// Whether a route end pins the channel, which then takes no other end.
    bool pinned;
  };

  /// How one shim tile's unpinned ends in one direction share channels, as
  /// objectFIFO allocation shares them.
  struct Plan {
    /// Per end, the index of the shared channel it takes, if any.
    llvm::SmallVector<std::optional<unsigned>> sharedChannel;
    /// The other ends, in groups taking one channel each.
    llvm::SmallVector<llvm::SmallVector<unsigned>> groups;
  };

  /// Plans `ends` (keyed as apart() takes keys); `names[i]` is how `shared`'s
  /// lists name `ends[i]`. Each channel in `shared`, in order, takes the ends
  /// its list names that are apart from those it already took; the rest
  /// group as groups() groups them. Indices are into `ends`.
  Plan plan(llvm::ArrayRef<mlir::StringAttr> ends,
            llvm::ArrayRef<mlir::StringAttr> names,
            llvm::ArrayRef<SharedChannel> shared) const;

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
  mlir::MLIRContext *context = nullptr;
  /// usableGates' answer, by end: depends only on the immutable analysis.
  mutable llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<const Gate *>>
      usableGatesCache;
  /// Whether some sequence names an end, by end.
  mutable llvm::DenseMap<mlir::StringAttr, bool> namedCache;
  /// The columns with a planned overlay end, none without `plannedOverlay`.
  int overlayColumns = 0;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIESHIMSHARING_H
