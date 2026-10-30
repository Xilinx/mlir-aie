//===- AIEChannelTurns.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIECHANNELTURNS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIECHANNELTURNS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace xilinx::AIE {

/// The order one core moves objects through several pools, read off its
/// program, so that one DMA channel can serve those pools in turn. The core's
/// period is the innermost loop holding every acquire and release on the
/// pools, with the loops of static trip count inside it unrolled and the
/// branches on their counters folded; where a branch depends on that loop's
/// own counter, the next loop out. Its order through them must not depend on
/// data.
class CoreTurns {
public:
  /// Reads the order off the program of the core working through `ends`, all
  /// draining (the channel fills their pools) or all filling (the channel
  /// drains them). Fails, saying why in `reason`, when the order is not
  /// static.
  mlir::LogicalResult read(llvm::ArrayRef<ObjectFifoCoreEndpointOp> ends,
                           std::string &reason);
  /// The same, before split: `names` are objectFIFOs `core` acquires from
  /// (`coreDrains`) or into.
  mlir::LogicalResult read(CoreOp core, llvm::ArrayRef<mlir::StringAttr> names,
                           bool coreDrains, std::string &reason);

  /// Indices into `ends`, one per object the channel moves in a period of the
  /// core's steady state.
  llvm::ArrayRef<unsigned> turns() const { return order; }

  /// The loop `read` took as the core's period, once it found one.
  mlir::Operation *periodLoop() const { return loop; }

  /// Whether `read` found a steady state in which the period moves none of
  /// the objects: the core keeps what it holds and never takes or hands over
  /// another. `read` fails then, since there are no turns to take.
  bool movesNone() const { return idle; }

  /// The first acquire the period runs, with its branches folded; null when
  /// it runs none.
  mlir::Operation *periodAcquire() const;

  /// Whether a channel moving objects in `turns()` order forever, through
  /// pools of `depths`, never has itself and the core both waiting. Settles
  /// `turns()` on the shortest such order cut from the period's. A core that
  /// ends must move a whole number of that order over its run.
  bool keepsUp(llvm::ArrayRef<int64_t> depths);

private:
  bool keepsUp(llvm::ArrayRef<unsigned> order,
               llvm::ArrayRef<int64_t> depths) const;
  mlir::LogicalResult readWholeRun(
      CoreOp core,
      llvm::ArrayRef<std::pair<ObjectFifoReleaseOp, unsigned>> releases,
      std::string &reason);
  bool wholeRunFits(llvm::ArrayRef<int64_t> counts) const;

  struct Event {
    unsigned end;
    int64_t count;
    bool acquire;
    mlir::Operation *op;
  };

  llvm::SmallVector<Event> period;
  llvm::SmallVector<unsigned> order;
  /// Objects of each end a finite core moves over its whole run; empty when
  /// the run never ends or has no release after its loops.
  llvm::SmallVector<int64_t> wholeRun;
  mlir::Operation *loop = nullptr;
  size_t ends = 0;
  bool coreDrains = true;
  bool idle = false;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIECHANNELTURNS_H
