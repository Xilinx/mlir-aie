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
/// pools, with the loops of static trip count inside it unrolled; its order
/// through them must not depend on data.
class CoreTurns {
public:
  /// Reads the order off the program of the core working through `ends`, all
  /// draining (the channel fills their pools) or all filling (the channel
  /// drains them). Fails, saying why in `reason`, when the order is not
  /// static.
  mlir::LogicalResult read(llvm::ArrayRef<ObjectFifoCoreEndpointOp> ends,
                           std::string &reason);

  /// Indices into `ends`, one per object the channel moves in a period of the
  /// core's steady state.
  llvm::ArrayRef<unsigned> turns() const { return order; }

  /// Whether a channel moving objects in `turns()` order forever, through
  /// pools of `depths`, never has itself and the core both waiting.
  bool keepsUp(llvm::ArrayRef<int64_t> depths) const;

private:
  struct Event {
    unsigned end;
    int64_t count;
    bool acquire;
  };

  llvm::SmallVector<Event> period;
  llvm::SmallVector<unsigned> order;
  size_t ends = 0;
  bool coreDrains = true;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIECHANNELTURNS_H
