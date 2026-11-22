//===- CtrlPacketRuns.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The runs of adjacent control packets and their delivery boundaries.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TRANSFORMS_CTRLPACKETRUNS_H
#define AIEX_TRANSFORMS_CTRLPACKETRUNS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/SmallVector.h"

#include <limits>

namespace xilinx::AIEX {

struct ContextualPacket {
  NpuControlPacketOp packet;
  mlir::FlatSymbolRefAttr device;
  unsigned phase = 0;
  unsigned streamEndPhase = std::numeric_limits<unsigned>::max();
  unsigned hostPhase = std::numeric_limits<unsigned>::max();
};

using PacketRun = llvm::SmallVector<ContextualPacket>;

llvm::SmallVector<PacketRun> packetRuns(mlir::Block &block,
                                        bool includePhases = false);
llvm::SmallVector<PacketRun> packetRuns(AIE::RuntimeSequenceOp seq,
                                        bool includePhases = false);
mlir::FailureOr<llvm::SmallVector<PacketRun>>
deliveryAnalysisRuns(AIE::RuntimeSequenceOp sequence);
llvm::SmallVector<unsigned> awaitSteps(const PacketRun &run,
                                       bool parallelShims);

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETRUNS_H
