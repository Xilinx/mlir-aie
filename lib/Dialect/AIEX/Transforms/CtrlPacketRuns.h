//===- CtrlPacketRuns.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The runs of adjacent aiex.control_packet_group ops that the control packet
// delivery passes rewrite or check together.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TRANSFORMS_CTRLPACKETRUNS_H
#define AIEX_TRANSFORMS_CTRLPACKETRUNS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/SmallVector.h"

namespace xilinx::AIEX {

// A packet of a run, with its group.
struct GroupedPacket {
  NpuControlPacketOp packet;
  ControlPacketGroupOp group;
};

using GroupRun = llvm::SmallVector<GroupedPacket>;

// Whether two groups hold packets of one kind for one device.
bool sameKindAndDevice(ControlPacketGroupOp a, ControlPacketGroupOp b);

// The runs of adjacent groups of one kind and device in `block`, each in
// program order. Any other op ends a run.
llvm::SmallVector<GroupRun> groupRuns(mlir::Block &block);

// The runs of every block of `seq` outside the groups.
llvm::SmallVector<GroupRun> groupRuns(AIE::RuntimeSequenceOp seq);

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETRUNS_H
