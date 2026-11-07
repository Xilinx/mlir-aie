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

// A packet of a run, with its group and the aiex.control_packet_barrier ops
// before it in the run.
struct GroupedPacket {
  NpuControlPacketOp packet;
  ControlPacketGroupOp group;
  unsigned phase = 0;
};

using GroupRun = llvm::SmallVector<GroupedPacket>;

// Where a run ends besides at an op other than a group or a barrier.
struct RunSplit {
  // Where the group kind or device changes.
  bool byKindAndDevice = true;
  // At a barrier, rather than going on in the next phase.
  bool atBarriers = true;
};

// Whether two groups hold packets of one kind for one device.
bool sameKindAndDevice(ControlPacketGroupOp a, ControlPacketGroupOp b);

// The runs of adjacent groups in `block`, each in program order.
llvm::SmallVector<GroupRun> groupRuns(mlir::Block &block, RunSplit split = {});

// The runs of every block of `seq` outside the groups.
llvm::SmallVector<GroupRun> groupRuns(AIE::RuntimeSequenceOp seq,
                                      RunSplit split = {});

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETRUNS_H
