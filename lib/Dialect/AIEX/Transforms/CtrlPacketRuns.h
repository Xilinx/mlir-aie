//===- CtrlPacketRuns.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The runs of adjacent control packet groups, and the order they are sent in.
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

// What a change of group kind or device does to a run.
enum class KindOrDeviceChange { EndsRun, NextPhase };

// Where a run ends besides at an op other than a group or a barrier.
struct RunSplit {
  KindOrDeviceChange kindOrDevice = KindOrDeviceChange::EndsRun;
  // At a barrier, rather than going on in the next phase.
  bool atBarriers = true;
};

// Whether two groups hold packets of one kind for one device.
bool sameKindAndDevice(ControlPacketGroupOp a, ControlPacketGroupOp b);

// The runs of adjacent groups in `block`, each in program order.
// aie-ctrl-packet-to-dma sends them in this order, which the passes before it
// keep and aie-check-ctrl-packet-delivery checks against:
// - Any op other than a group or a barrier ends a run, awaited before it.
// - A barrier, or a change of group kind or device, ends a phase, awaited
//   before the next phase starts.
// - Within a phase, serially, each shim column's transfers are awaited before
//   another column's start; with parallel shims all start before any is
//   awaited.
// - What one shim channel sends keeps its order only along the switch path it
//   shares. Anything else counts as landed only once awaited.
llvm::SmallVector<GroupRun> groupRuns(mlir::Block &block, RunSplit split = {});

// The runs of every block of `seq` outside the groups.
llvm::SmallVector<GroupRun> groupRuns(AIE::RuntimeSequenceOp seq,
                                      RunSplit split = {});

// Each packet's transfer step in `run`, split with
// KindOrDeviceChange::NextPhase and without atBarriers: each step is awaited
// before the next.
llvm::SmallVector<unsigned> awaitSteps(const GroupRun &run, bool parallelShims);

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETRUNS_H
