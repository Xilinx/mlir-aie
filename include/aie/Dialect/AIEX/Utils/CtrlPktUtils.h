//===- CtrlPktUtils.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEX_UTILS_CTRLPKTUTILS_H
#define AIE_DIALECT_AIEX_UTILS_CTRLPKTUTILS_H

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

namespace xilinx::AIEX {

// A control packet is a core-enable write iff it targets a compute tile's
// Core_Control register with the Enable bit (bit 0) set. This discriminates
// enable (data=0x1) from disable (0x0) and the reset-pulse (0x2), both of which
// stay in the config region. The Core_Control offset and the core-tile check
// come from the target model's register database rather than being restated
// here, so this stays correct across target models (mirrors AIELowerCoreReset's
// Core_Control lookup); lookupRegister returns null for a mem/shim tile, so it
// doubles as the "core tile only" guard.
//
// This is the config/enable boundary predicate used by aie-ctrl-packet-to-dma
// to keep the core-enables in a trailing phase behind the join barrier.
inline bool isCoreEnableControlPacket(NpuControlPacketOp op) {
  auto data = op.getData();
  if (!data || data->empty())
    return false;
  const xilinx::AIE::AIETargetModel &tm = xilinx::AIE::getTargetModel(op);
  int col = static_cast<int>(op.getColumnFromAddr());
  int row = static_cast<int>(op.getRowFromAddr());
  if (!tm.isCoreTile(col, row))
    return false;
  const xilinx::AIE::RegisterInfo *ctrlReg =
      tm.lookupRegister("Core_Control", xilinx::AIE::TileID{col, row});
  if (!ctrlReg)
    return false;
  uint32_t localOffset = static_cast<uint32_t>(op.getAddress()) & 0xFFFFFu;
  if (localOffset != (static_cast<uint32_t>(ctrlReg->offset) & 0xFFFFFu))
    return false;
  return ((*data)[0] & 0x1u) != 0;
}

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_CTRLPKTUTILS_H
