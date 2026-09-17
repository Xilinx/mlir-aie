//===- CtrlPktUtils.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEX_UTILS_CTRLPKTUTILS_H
#define AIE_DIALECT_AIEX_UTILS_CTRLPKTUTILS_H

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

namespace xilinx::AIEX {

// A control packet is a core-enable write iff it targets the compute-tile
// CORE_CONTROL register (tile-local offset 0x32000) with the Enable bit (bit0)
// set. This discriminates enable (data=0x1) from disable (0x0) and the
// reset-pulse (0x2), both of which stay in the config region. The row>=2 guard
// is insurance against a memtile/shim register aliasing this local offset.
//
// This is the config/enable boundary predicate used by aie-ctrl-packet-to-dma
// to keep the core-enables in a trailing phase behind the join barrier.
inline bool isCoreEnableControlPacket(NpuControlPacketOp op) {
  auto data = op.getData();
  if (!data || data->empty())
    return false;
  uint32_t localOffset = static_cast<uint32_t>(op.getAddress()) & 0xFFFFFu;
  int row = op.getRowFromAddr();
  return localOffset == 0x32000u && row >= 2 && ((*data)[0] & 0x1u) != 0;
}

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_CTRLPKTUTILS_H
