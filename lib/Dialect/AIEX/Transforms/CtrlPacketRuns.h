//===- CtrlPacketRuns.h -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TRANSFORMS_CTRLPACKETRUNS_H
#define AIEX_TRANSFORMS_CTRLPACKETRUNS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "llvm/ADT/SmallVector.h"

namespace xilinx::AIEX {

struct PacketInterval {
  llvm::SmallVector<NpuControlPacketOp> packets;
  mlir::FlatSymbolRefAttr device;
  unsigned delivery = 0;
  bool ordinary = false;
  bool independent = false;
};

llvm::SmallVector<PacketInterval> packetIntervals(mlir::Block &block);
mlir::ArrayAttr controlPacketMulticasts(mlir::ModuleOp module,
                                        mlir::FlatSymbolRefAttr device);

} // namespace xilinx::AIEX

#endif // AIEX_TRANSFORMS_CTRLPACKETRUNS_H
