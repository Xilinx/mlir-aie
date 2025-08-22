//===- AIERT.h --------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_AIERT_H
#define AIE_AIERT_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <string>

namespace xilinx::AIE {
struct AIERTControl {

  AIERTControl(const xilinx::AIE::AIETargetModel &tm);
  ~AIERTControl();

  mlir::LogicalResult setIOBackend(bool aieSim, bool xaieDebug);
  mlir::LogicalResult pushToBdQueueAndEnable(mlir::Operation &op, int col,
                                             int row, int chNum,
                                             const DMAChannelDir &channelDir,
                                             int bdId, int repeatCount);
  mlir::LogicalResult configureLocksAndBd(mlir::Block &block, int col, int row);
  mlir::LogicalResult initLocks(DeviceOp &targetOp);
  mlir::LogicalResult initBuffers(DeviceOp &targetOp);
  mlir::LogicalResult configureSwitches(DeviceOp &targetOp);
  mlir::LogicalResult addInitConfig(DeviceOp &targetOp);
  mlir::LogicalResult addCoreEnable(DeviceOp &targetOp);
  mlir::LogicalResult addAieElf(uint8_t col, uint8_t row,
                                const mlir::StringRef elfPath, bool aieSim);
  mlir::LogicalResult addAieElfs(DeviceOp &targetOp,
                                 const mlir::StringRef workDirPath,
                                 bool aieSim);
  void startTransaction();
  void dmaUpdateBdAddr(int col, int row, size_t addr, size_t bdId);
  std::vector<uint8_t> exportSerializedTransaction();

private:
  const AIETargetModel &targetModel;
  struct AIERtImpl;
  std::unique_ptr<AIERtImpl> aiert;
};

} // namespace xilinx::AIE

#endif // AIE_AIERT_H
