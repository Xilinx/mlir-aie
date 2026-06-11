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

#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace xilinx::AIE {
struct AIERTControl;

// RAII helper. Constructs while AIERTControl is recording an aie-rt
// transaction, captures the current XAie_TxnInst command count at entry, and
// at scope end attributes every aie-rt command produced during the scope to
// `loc`. The captured ranges are stored on AIERTControl::txnInstrLocs and read
// out by the AIEToConfiguration round-trip so re-emitted aiex.npu.* ops carry
// the source op's MLIR Location instead of the device's fallback location.
class TxnLocBracket {
public:
  TxnLocBracket(AIERTControl &ctl, mlir::Location loc);
  ~TxnLocBracket();
  TxnLocBracket(const TxnLocBracket &) = delete;
  TxnLocBracket &operator=(const TxnLocBracket &) = delete;

private:
  AIERTControl &ctl;
  mlir::Location loc;
  uint32_t startCmds;
};

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

  // Per-aie-rt-command source locations, indexed by command order in the
  // serialized transaction. Populated by TxnLocBracket scopes around each
  // per-source-op block in the AIERT methods. Empty entries fall back to
  // mlir::UnknownLoc.
  const std::vector<mlir::Location> &getTxnInstrLocs() const;

  // Current XAie_TxnInst::NumCmds for the active transaction (zero if no
  // transaction is being recorded). Used by TxnLocBracket.
  uint32_t getCurrentTxnNumCmds() const;

  // Append `loc` to txnInstrLocs at indices [startCmds, endCmds). Used by
  // TxnLocBracket on scope exit.
  void recordTxnLocRange(uint32_t startCmds, uint32_t endCmds,
                         mlir::Location loc);
  mlir::LogicalResult resetPartition();
  mlir::LogicalResult resetDMA(int col, int row, bool on);
  mlir::LogicalResult resetCore(int col, int row);
  mlir::LogicalResult resetCoreUnreset(int col, int row);
  mlir::LogicalResult resetSwitch(int col, int row);
  mlir::LogicalResult resetLock(int col, int row, int lockId);
  mlir::LogicalResult resetSwitchConnection(int col, int row,
                                            WireBundle sourceBundle,
                                            int sourceChannel,
                                            WireBundle destBundle,
                                            int destChannel);
  mlir::LogicalResult resetPerfCounters(int col, int row);

private:
  const AIETargetModel &targetModel;
  struct AIERtImpl;
  std::unique_ptr<AIERtImpl> aiert;
};

} // namespace xilinx::AIE

#endif // AIE_AIERT_H
