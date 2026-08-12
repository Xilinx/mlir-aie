//===- AIERT.h --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// `loc`. exportSerializedTransaction() projects the captured ranges onto the
// serialized transaction's operations, and the AIEToConfiguration round-trip
// reads them back out so re-emitted aiex.npu.* ops carry the source op's MLIR
// Location instead of the device's fallback location.
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
                                             int bdId, int repeatCount,
                                             uint32_t padValue = 0);
  mlir::LogicalResult configureLocksAndBd(mlir::Block &block, int col, int row);
  mlir::LogicalResult initLocks(DeviceOp &targetOp);
  mlir::LogicalResult initBuffers(DeviceOp &targetOp);
  mlir::LogicalResult configureSwitches(DeviceOp &targetOp,
                                        bool skipCtrlPktOverlay = false);
  mlir::LogicalResult addInitConfig(DeviceOp &targetOp,
                                    bool skipCtrlPktOverlay = false);
  mlir::LogicalResult addCoreEnable(DeviceOp &targetOp);
  mlir::LogicalResult addAieElf(uint8_t col, uint8_t row,
                                const mlir::StringRef elfPath, bool aieSim);
  mlir::LogicalResult addAieElfs(DeviceOp &targetOp,
                                 const mlir::StringRef workDirPath,
                                 bool aieSim);
  void startTransaction();
  void dmaUpdateBdAddr(int col, int row, size_t addr, size_t bdId);
  std::vector<uint8_t> exportSerializedTransaction();

  // Source locations for the operations in the transaction binary returned by
  // the last exportSerializedTransaction() call, in binary order. Derived
  // there from the TxnLocBracket scopes around each per-source-op block in the
  // AIERT methods; empty if nothing was bracketed. Entries the brackets did
  // not cover are mlir::UnknownLoc.
  const std::vector<mlir::Location> &getTxnOpLocs() const;

  // Current XAie_TxnInst::NumCmds for the active transaction (zero if no
  // transaction is being recorded). Used by TxnLocBracket.
  uint32_t getCurrentTxnNumCmds() const;

  // Append `loc` to the per-command locations at indices [startCmds, endCmds).
  // Used by TxnLocBracket on scope exit.
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
