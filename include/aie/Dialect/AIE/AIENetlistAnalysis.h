//===- AIENetlistAnalysis.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_LOCKANALYSIS_H
#define MLIR_AIE_LOCKANALYSIS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringSwitch.h"

#include <map>

using namespace mlir;

namespace xilinx {
namespace AIE {

class NetlistAnalysis {
  DeviceOp &device;
  DenseMap<std::pair<int, int>, Operation *> &tiles;
  DenseMap<Operation *, CoreOp> &cores;
  DenseMap<Operation *, MemOp> &mems;
  DenseMap<std::pair<Operation *, int>, LockOp> &locks;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers;
  DenseMap<Operation *, SwitchboxOp> &switchboxes;
  DenseMap<Operation *, SmallVector<Operation *, 4>> bufferUsers;
  DenseMap<Operation *, SmallVector<Operation *, 4>> dma2BufMap;
  DenseMap<std::pair<Operation *, xilinx::AIE::DMAChannel>, Operation *> dmas;
  DenseMap<Operation *, SmallVector<Operation *, 4>> dmaConnections;
  DenseMap<Operation *, SmallVector<Operation *, 4>> dma2ConnectsMap;
  DenseMap<Operation *, Operation *> lockPairs;
  SmallVector<std::pair<Operation *, Operation *>, 4> lockChains;
  DenseMap<Operation *, SmallVector<Operation *, 4>> bufAcqLocks;

public:
  NetlistAnalysis(DeviceOp &d,
                  DenseMap<std::pair<int, int>, Operation *> &tiles,
                  DenseMap<Operation *, CoreOp> &cores,
                  DenseMap<Operation *, MemOp> &mems,
                  DenseMap<std::pair<Operation *, int>, LockOp> &locks,
                  DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers,
                  DenseMap<Operation *, SwitchboxOp> &switchboxes)
      : device(d), tiles(tiles), cores(cores), mems(mems), locks(locks),
        buffers(buffers), switchboxes(switchboxes) {}

  void runAnalysis();

  void collectTiles(DenseMap<std::pair<int, int>, Operation *> &tiles);
  void collectCores(DenseMap<Operation *, CoreOp> &cores);
  void collectMems(DenseMap<Operation *, MemOp> &mems);
  void collectLocks(DenseMap<std::pair<Operation *, int>, LockOp> &locks);
  void collectBuffers(DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers);
  void collectSwitchboxes(DenseMap<Operation *, SwitchboxOp> &switchboxes);

  auto getBufferUsers() const { return bufferUsers; }

  auto getDMA2BufMap() const { return dma2BufMap; }

  auto getDMAs() const { return dmas; }

  auto getDMAConnections() const { return dmaConnections; }

  auto getLockPairs() const { return lockPairs; }

  auto getLockChains() const { return lockChains; }

  auto getBufAcqLocks() const { return bufAcqLocks; }

  auto getDma2ConnectsMap() const { return dma2ConnectsMap; }

  std::pair<int, int> getCoord(Operation *Op) const;
  bool isLegalAffinity(Operation *src, Operation *user) const;
  bool validateCoreOrMemRegion(Operation *CoreOrMemOp);
  void collectBufferUsage();
  void collectDMAUsage();
  uint64_t getMemUsageInBytes(Operation *tileOp) const;
  uint64_t getBufferBaseAddress(Operation *bufOp) const;

  SmallVector<Operation *, 4> getNextConnectOps(ConnectOp currentConnect) const;
  SmallVector<Operation *, 4> findDestConnectOps(ConnectOp source,
                                                 WireBundle destBundle) const;
  SmallVector<Operation *, 4> findRoutes(Operation *sourceConnectOp,
                                         Operation *destConnectOp) const;
  void dmaAnalysis();
  void lockAnalysis();
  int getAvailableLockID(Operation *tileOp);

  void print(raw_ostream &os);
};

} // namespace AIE
} // namespace xilinx

#endif
