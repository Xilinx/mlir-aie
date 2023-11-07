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
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "llvm/ADT/StringSwitch.h"

#include <map>

namespace xilinx::AIE {

class NetlistAnalysis {
  DeviceOp &device;
  llvm::DenseMap<TileID, mlir::Operation *> &tiles;
  llvm::DenseMap<mlir::Operation *, CoreOp> &cores;
  llvm::DenseMap<mlir::Operation *, MemOp> &mems;
  llvm::DenseMap<std::pair<mlir::Operation *, int>, LockOp> &locks;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<BufferOp, 4>> &buffers;
  llvm::DenseMap<mlir::Operation *, SwitchboxOp> &switchboxes;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      bufferUsers;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      dma2BufMap;
  llvm::DenseMap<std::pair<mlir::Operation *, xilinx::AIE::DMAChannel>,
                 mlir::Operation *>
      dmas;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      dmaConnections;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      dma2ConnectsMap;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> lockPairs;
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>, 4>
      lockChains;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      bufAcqLocks;

public:
  NetlistAnalysis(
      DeviceOp &d, llvm::DenseMap<TileID, mlir::Operation *> &tiles,
      llvm::DenseMap<mlir::Operation *, CoreOp> &cores,
      llvm::DenseMap<mlir::Operation *, MemOp> &mems,
      llvm::DenseMap<std::pair<mlir::Operation *, int>, LockOp> &locks,
      llvm::DenseMap<mlir::Operation *, llvm::SmallVector<BufferOp, 4>>
          &buffers,
      llvm::DenseMap<mlir::Operation *, SwitchboxOp> &switchboxes)
      : device(d), tiles(tiles), cores(cores), mems(mems), locks(locks),
        buffers(buffers), switchboxes(switchboxes) {}

  void collectTiles(llvm::DenseMap<TileID, mlir::Operation *> &tiles);
  void collectCores(llvm::DenseMap<mlir::Operation *, CoreOp> &cores);
  void collectBuffers(llvm::DenseMap<mlir::Operation *,
                                     llvm::SmallVector<BufferOp, 4>> &buffers);

  auto getBufferUsers() const { return bufferUsers; }

  auto getDMA2BufMap() const { return dma2BufMap; }

  auto getDMAs() const { return dmas; }

  void collectDMAUsage();
  uint64_t getBufferBaseAddress(mlir::Operation *bufOp) const;

  llvm::SmallVector<mlir::Operation *, 4>
  getNextConnectOps(ConnectOp currentConnect) const;
  llvm::SmallVector<mlir::Operation *, 4>
  findDestConnectOps(ConnectOp source, WireBundle destBundle) const;
  void dmaAnalysis();
};

} // namespace xilinx::AIE

#endif
