// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
//===- AIEDialect.h - Dialect definition for the AIE IR ----------------===//
//
// Copyright 2019 Xilinx
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_AIE_LOCKANALYSIS_H
#define MLIR_AIE_LOCKANALYSIS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/StringSwitch.h"

#include <map>

using namespace mlir;

namespace xilinx {
namespace AIE {

class NetlistAnalysis {
  ModuleOp &module;
  DenseMap<std::pair<int, int>, Operation *> &tiles;
  DenseMap<Operation *, CoreOp> &cores;
  DenseMap<Operation *, MemOp> &mems;
  DenseMap<std::pair<Operation *, int>, LockOp> &locks;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers;
  DenseMap<Operation *, SwitchboxOp> &switchboxes;

public:
  NetlistAnalysis(ModuleOp &m,
    DenseMap<std::pair<int, int>, Operation *> &tiles,
    DenseMap<Operation *, CoreOp> &cores,
    DenseMap<Operation *, MemOp> &mems,
    DenseMap<std::pair<Operation *, int>, LockOp> &locks,
    DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers,
    DenseMap<Operation *, SwitchboxOp> &switchboxes
  ) : module(m), tiles(tiles), cores(cores), mems(mems),
      locks(locks), buffers(buffers), switchboxes(switchboxes) {}

  void runAnalysis();

  void collectTiles(DenseMap<std::pair<int, int>, Operation *> &tiles);
  void collectCores(DenseMap<Operation *, CoreOp> &cores);
  void collectMems(DenseMap<Operation *, MemOp> &mems);
  void collectLocks(DenseMap<std::pair<Operation *, int>, LockOp> &locks);
  void collectBuffers(DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers);
  void collectSwitchboxes(DenseMap<Operation *, SwitchboxOp> &switchboxes);

  std::pair<int, int> getCoord(Operation *Op) const;
  bool isLegalAffinity(Operation *src, Operation *user) const;
  ArrayRef<Operation *> getCompatibleTiles(Operation *buf) const ;
  bool validateCoreOrMemRegion(Operation *CoreOrMemOp) const;
  int getMemUsageInBytes(Operation *tileOp) const;
  int getBufferBaseAddress(Operation *bufOp) const;
  void print(raw_ostream &os);
};

} // AIE
} // xilinx

#endif
