//===- AIETokenAnalysis.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_LOCKANALYSIS_H
#define MLIR_AIE_LOCKANALYSIS_H

#include "aie/AIEDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"

using llvm::SmallSet;
using namespace mlir;

namespace xilinx {
namespace AIE {

class TokenAnalysis {
  ModuleOp &module;
  // tokenSymbols[name] == initialValue
  DenseMap<StringRef, int> tokenSymbols;
  // tokenValues[name] == {value, ...}
  DenseMap<StringRef, SmallSet<int, 4>> tokenValues;
  // tokenAcqMap[name] == {UseTokenOp/MemcpyOp, ...} (Acquire)
  DenseMap<StringRef, SmallVector<Operation *>> tokenAcqMap;
  // tokenRelMap[name] == {UseTokenOp/MemcpyOp, ...} (Release)
  DenseMap<StringRef, SmallVector<Operation *>> tokenRelMap;
  // tokenPairs == {(Acquire Op, Release Op), ...}
  SmallVector<std::pair<Operation *, Operation *>> tokenPairs;
  // tiles[(col, rol)] == TileOp
  DenseMap<std::pair<int, int>, TileOp> tiles;
  // tileLocks[tile][lockId] == LockOp
  DenseMap<TileOp, DenseMap<int, LockOp>> tileLocks;

public:
  TokenAnalysis(ModuleOp &m) : module(m) {}

  void runAnalysis();

  auto &getTokenSymbols() { return tokenSymbols; }
  auto &getTokenValues() { return tokenValues; }
  auto &getTokenAcqMap() { return tokenAcqMap; }
  auto &getTokenRelMap() { return tokenRelMap; }
  auto &getTokenPairs() { return tokenPairs; }
  auto &getTiles() { return tiles; }
  auto &getTileLocks() { return tileLocks; }

  // CoreOp, MemOp or ShimDMAOp
  Operation *getTokenUserOp(Operation *Op);
  SmallSet<TileOp, 4> getAccessibleTileOp(Operation *Op);
  std::pair<int, int> getCoord(Operation *Op);
  std::pair<StringRef, int> getTokenUseNameValue(Operation *Op, bool acquire);

  void print(raw_ostream &os);
};

} // namespace AIE
} // namespace xilinx

#endif
