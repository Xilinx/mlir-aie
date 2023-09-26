//===- AIETokenAnalysis.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TOKENANALYSIS_H
#define AIEX_TOKENANALYSIS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
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
namespace AIEX {

class TokenAnalysis {
  AIE::DeviceOp &device;
  DenseMap<StringRef, int> tokenSymbols;
  DenseMap<StringRef, SmallVector<Operation *, 4>> tokenAcqMap;
  DenseMap<StringRef, SmallVector<Operation *, 4>> tokenRelMap;
  SmallVector<std::pair<Operation *, Operation *>, 4> tokenChains;
  SmallVector<std::pair<Operation *, Operation *>, 4> tokenPairs;
  DenseMap<std::pair<int, int>, Operation *> tiles;

public:
  TokenAnalysis(AIE::DeviceOp &d) : device(d) {}

  void runAnalysis();

  auto getTokenSymbols() const { return tokenSymbols; }

  auto getTokenAcqMap() const { return tokenAcqMap; }

  auto getTokenRelMap() const { return tokenRelMap; }

  auto getTokenChains() const { return tokenChains; }

  auto getTokenPairs() const { return tokenPairs; }

  auto getTiles() const { return tiles; }

  // CoreOp or MemOp
  Operation *getTokenUserOp(Operation *Op);
  Operation *getShareableTileOp(Operation *Op1, Operation *Op2);
  std::pair<int, int> getCoord(Operation *Op);

  void print(raw_ostream &os);
};

} // namespace AIEX
} // namespace xilinx

#endif
