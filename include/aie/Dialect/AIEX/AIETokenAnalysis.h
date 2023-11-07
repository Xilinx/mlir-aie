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
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

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

namespace xilinx::AIEX {

using namespace xilinx::AIE;

class TokenAnalysis {
  AIE::DeviceOp &device;
  llvm::DenseMap<llvm::StringRef, int> tokenSymbols;
  llvm::DenseMap<llvm::StringRef, llvm::SmallVector<mlir::Operation *, 4>>
      tokenAcqMap;
  llvm::DenseMap<llvm::StringRef, llvm::SmallVector<mlir::Operation *, 4>>
      tokenRelMap;
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>, 4>
      tokenChains;
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>, 4>
      tokenPairs;
  llvm::DenseMap<TileID, mlir::Operation *> tiles;

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
  mlir::Operation *getTokenUserOp(mlir::Operation *Op);
  mlir::Operation *getShareableTileOp(mlir::Operation *Op1,
                                      mlir::Operation *Op2);
  TileID getCoord(mlir::Operation *Op);

  void print(llvm::raw_ostream &os);
};

} // namespace xilinx::AIEX

#endif
