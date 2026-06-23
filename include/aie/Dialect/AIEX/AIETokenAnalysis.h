//===- AIETokenAnalysis.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2019, Xilinx, Inc. All rights reserved.
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_TOKENANALYSIS_H
#define AIEX_TOKENANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace xilinx::AIE {
class DeviceOp;
struct TileID;
} // namespace xilinx::AIE

namespace xilinx::AIEX {

class TokenAnalysis {
  AIE::DeviceOp &device;
  llvm::DenseMap<llvm::StringRef, int> tokenSymbols;
  std::vector<std::pair<llvm::StringRef, std::vector<mlir::Operation *>>>
      tokenAcqMap;
  std::vector<std::pair<llvm::StringRef, std::vector<mlir::Operation *>>>
      tokenRelMap;
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>, 4>
      tokenChains;
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>, 4>
      tokenPairs;
  llvm::DenseMap<xilinx::AIE::TileID, mlir::Operation *> tiles;

public:
  TokenAnalysis(AIE::DeviceOp &d) : device(d) {}

  void runAnalysis();

  auto getTokenSymbols() const { return tokenSymbols; }

  auto getTokenChains() const { return tokenChains; }

  auto getTokenPairs() const { return tokenPairs; }

  auto getTiles() const { return tiles; }

  // CoreOp or MemOp
  mlir::Operation *getTokenUserOp(mlir::Operation *Op);
  mlir::Operation *getShareableTileOp(mlir::Operation *Op1,
                                      mlir::Operation *Op2);
  xilinx::AIE::TileID getCoord(mlir::Operation *Op);

  void print(llvm::raw_ostream &os);
};

} // namespace xilinx::AIEX

#endif
