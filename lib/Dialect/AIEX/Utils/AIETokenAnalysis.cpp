//===- AIETokenAnalysis.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/AIETokenAnalysis.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

static void tokenMapPushBack(
    std::vector<std::pair<llvm::StringRef, std::vector<mlir::Operation *>>>
        &map,
    StringRef tokenName, Operation *op) {
  for (auto &pair : map) {
    if (pair.first == tokenName) {
      pair.second.push_back(op);
      return;
    }
  }
  map.push_back({tokenName, {op}});
}

void xilinx::AIEX::TokenAnalysis::runAnalysis() {

  // Collecting token symbols
  for (auto op : device.getOps<TokenOp>()) {
    StringRef tokenName =
        op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
            .getValue();
    int value = op.getTokenValue();
    tokenSymbols[tokenName] = value;
  }

  // collect all the UseTokenOps and MemcpyOps
  std::map<StringRef, SmallVector<UseTokenOp, 4>> visitors;
  device.getBodyRegion().walk([&](Operation *Op) {
    if (auto op = dyn_cast<UseTokenOp>(Op)) {
      StringRef tokenName = op.getTokenName();
      assert(tokenSymbols.find(tokenName) != tokenSymbols.end() &&
             "Token not found!");
      if (op.acquire()) {
        tokenMapPushBack(tokenAcqMap, tokenName, op.getOperation());
        visitors[tokenName].push_back(op);
      } else {
        tokenMapPushBack(tokenRelMap, tokenName, op.getOperation());
        if (!visitors[tokenName].empty()) {
          Operation *Op = visitors[tokenName].pop_back_val();
          tokenPairs.push_back({Op, op.getOperation()});
        }
      }
    } else if (auto op = dyn_cast<MemcpyOp>(Op)) {
      StringRef tokenName = op.getTokenName();
      assert(tokenSymbols.find(tokenName) != tokenSymbols.end() &&
             "Token not found!");
      Operation *Op = op.getOperation();
      tokenMapPushBack(tokenAcqMap, tokenName, op.getOperation());
      tokenMapPushBack(tokenRelMap, tokenName, op.getOperation());
      tokenPairs.push_back({Op, Op});
    }
  });

  // sanity check: ensure that acquiring a token is followed by releasing a
  // token
  for (const auto &map : tokenAcqMap) {
    StringRef tokenName = map.first;
    for (auto Op : map.second) {
      [[maybe_unused]] bool isReleased = false;
      if (auto op = dyn_cast<MemcpyOp>(Op))
        isReleased = true;

      for (auto pair : tokenPairs) {
        if (UseTokenOp aop = dyn_cast<UseTokenOp>(pair.first)) {
          if (tokenName == aop.getTokenName() && Op == aop.getOperation()) {
            isReleased = true;
            break;
          }
        }
      }

      assert(isReleased && "No release found for acquire!"
                           "This might potentially lead to deadlock");
    }
  }

  int releaseValue = 0;
  int acquireValue = 0;
  // Look for a pair of UseTokenOps (or UseTokenOp and MemcpyOp) such that one
  // releases and one acquires the same token + value. They form a chain of
  // releasing and acquiring a token. From the chains of tokens collected, we
  // can infer the dependency of the parentOps
  for (const auto &map : tokenRelMap) {
    StringRef tokenName = map.first;
    auto tokenRels = map.second;
    auto tokenAcqs = [&]() {
      for (auto &pair : tokenAcqMap)
        if (pair.first == tokenName)
          return pair.second;
      return std::vector<Operation *>();
    }();
    for (auto ROp : tokenRels) {

      if (auto op = dyn_cast<UseTokenOp>(ROp))
        releaseValue = op.getTokenValue();
      else if (auto op = dyn_cast<MemcpyOp>(ROp))
        releaseValue = op.getReleaseTokenValue();

      for (auto AOp : tokenAcqs) {

        if (auto op = dyn_cast<UseTokenOp>(AOp))
          acquireValue = op.getTokenValue();
        else if (auto op = dyn_cast<MemcpyOp>(AOp))
          acquireValue = op.getAcquireTokenValue();
        else
          continue;

        // Release and Acquire form a chain if they set/get the token with the
        // same value
        if (releaseValue != acquireValue)
          continue;

        tokenChains.push_back({ROp, AOp});
      }
    }
  }

  for (auto tile : device.getOps<TileOp>()) {
    int colIndex = tile.colIndex();
    int rowIndex = tile.rowIndex();
    tiles[{colIndex, rowIndex}] = tile;
  }
}

Operation *xilinx::AIEX::TokenAnalysis::getTokenUserOp(Operation *Op) {

  if (UseTokenOp op = dyn_cast<UseTokenOp>(Op)) {
    while (Operation *parentOp = op->getParentOp()) {
      if (isa<CoreOp>(parentOp) || isa<MemOp>(parentOp) ||
          isa<ShimDMAOp>(parentOp))
        return parentOp;
    }
  }

  return nullptr;
}

TileID xilinx::AIEX::TokenAnalysis::getCoord(Operation *Op) {
  int colIndex = 0;
  int rowIndex = 0;

  if (CoreOp core = dyn_cast<CoreOp>(Op)) {
    colIndex = core.colIndex();
    rowIndex = core.rowIndex();
  } else if (MemOp mem = dyn_cast<MemOp>(Op)) {
    colIndex = mem.colIndex();
    rowIndex = mem.rowIndex();
  } else if (ShimDMAOp shimDma = dyn_cast<ShimDMAOp>(Op)) {
    colIndex = shimDma.colIndex();
    rowIndex = shimDma.rowIndex();
  }

  return {colIndex, rowIndex};
}

Operation *xilinx::AIEX::TokenAnalysis::getShareableTileOp(Operation *Op1,
                                                           Operation *Op2) {
  bool IsOp1Mem = isa<MemOp>(Op1) || isa<ShimDMAOp>(Op1);
  bool IsOp2Mem = isa<MemOp>(Op2) || isa<ShimDMAOp>(Op2);

  assert((!IsOp1Mem || !IsOp2Mem) &&
         "Op1 and Op2 cannot be both Mem operation!");

  TileID coord1 = getCoord(Op1);
  TileID coord2 = getCoord(Op2);

  int col1 = coord1.col;
  int row1 = coord1.row;
  int col2 = coord2.col;
  int row2 = coord2.row;

  const auto &targetModel = xilinx::AIE::getTargetModel(Op1);

  bool IsOp1ShareableMem =
      IsOp1Mem && targetModel.isLegalMemAffinity(col2, row2, col1, row1);
  bool IsOp2ShareableMem =
      IsOp2Mem && targetModel.isLegalMemAffinity(col1, row1, col2, row2);

  if (IsOp1ShareableMem)
    return tiles[coord1];
  if (IsOp2ShareableMem)
    return tiles[coord2];

  // both Op1 and Op2 are core ops
  if (!IsOp1Mem && !IsOp2Mem) {
    bool IsS = targetModel.isSouth(col1, row1, col2, row2);
    bool IsW = targetModel.isWest(col1, row1, col2, row2);
    bool IsN = targetModel.isNorth(col1, row1, col2, row2);
    bool IsE = targetModel.isEast(col1, row1, col2, row2);
    bool IsInternal = targetModel.isInternal(col1, row1, col2, row2);
    bool IsEvenRow = ((row1 % 2) == 0);

    // FIXME: This logic appears AIE1 specific.
    if (IsS || IsN || (IsW && !IsEvenRow) || (IsE && IsEvenRow))
      return tiles[coord2];
    if ((IsW && IsEvenRow) || (IsE && !IsEvenRow) || IsInternal)
      return tiles[coord1];
  }

  return nullptr;
}

void xilinx::AIEX::TokenAnalysis::print(raw_ostream &os) {
  os << "\n=====tokenPairs: \n";
  for (auto pair : tokenPairs) {
    Operation *acquire = pair.first;
    Operation *release = pair.second;
    acquire->print(os);
    os << " -> ";
    release->print(os);
    os << "\n";
  }

  os << "\n=====tokenChains: \n";
  for (auto pair : tokenChains) {
    Operation *release = pair.first;
    Operation *acquire = pair.second;
    release->print(os);
    os << " -> ";
    acquire->print(os);
    os << "\n";
  }
}
