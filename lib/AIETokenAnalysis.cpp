//===- AIETokenAnalysis.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIETokenAnalysis.h"
#include "aie/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

void xilinx::AIE::TokenAnalysis::runAnalysis() {

  // Collecting token symbols
  for (auto op : module.getOps<TokenOp>()) {
    StringRef tokenName =
        op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
            .getValue();
    int value = op.getTokenValue();
    tokenSymbols[tokenName] = value;
    tokenValues[tokenName].insert(value);
  }

  // Collect all the UseTokenOps and MemcpyOps
  std::map<StringRef, SmallVector<Operation *>> visitors;
  module.getBodyRegion().walk([&](Operation *Op) {
    if (auto op = dyn_cast<UseTokenOp>(Op)) {
      StringRef tokenName = op.tokenName();
      assert(tokenSymbols.find(tokenName) != tokenSymbols.end() &&
             "Token not found!");
      if (op.acquire()) {
        tokenAcqMap[tokenName].push_back(Op);
        visitors[tokenName].push_back(Op);
      } else {
        tokenRelMap[tokenName].push_back(Op);
        if (!visitors[tokenName].empty()) {
          auto previousOp = visitors[tokenName].pop_back_val();
          tokenPairs.push_back(std::make_pair(previousOp, Op));
        }
      }
      tokenValues[tokenName].insert(op.getTokenValue());

    } else if (auto op = dyn_cast<MemcpyOp>(Op)) {
      StringRef tokenName = op.tokenName();
      assert(tokenSymbols.find(tokenName) != tokenSymbols.end() &&
             "Token not found!");
      tokenAcqMap[tokenName].push_back(Op);
      tokenValues[tokenName].insert(op.getAcquireTokenValue());
      tokenRelMap[tokenName].push_back(Op);
      tokenValues[tokenName].insert(op.getReleaseTokenValue());
      tokenPairs.push_back(std::make_pair(Op, Op));
    }
  });

  // sanity check: ensure that acquiring a token is followed by releasing a
  // token
  for (auto map : tokenAcqMap) {
    StringRef tokenName = map.first;
    for (auto Op : map.second) {
      bool isReleased = false;
      if (auto op = dyn_cast<MemcpyOp>(Op))
        isReleased = true;

      for (auto pair : tokenPairs) {
        if (UseTokenOp aop = dyn_cast<UseTokenOp>(pair.first)) {
          if (tokenName == aop.tokenName() && Op == aop.getOperation()) {
            isReleased = true;
            break;
          }
        }
      }

      assert(isReleased && "No release found for acquire!"
                           "This might potentially lead to deadlock");
    }
  }

  // Collecting tiles
  for (auto tile : module.getOps<TileOp>()) {
    int colIndex = tile.colIndex();
    int rowIndex = tile.rowIndex();
    tiles[std::make_pair(colIndex, rowIndex)] = tile;
  }

  // Collecting existing locks
  for (auto lock : module.getOps<LockOp>()) {
    TileOp tile = dyn_cast<TileOp>(lock.tile().getDefiningOp());
    tileLocks[tile][lock.getLockID()] = lock;
  }
}

Operation *xilinx::AIE::TokenAnalysis::getTokenUserOp(Operation *Op) {

  if (UseTokenOp op = dyn_cast<UseTokenOp>(Op)) {
    while (Operation *parentOp = op->getParentOp()) {
      if (isa<CoreOp>(parentOp) || isa<MemOp>(parentOp) ||
          isa<ShimDMAOp>(parentOp))
        return parentOp;
    }
  }

  return nullptr;
}

std::pair<int, int> xilinx::AIE::TokenAnalysis::getCoord(Operation *Op) {
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
  } else {
    assert(false && "unknown usage operation");
  }

  return std::make_pair(colIndex, rowIndex);
}

SmallSet<TileOp, 4>
xilinx::AIE::TokenAnalysis::getAccessibleTileOp(Operation *Op) {
  SmallSet<TileOp, 4> possibleTiles;

  bool IsOpCore = isa<CoreOp>(Op);
  auto coord1 = getCoord(Op);
  int col1 = coord1.first, row1 = coord1.second;

  possibleTiles.insert(tiles[coord1]);

  // Operations in CoreOp could use neighbor locks
  if (IsOpCore) {
    for (auto tile : tiles) {
      int col2 = tile.first.first, row2 = tile.first.second;
      bool IsS = isSouth(col1, row1, col2, row2);
      bool IsW = isWest(col1, row1, col2, row2);
      bool IsN = isNorth(col1, row1, col2, row2);
      bool IsE = isEast(col1, row1, col2, row2);
      bool IsEvenRow = ((row1 % 2) == 0);

      if (IsS || IsN || (IsW && !IsEvenRow) || (IsE && IsEvenRow))
        possibleTiles.insert(tile.second);
    }
  }

  return possibleTiles;
}

std::pair<StringRef, int>
xilinx::AIE::TokenAnalysis::getTokenUseNameValue(Operation *Op, bool acquire) {
  if (auto mop = dyn_cast<MemcpyOp>(Op)) {
    if (acquire) {
      return std::make_pair(mop.tokenName(), mop.getAcquireTokenValue());
    } else {
      return std::make_pair(mop.tokenName(), mop.getReleaseTokenValue());
    }
  } else if (auto utop = dyn_cast<UseTokenOp>(Op)) {
    return std::make_pair(utop.tokenName(), utop.value());
  }
  assert(false && "unknown token use operation.");
}

void xilinx::AIE::TokenAnalysis::print(raw_ostream &os) {
  os << "\n=====tokenPairs: \n";
  for (auto pair : tokenPairs) {
    Operation *acquire = pair.first;
    Operation *release = pair.second;
    acquire->print(os);
    os << " -> ";
    release->print(os);
    os << "\n";
  }
}
