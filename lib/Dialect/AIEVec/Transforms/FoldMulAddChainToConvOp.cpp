//===--FoldMulAddChainToConvOp.h - Fold Mul Add Chain To AIEVec Conv Op --===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This is the implementation of the folding pass from mul add chain
// to AIEVec convolution operations, compatible with the AIE-ML architecture.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <tuple>

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

typedef std::tuple<int8_t, aievec::UPDOp, arith::MulIOp> MulDefTupleTy;
using MulDefTupleVecTy = SmallVector<MulDefTupleTy, 8>;
using MulDefMapTy = DenseMap<Value, MulDefTupleVecTy>;

// If only one of the operands of given add is an add, return that operand's def
// op; otherwise return null.
arith::AddIOp getDefAddOp(arith::AddIOp addOp) {
  auto defLhs = dyn_cast<arith::AddIOp>(addOp->getOperand(0).getDefiningOp());
  auto defRhs = dyn_cast<arith::AddIOp>(addOp->getOperand(1).getDefiningOp());
  if ((!defLhs && !defRhs) || (defLhs && defRhs)) {
    return nullptr;
  }
  return defLhs ? defLhs : defRhs;
}

// Return true if one of the operands of given mul op is a broadcast of a upd op
// and another operand of the mul op is a upd op. In this case, argument book
// keeps arguments. Otherwise, return false and leave book keeping unchanged.
bool checkChainPattern(arith::MulIOp mulOp, MulDefMapTy &macChainMap,
                       SmallVectorImpl<Value> &bcastOpSourceVec) {
  aievec::BroadcastOp bcastOp = nullptr;
  aievec::UPDOp updOp = nullptr;

  if (isa<aievec::BroadcastOp>(mulOp.getOperand(0).getDefiningOp())) {
    bcastOp = cast<aievec::BroadcastOp>(mulOp->getOperand(0).getDefiningOp());
    if (!isa<aievec::UPDOp>(mulOp->getOperand(1).getDefiningOp())) {
      return false;
    }
    updOp = cast<aievec::UPDOp>(mulOp->getOperand(1).getDefiningOp());
  } else if (isa<aievec::BroadcastOp>(mulOp.getOperand(1).getDefiningOp())) {
    bcastOp = cast<aievec::BroadcastOp>(mulOp->getOperand(1).getDefiningOp());
    if (!isa<aievec::UPDOp>(mulOp->getOperand(0).getDefiningOp())) {
      return false;
    }
    updOp = cast<aievec::UPDOp>(mulOp->getOperand(0).getDefiningOp());
  } else {
    return false;
  }

  if (!isa<aievec::UPDOp>(bcastOp.getSource().getDefiningOp())) {
    return false;
  }

  if (!macChainMap.count(bcastOp.getSource())) {
    bcastOpSourceVec.push_back(bcastOp.getSource());
    MulDefTupleVecTy tupleVec;
    tupleVec.push_back(std::make_tuple(bcastOp.getIdx(), updOp, mulOp));
    macChainMap.insert(std::make_pair(bcastOp.getSource(), tupleVec));
  } else {
    macChainMap[bcastOp.getSource()].push_back(
        std::make_tuple(bcastOp.getIdx(), updOp, mulOp));
  }
  return true;
}

// The defs of mul ops consist of an upd op and a broadcast op.
// The chain map looks like below:
// | BroadcastOp source | vector<tuple<broadcastOp idx, UPDOp, MulIOp>> |
// The mul add op chain can be grouped by broadcast op's source.
// For each group, broadcastOp idx can be sorted to find the start of the
// memrefs used by broadcast op and upd op.
void buildChainMap(arith::AddIOp curAddOp, bool &hasMulConv, Value &acc,
                   MulDefMapTy &macChainMap,
                   SmallVectorImpl<Value> &bcastOpSourceVec) {
  while (true) {
    auto defLhs =
        dyn_cast<arith::MulIOp>(curAddOp->getOperand(0).getDefiningOp());
    auto defRhs =
        dyn_cast<arith::MulIOp>(curAddOp->getOperand(1).getDefiningOp());

    if (!defLhs && !defRhs) {
      break;
    }
    // If both ops of add op are mul ops, this will reach the top of the
    // chain. Check the legality for both mul op and insert them to the chain
    // map.
    else if (defLhs && defRhs) {
      if (!checkChainPattern(defLhs, macChainMap, bcastOpSourceVec) ||
          !checkChainPattern(defRhs, macChainMap, bcastOpSourceVec)) {
        break;
      }
      hasMulConv = true;
    } else {
      arith::MulIOp curMulOp = defLhs ? defLhs : defRhs;
      if (!checkChainPattern(curMulOp, macChainMap, bcastOpSourceVec)) {
        break;
      }
      acc = defLhs ? curAddOp->getOperand(1) : curAddOp->getOperand(0);
    }

    // Get the def add op the curOp operands
    arith::AddIOp defAddOp = getDefAddOp(curAddOp);

    // The user/consumer user operation must be an add op, belonging to
    // the same basic block as curOp.
    if (!defAddOp || !defAddOp->hasOneUse() ||
        curAddOp->getBlock() != defAddOp->getBlock()) {
      break;
    }
    curAddOp = defAddOp;
  }
}

void refreshFusedGroups(
    MulDefTupleTy defTuple, arith::MulIOp nextMulOp,
    SmallVector<arith::MulIOp, 8> &fusedOps,
    SmallVectorImpl<SmallVector<arith::MulIOp, 8>> &groupFusedOps,
    int8_t &curIdx, aievec::UPDOp &curUpdOp, arith::MulIOp &curMulOp) {
  groupFusedOps.push_back(fusedOps);
  fusedOps.clear();
  fusedOps.push_back(nextMulOp);
  std::tie(curIdx, curUpdOp, curMulOp) = defTuple;
}

// Check whether mul add chain is valid for the transformation and classify the
// fused ops into different groups with valid constant memref distances.
bool collectFusedOps(
    unsigned maxGroupSize, unsigned &dupFactor,
    SmallVectorImpl<Value> &bcastOpSourceVec,
    SmallVectorImpl<SmallVector<arith::MulIOp, 8>> &groupFusedOps,
    MulDefMapTy &macChainMap) {
  int xDist = -1, zDist = -1;
  for (auto item : bcastOpSourceVec) {
    auto macChain = macChainMap[item];
    std::sort(macChain.begin(), macChain.end());
    int8_t curIdx = 0;
    aievec::UPDOp curUpdOp = nullptr;
    arith::MulIOp curMulOp = nullptr;
    std::tie(curIdx, curUpdOp, curMulOp) = *macChain.begin();
    SmallVector<int32_t, 2> dists;
    SmallVector<arith::MulIOp, 8> fusedOps;
    fusedOps.push_back(curMulOp);

    for (auto it = std::next(macChain.begin()); it != macChain.end(); ++it) {
      int8_t nextIdx = 0;
      aievec::UPDOp nextUpdOp = nullptr;
      arith::MulIOp nextMulOp = nullptr;
      MulDefTupleTy defTuple = *it;
      std::tie(nextIdx, nextUpdOp, nextMulOp) = defTuple;

      int32_t dist = nextIdx - curIdx;

      // Target AIE-ML intrinsic mac_conv_32x8 for v32int8 type and
      // mac_conv_16x4 for v16int16 type. Thus, the distance of broadcast op
      // source between two mul add ops cannot be larger than 32/8 or 16/4,
      // which is 4. If dist is larger than 1, we need to shuffle the load to
      // get the elements with the interval of dist.
      if (dist > 4) {
        if (fusedOps.size() < 2) {
          return false;
        }
        refreshFusedGroups(defTuple, nextMulOp, fusedOps, groupFusedOps, curIdx,
                           curUpdOp, curMulOp);
        continue;
      }

      dists.push_back(dist);
      if (curUpdOp.getSource() != nextUpdOp.getSource()) {
        if (fusedOps.size() < 2) {
          return false;
        }
        refreshFusedGroups(defTuple, nextMulOp, fusedOps, groupFusedOps, curIdx,
                           curUpdOp, curMulOp);
        continue;
      }

      MemRefType curMemRefType =
          cast<MemRefType>(curUpdOp.getSource().getType());
      MemRefType nextMemRefType =
          cast<MemRefType>(nextUpdOp.getSource().getType());

      ArrayRef<int64_t> curSizes = curMemRefType.getShape();
      ArrayRef<int64_t> nextSizes = nextMemRefType.getShape();
      if (curSizes.size() != nextSizes.size()) {
        if (fusedOps.size() < 2) {
          return false;
        }
        refreshFusedGroups(defTuple, nextMulOp, fusedOps, groupFusedOps, curIdx,
                           curUpdOp, curMulOp);
        continue;
      }

      AffineExpr curLinearAccess =
          constructLinearizedAffineExprForUPDOp(curUpdOp);
      AffineExpr nextLinearAccess =
          constructLinearizedAffineExprForUPDOp(nextUpdOp);
      if (!curLinearAccess || !nextLinearAccess) {
        if (fusedOps.size() < 2) {
          return false;
        }
        refreshFusedGroups(defTuple, nextMulOp, fusedOps, groupFusedOps, curIdx,
                           curUpdOp, curMulOp);
        continue;
      }

      AffineExpr curBase, nextBase;
      int32_t curOffset, nextOffset;

      // Get the base and offset from linear access expr
      std::tie(curBase, curOffset) = extractBaseAndOffset(curLinearAccess);
      std::tie(nextBase, nextOffset) = extractBaseAndOffset(nextLinearAccess);
      if (curBase != nextBase) {
        if (fusedOps.size() < 2) {
          return false;
        }
        refreshFusedGroups(defTuple, nextMulOp, fusedOps, groupFusedOps, curIdx,
                           curUpdOp, curMulOp);
        continue;
      }

      dist = nextOffset - curOffset;
      if (dist != 1) {
        if (fusedOps.size() < 2) {
          return false;
        }
        refreshFusedGroups(defTuple, nextMulOp, fusedOps, groupFusedOps, curIdx,
                           curUpdOp, curMulOp);
        continue;
      }
      dists.push_back(dist);

      if ((xDist != -1 && xDist != dists[0]) ||
          (zDist != -1 && zDist != dists[1])) {
        if (fusedOps.size() < 2) {
          return false;
        }
        refreshFusedGroups(defTuple, nextMulOp, fusedOps, groupFusedOps, curIdx,
                           curUpdOp, curMulOp);
        continue;
      }

      xDist = dists[0];
      zDist = dists[1];
      dupFactor = dists[0];

      fusedOps.push_back(nextMulOp);
      std::tie(curIdx, curUpdOp, curMulOp) = defTuple;

      if (fusedOps.size() > maxGroupSize) {
        fusedOps.pop_back();
        refreshFusedGroups(defTuple, nextMulOp, fusedOps, groupFusedOps, curIdx,
                           curUpdOp, curMulOp);
        continue;
      }
    }
    groupFusedOps.push_back(fusedOps);
  }
  return true;
}
