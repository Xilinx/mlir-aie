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
static arith::AddIOp getDefAddOp(arith::AddIOp addOp) {
  auto defLhs = dyn_cast<arith::AddIOp>(addOp->getOperand(0).getDefiningOp());
  auto defRhs = dyn_cast<arith::AddIOp>(addOp->getOperand(1).getDefiningOp());
  if ((!defLhs && !defRhs) || (defLhs && defRhs)) {
    return nullptr;
  }
  return defLhs ? defLhs : defRhs;
}

// The mul add chain pattern is valid if the defining ops of a mul op consist of
// a broadcast op and an upd op.
static bool checkChainPattern(arith::MulIOp mulOp, MulDefMapTy &macChainMap,
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
static void buildChainMap(arith::AddIOp curAddOp, MulDefMapTy &macChainMap,
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
    } else {
      arith::MulIOp curMulOp = defLhs ? defLhs : defRhs;
      if (!checkChainPattern(curMulOp, macChainMap, bcastOpSourceVec)) {
        break;
      }
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

// Check whether mul add chain is valid for the transformation and classify the
// fused ops into different groups with valid constant memref distances.
static bool
collectFusedOps(unsigned groupSize, unsigned &dupFactor,
                SmallVectorImpl<Value> &bcastOpSourceVec,
                SmallVectorImpl<SmallVector<arith::MulIOp, 8>> &groupFusedOps,
                MulDefMapTy &macChainMap) {
  for (auto item : bcastOpSourceVec) {
    auto macChain = macChainMap[item];
    std::sort(macChain.begin(), macChain.end());
    int8_t curIdx = 0;
    aievec::UPDOp curUpdOp = nullptr;
    arith::MulIOp curMulOp = nullptr;
    std::tie(curIdx, curUpdOp, curMulOp) = *macChain.begin();
    int xDist = -1, zDist = -1;
    SmallVector<int32_t, 2> dists;
    SmallVector<arith::MulIOp, 8> fusedOps;
    fusedOps.push_back(curMulOp);

    for (auto it = std::next(macChain.begin()); it != macChain.end(); ++it) {
      int8_t nextIdx = 0;
      aievec::UPDOp nextUpdOp = nullptr;
      arith::MulIOp nextMulOp = nullptr;
      std::tie(nextIdx, nextUpdOp, nextMulOp) = *it;

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
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        std::tie(curIdx, curUpdOp, curMulOp) = *it;
        continue;
      }

      dists.push_back(dist);
      if (curUpdOp.getSource() != nextUpdOp.getSource()) {
        if (fusedOps.size() < 2) {
          return false;
        }
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        std::tie(curIdx, curUpdOp, curMulOp) = *it;
        continue;
      }

      MemRefType curMemRefType =
          curUpdOp.getSource().getType().cast<MemRefType>();
      MemRefType nextMemRefType =
          nextUpdOp.getSource().getType().cast<MemRefType>();

      ArrayRef<int64_t> curSizes = curMemRefType.getShape();
      ArrayRef<int64_t> nextSizes = nextMemRefType.getShape();
      if (curSizes.size() != nextSizes.size()) {
        if (fusedOps.size() < 2) {
          return false;
        }
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        std::tie(curIdx, curUpdOp, curMulOp) = *it;
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
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        std::tie(curIdx, curUpdOp, curMulOp) = *it;
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
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        std::tie(curIdx, curUpdOp, curMulOp) = *it;
        continue;
      }

      dist = nextOffset - curOffset;
      if (dist != 1) {
        if (fusedOps.size() < 2) {
          return false;
        }
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        std::tie(curIdx, curUpdOp, curMulOp) = *it;
        continue;
      }
      dists.push_back(dist);

      if ((xDist != -1 && xDist != dists[0]) ||
          (zDist != -1 && zDist != dists[1])) {
        if (fusedOps.size() < 2) {
          return false;
        }
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        std::tie(curIdx, curUpdOp, curMulOp) = *it;
        continue;
      }

      xDist = dists[0];
      zDist = dists[1];
      dupFactor = dists[0];

      fusedOps.push_back(nextMulOp);
      std::tie(curIdx, curUpdOp, curMulOp) = *it;

      if (fusedOps.size() > groupSize) {
        fusedOps.pop_back();
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        continue;
      }
    }
    groupFusedOps.push_back(fusedOps);
  }
  return true;
}

static bool canFoldMulAddChainToConvOp(
    arith::AddIOp addOp, MulDefMapTy &macChainMap,
    SmallVectorImpl<SmallVector<arith::MulIOp, 8>> &groupFusedOps,
    unsigned &dupFactor) {
  if (!isa<VectorType>(addOp.getType())) {
    return false;
  }

  VectorType resultType = addOp.getResult().getType().cast<VectorType>();

  if (!resultType.getElementType().isa<IntegerType>()) {
    return false;
  }

  IntegerType resultElType = resultType.getElementType().cast<IntegerType>();
  unsigned resultElWidth = resultElType.getWidth();
  unsigned laneSize = getVectorLaneSize(resultType);

  if ((laneSize != 32 || resultElWidth != 8) &&
      (laneSize != 16 || resultElWidth != 16)) {
    return false;
  }

  if (!addOp->hasOneUse()) {
    return false;
  }

  // Search for the last add op in the block.
  auto usrOp = *addOp->getUsers().begin();
  if (!usrOp || isa<arith::AddIOp>(usrOp)) {
    return false;
  }

  arith::AddIOp curAddOp = addOp;
  // bcastOpSourceVec is a container to trace the order of broadcast ops' source
  // in the chain.
  SmallVector<Value, 8> bcastOpSourceVec;

  // Identify the chain and build a mul add Chain map by recording the def of
  // mul ops.
  buildChainMap(curAddOp, macChainMap, bcastOpSourceVec);

  if (macChainMap.empty() ||
      std::any_of(macChainMap.begin(), macChainMap.end(),
                  [](const auto &p) { return p.second.size() < 2; })) {
    return false;
  }

  // Since we trace the order forwards, now reverse the vector.
  std::reverse(bcastOpSourceVec.begin(), bcastOpSourceVec.end());

  unsigned groupSize = resultElWidth == 16 ? 4 : 8;

  // Legality check for the mul add chain, and collect the ops that can be
  // transformed to mul_conv and mul_conv.
  if (!collectFusedOps(groupSize, dupFactor, bcastOpSourceVec, groupFusedOps,
                       macChainMap)) {
    return false;
  }

  if (std::any_of(groupFusedOps.begin(), groupFusedOps.end(),
                  [](const auto &ops) { return ops.size() < 2; }))
    return false;

  return true;
}

// This conversion pattern folds a mul add chain into mul_conv and mac_conv
// ops. Currently, we are handling the mul add chain with a sorted order so that
// the memrefs are sorted by increasing constant distances.
// TODO: handle the mul add chain with a random order.
struct FoldMulAddChainToConvOpPattern
    : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  FoldMulAddChainToConvOpPattern(MLIRContext *context, unsigned shiftParam = 0)
      : OpConversionPattern<arith::AddIOp>(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::AddIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<SmallVector<arith::MulIOp, 8>, 8> groupFusedOps;
    MulDefMapTy macChainMap;
    unsigned dupFactor = 1;

    if (!canFoldMulAddChainToConvOp(srcOp, macChainMap, groupFusedOps,
                                    dupFactor)) {
      return failure();
    }

    Value acc = nullptr;
    for (auto fusedOps : groupFusedOps) {
      arith::MulIOp mulOp = (*fusedOps.begin());
      arith::AddIOp addOp = cast<arith::AddIOp>(*mulOp->getUsers().begin());
      auto addLhs =
          dyn_cast<arith::MulIOp>(addOp->getOperand(0).getDefiningOp());
      auto addRhs =
          dyn_cast<arith::MulIOp>(addOp->getOperand(1).getDefiningOp());
      bool isMulConv = false;

      if (addLhs && addRhs) {
        isMulConv = true;
      } else {
        if (!acc)
          acc = addLhs ? addOp->getOperand(1) : addOp->getOperand(0);
      }

      // Get the mul op's lhs and rhs defining ops. We keep splat op at rhs.
      if (isa<aievec::BroadcastOp>(mulOp->getOperand(0).getDefiningOp())) {
        Value left = mulOp->getOperand(0);
        Value right = mulOp->getOperand(1);
        mulOp->setOperand(0, right);
        mulOp->setOperand(1, left);
      }

      Value lhs = mulOp->getOperand(0);
      Value rhs = mulOp->getOperand(1);

      VectorType vType = mulOp.getResult().getType().cast<VectorType>();
      Type sType = vType.getElementType();
      IntegerType iType = sType.cast<IntegerType>();
      unsigned width = iType.getWidth() <= 8 ? 32 : 64;
      int32_t M = iType.getWidth() == 8 ? 32 : 16;
      int32_t N = iType.getWidth() == 8 ? 8 : 4;

      Type ctype = mlir::IntegerType::get(iType.getContext(), width);
      Type opType = VectorType::get(vType.getShape(), ctype);

      aievec::BroadcastOp bcastOp =
          cast<aievec::BroadcastOp>(rhs.getDefiningOp());
      aievec::UPDOp bcastUPDOp =
          cast<aievec::UPDOp>(bcastOp.getSource().getDefiningOp());
      SmallVector<Value, 4> indices(bcastUPDOp.getIndices().begin(),
                                    bcastUPDOp.getIndices().end());
      unsigned lanes = 512 / getElementSizeInBits(vType);
      VectorType resType = createVectorType(lanes, sType);
      Value innerMostIdx = indices[indices.size() - 1];
      Value newIdx = innerMostIdx;
      int64_t val = -1;
      int64_t defIdx = -1;
      // Transfer
      // %c32 = arith.constant 32 : index
      // %1 = aievec.upd %arg1[%c32] {index = 0 : i8} : vector<32xi8>
      // %2 = aievec.broadcast %1 {idx = 0 : i8} : vector<32xi8>
      // to -
      // %c0 = arith.constant 0 : index
      // %1 = aievec.upd %arg1[%c0] {index = 0 : i8} : vector<64xi8>
      // %2 = aievec.broadcast %1 {idx = 32 : i8} : vector<32xi8>
      if (auto idxDefOp = innerMostIdx.getDefiningOp()) {
        if (auto constOp = dyn_cast<arith::ConstantOp>(idxDefOp)) {
          val = constOp.getValue().cast<IntegerAttr>().getInt();
          if (val) {
            defIdx = val / lanes * lanes;
            val %= lanes;
            newIdx = rewriter.create<arith::ConstantOp>(
                constOp.getLoc(),
                rewriter.getIntegerAttr(constOp.getType(), defIdx));
            indices[indices.size() - 1] = newIdx;
          }
        }
      }

      aievec::UPDOp newBcastOp = bcastUPDOp;

      // Rewrite the upd op with maximum vector lanes
      if (vType != resType) {
        newBcastOp = rewriter.create<aievec::UPDOp>(
            bcastUPDOp->getLoc(), resType, bcastUPDOp.getSource(), indices, 0,
            0, TypedValue<VectorType>(nullptr));
      }

      // Since we do not need to use duplicated data like in AIE1, if a
      // dup-factor exists, we extract the identical data by shuffle op. We use
      // mode 0 to extract the elements with even indices for i8 type data.
      Operation *shuffleOp = newBcastOp;
      if (dupFactor != 1) {
        shuffleOp = rewriter.create<aievec::ShuffleOp>(
            newBcastOp.getLoc(), resType, newBcastOp.getResult(), 0);
      }

      int32_t shiftBytes = (bcastOp.getIdx() + val) *
                           getElementSizeInBits(vType) / 8 / dupFactor;

      rhs = shuffleOp->getResult(0);

      // Generate a shift_bytes operation for rhs if the start position is not
      // 0.
      if (shiftBytes) {
        SmallVector<Value> sources = {shuffleOp->getResult(0)};

        rhs = rewriter.create<aievec::ShiftOp>(
            shuffleOp->getLoc(), sources.back().getType().cast<VectorType>(),
            sources, shiftBytes);
      }

      aievec::UPDOp lUPDOp = cast<aievec::UPDOp>(lhs.getDefiningOp());
      SmallVector<Value, 8> lIndices;
      lIndices.append(lUPDOp.getIndices().begin(), lUPDOp.getIndices().end());

      lhs = rewriter.create<aievec::UPDOp>(lUPDOp->getLoc(), resType,
                                           lUPDOp.getSource(), lIndices, 0, 0,
                                           TypedValue<VectorType>(nullptr));

      auto notMulOrAddOp = [&](Operation *op) {
        return !isa<arith::MulIOp, arith::AddIOp>(op);
      };

      Operation *convOp = nullptr;
      arith::AddIOp lastAddOp =
          cast<arith::AddIOp>(*(fusedOps.back()->getUsers().begin()));

      if (llvm::any_of(lastAddOp->getUsers(), notMulOrAddOp)) {
        if (isMulConv) {
          convOp = rewriter.create<aievec::MulConvOp>(srcOp->getLoc(), opType,
                                                      lhs, rhs, M, N);
        } else {
          convOp = rewriter.create<aievec::FMAConvOp>(
              srcOp->getLoc(), opType, lhs, rhs, acc, M, N, false);
        }
        rewriter.replaceOpWithNewOp<aievec::SRSOp>(
            srcOp, vType, convOp->getResult(0), shiftParam);
        return success();
      } else {
        if (isMulConv) {
          convOp = rewriter.create<aievec::MulConvOp>(srcOp->getLoc(), opType,
                                                      lhs, rhs, M, N);
        } else {
          convOp = rewriter.create<aievec::FMAConvOp>(
              srcOp->getLoc(), opType, lhs, rhs, acc, M, N, false);
        }
      }
      acc = convOp->getResult(0);
    }

    return failure();
  }

  unsigned shiftParam;
};
