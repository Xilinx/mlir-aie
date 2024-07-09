//===--FoldMulAddChainToConvOp.cpp - Fold Mul Add Chain To AIEVec Conv Op--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This is the implementation of the folding pass from mul add chain
// to AIEVec convolution operations, compatible with the AIE2 architecture.
//===----------------------------------------------------------------------===//

#include "FoldMulAddChainToConvOp.h"

#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/Analysis/Passes.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fold-mul-add-chain-to-conv"

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

namespace xilinx::aievec {
#define GEN_PASS_DEF_AIEVECCONVANALYSIS
#include "aie/Dialect/AIEVec/Analysis/Passes.h.inc"
} // namespace xilinx::aievec

/// This analysis builds the longest possible chain of MAC operations whose
/// operands are a vector that may or may not be shifted, and a broadcast.
/// That is, these MACs represent `vector x scalar` ops, and are candidates to
/// be grouped and replaced by mul_conv/fma_conv ops in AIE2.
//
// We build this chain recursively, climbing up the
struct LongestConvMACChainAnalysis {
  static AnalysisManager *am;

  struct ConvMac {
    // If there's a non-accumulating convolution upchain,
    // store it here temorarily.
    std::unique_ptr<ConvMac> topOfChainMulConv;
    // Accumulator value, if there is one.
    Value acc;
    // Left-hand side (non-broadcasting) source value
    Value lhs;
    // Left-hand side (broadcasting) source value
    Value rhs;
    // Amount that lhs is shifted
    uint8_t shift;
    // Element in rhs that is broadcasted
    uint8_t bcastIdx;
    ConvMac(Value lhs, Value rhs, uint8_t shift, uint8_t bcastIdx)
        : topOfChainMulConv(nullptr), acc(nullptr), lhs(lhs), rhs(rhs),
          shift(shift), bcastIdx(bcastIdx) {}
  };

  struct ConvMacChainGroup {
    // Group start index within the chain
    uint64_t fromIdx;
    // Index in chain after group last MAC
    uint64_t toIdx;
    // Initial position of the signal to be convolved
    int64_t signalShift;
    // Initial position of the convolution filter
    int64_t bcastShift;
    // Distance between elements in the filter
    int64_t bcastDist; // Must be 1 or 2
  };

  typedef SmallVector<std::unique_ptr<ConvMac>, 8> ConvMacChain;
  typedef SmallVector<ConvMacChainGroup, 8> ConvMacChainGroupList;

  std::unique_ptr<ConvMacChain> convMacChain;
  ConvMacChainGroupList groupsInChain;

  /// Sort the chain of MACs by sources. When two MACs share the same sources,
  /// sort them by the broadcast index. If they don't, sort them by the order
  /// of the ops in the code. This function should be called after the chain
  /// is completed, and before operating on the groups of MACs. After sorting,
  /// MACs that can be fused into single convolution ops will be contiguous in
  /// the chain.
  void sortChain() {
    if ((*convMacChain)[0]->acc) {
      std::sort(convMacChain->begin(), convMacChain->end(),
                [](const auto &a, const auto &b) {
                  if (a->lhs == b->lhs) {
                    if (a->rhs == b->rhs)
                      return a->bcastIdx < b->bcastIdx;
                    return a->rhs.getDefiningOp()->isBeforeInBlock(
                        b->rhs.getDefiningOp());
                  }
                  // We should probably sort by lhs load address, if it exists
                  // XXX: We assume all MACs in the same block. If they're not,
                  // XXX: this will assert.
                  return a->lhs.getDefiningOp()->isBeforeInBlock(
                      b->lhs.getDefiningOp());
                });
    } else {
      // If the top of the chain is not an accumulation, bring up all related
      // convolution MACs and sort the rest by lhs.
      auto firstLhs = (*convMacChain)[0]->lhs;
      std::sort(convMacChain->begin(), convMacChain->end(),
                [&firstLhs](const auto &a, const auto &b) {
                  if (a->lhs == b->lhs) {
                    if (a->rhs == b->rhs)
                      return a->bcastIdx < b->bcastIdx;
                    return a->rhs.getDefiningOp()->isBeforeInBlock(
                        b->rhs.getDefiningOp());
                  }
                  if (a->lhs == firstLhs)
                    return true;
                  if (b->lhs == firstLhs)
                    return false;
                  return a->lhs.getDefiningOp()->isBeforeInBlock(
                      b->lhs.getDefiningOp());
                });
      // Float the empty accumulator to the top.
      if ((*convMacChain)[0]->acc)
        for (auto &convMac : *convMacChain)
          if (!convMac->acc) {
            std::swap((*convMacChain)[0]->acc, convMac->acc);
            break;
          }
    }
  }

  // Return the list of convolution mac ops in the chain as pairs of indices
  // indicating the position within the chain where a group starts and the
  // position where it ends: [start, end). If they have not been precomputed
  // yet, this method will generate them.
  const ConvMacChainGroupList &getGroupsInChain() {
    // If there's no group or it's been computed already, return stored list.
    if (groupsInChain.size() > 0 || !convMacChain || convMacChain->size() == 0)
      return groupsInChain;

    uint64_t grpStartIdx = 0;
    uint64_t grpCurIdx = 0;
    Value curLhs = (*convMacChain)[0]->lhs;
    Value curRhs = (*convMacChain)[0]->rhs;
    for (const auto &convMac : *convMacChain) {
      if (grpCurIdx > grpStartIdx) {
        if (curLhs != convMac->lhs || curRhs != convMac->rhs) {
          groupsInChain.push_back({grpStartIdx, grpCurIdx,
                                   getGroupSignalShift(grpStartIdx, grpCurIdx),
                                   getGroupBcastShift(grpStartIdx, grpCurIdx),
                                   getGroupBcastDist(grpStartIdx, grpCurIdx)});
          grpStartIdx = grpCurIdx;
          curLhs = convMac->lhs;
          curRhs = convMac->rhs;
        }
      }
      grpCurIdx++;
    }
    if (grpStartIdx < grpCurIdx)
      groupsInChain.push_back({grpStartIdx, grpCurIdx,
                               getGroupSignalShift(grpStartIdx, grpCurIdx),
                               getGroupBcastShift(grpStartIdx, grpCurIdx),
                               getGroupBcastDist(grpStartIdx, grpCurIdx)});
    return groupsInChain;
  }

  // Return the signal shift for the group in the MAC chain in [fromIdx, toIdx)
  // the top. This method verifies that the elements of the signal are
  // contiguously accessed. If they do not, or the specified group doesn't
  // exist, this function returns -1.
  int64_t getGroupSignalShift(uint64_t fromIdx, uint64_t toIdx) {
    if (fromIdx >= toIdx || toIdx > convMacChain->size())
      return -1;
    if (toIdx == fromIdx + 1)
      return static_cast<int64_t>((*convMacChain)[fromIdx]->shift);
    for (uint64_t i = fromIdx; i < toIdx - 1; i++)
      if ((static_cast<int64_t>((*convMacChain)[i + 1]->shift) -
           static_cast<int64_t>((*convMacChain)[i]->shift)) != 1)
        return -1;
    return static_cast<int64_t>((*convMacChain)[fromIdx]->shift);
  }

  // Return the shift in value of the first broadcasted element in the i-th
  // group. If there is no chain, or the i-th group does not exist,
  // returns -1.
  int64_t getGroupBcastShift(uint64_t fromIdx, uint64_t toIdx) {
    if (fromIdx >= toIdx || toIdx > convMacChain->size())
      return -1;
    return static_cast<int64_t>((*convMacChain)[fromIdx]->bcastIdx);
  }

  // Returns the broadcast distance between elements within the group. If the
  // distance is not constant and equal to 1 or 2, it returns -1.
  int64_t getGroupBcastDist(uint64_t fromIdx, uint64_t toIdx) {
    if (fromIdx >= toIdx || toIdx > convMacChain->size())
      return -1;
    if (toIdx == fromIdx + 1)
      return 1;
    int64_t bcastDist =
        static_cast<int64_t>((*convMacChain)[fromIdx + 1]->bcastIdx) -
        static_cast<int64_t>((*convMacChain)[fromIdx]->bcastIdx);
    if (bcastDist != 1 && bcastDist != 2)
      return -1;
    for (uint64_t i = fromIdx + 1; i < toIdx - 1; i++)
      if ((static_cast<int64_t>((*convMacChain)[i + 1]->bcastIdx) -
           static_cast<int64_t>((*convMacChain)[i]->bcastIdx)) != bcastDist)
        return -1;
    return bcastDist;
  }

  bool canChainBeReplacedWithConvOps() {
    const auto &groups = getGroupsInChain();
    if (groups.size() == 0)
      return false;
    for (const auto &group : groups)
      if (group.signalShift == -1 || group.bcastShift == -1 ||
          group.bcastDist == -1)
        return false;
    return true;
  }

  std::unique_ptr<ConvMac> getConvMacFromMulOp(arith::MulIOp mulOp) {
    auto mulOpLhsDefOp = mulOp.getLhs().getDefiningOp();
    auto mulOpRhsDefOp = mulOp.getRhs().getDefiningOp();
    if (!mulOpLhsDefOp || !mulOpRhsDefOp)
      return nullptr;

    Value convMacRhs = nullptr;
    uint8_t convMacBcastIdx = 0;

    auto getConvMacRhs = [&](Operation *mulOpOperand) -> bool {
      SetVector<Operation *> opBwdSlices;
      auto opFilter = [](Operation *op) {
        return isa<aievec::BroadcastOp>(op) || isa<aievec::ExtOp>(op) ||
               isa<aievec::ConcatOp>(op);
      };
      BackwardSliceOptions backwardSliceOptions;
      backwardSliceOptions.filter = opFilter;

      getBackwardSlice(mulOpOperand, &opBwdSlices, backwardSliceOptions);
      opBwdSlices.insert(mulOpOperand);

      LLVM_DEBUG(llvm::dbgs() << "opBwdSlices = [\n");
      for ([[maybe_unused]] auto op : opBwdSlices) {
        LLVM_DEBUG(llvm::dbgs() << *op << "\n");
      }
      LLVM_DEBUG(llvm::dbgs() << "]\n");

      if (opBwdSlices.size() == 1) {
        if (auto bcastOp = dyn_cast<aievec::BroadcastOp>(opBwdSlices[0])) {
          convMacRhs = bcastOp.getSource();
          convMacBcastIdx = bcastOp.getIdx();
          return true;
        }
      } else if (opBwdSlices.size() >= 3) {
        auto sliceSz = opBwdSlices.size();
        if ((isa<aievec::ExtOp>(opBwdSlices[sliceSz - 3]) &&
             isa<aievec::BroadcastOp>(opBwdSlices[sliceSz - 2]) &&
             isa<aievec::ConcatOp>(opBwdSlices[sliceSz - 1])) ||
            (isa<aievec::ConcatOp>(opBwdSlices[sliceSz - 3]) &&
             isa<aievec::BroadcastOp>(opBwdSlices[sliceSz - 2]) &&
             isa<aievec::ExtOp>(opBwdSlices[sliceSz - 1]))) {
          convMacRhs = opBwdSlices[sliceSz - 3]->getOperand(0);
          convMacBcastIdx =
              dyn_cast<aievec::BroadcastOp>(opBwdSlices[sliceSz - 2]).getIdx();
          return true;
        }
      }

      return false;
    };

    // Obtain the broadcast operation feeding into the MulIOp
    if (!getConvMacRhs(mulOpRhsDefOp)) {
      if (getConvMacRhs(mulOpLhsDefOp)) {
        std::swap(mulOpLhsDefOp, mulOpRhsDefOp);
      }
    }
    if (!convMacRhs)
      return nullptr;

    // Obtain the ext or ext->shift op feeding into the MulIOp
    aievec::ExtOp extOp;
    aievec::ShiftOp shiftOp;
    shiftOp = dyn_cast<aievec::ShiftOp>(mulOpLhsDefOp);
    if (shiftOp)
      extOp = shiftOp.getLhs().getDefiningOp<aievec::ExtOp>();
    else
      extOp = dyn_cast<aievec::ExtOp>(mulOpLhsDefOp);

    // XXX: Actually, ExtOp might not exist but should work anyway.
    // XXX: Should it, though?
    if (!extOp)
      return nullptr;

    Value convMacLhs = extOp.getSource();
    uint8_t shift = 0;
    if (shiftOp) {
      auto shiftConstDefOp =
          shiftOp.getShift().getDefiningOp<arith::ConstantOp>();
      if (shiftConstDefOp) {
        auto shiftAttr = cast<IntegerAttr>(shiftConstDefOp.getValue());
        auto vType = cast<VectorType>(mulOp.getResult().getType());
        shift = 8 * shiftAttr.getInt() / getElementSizeInBits(vType);
      }
    }

    return std::make_unique<ConvMac>(convMacLhs, convMacRhs, shift,
                                     convMacBcastIdx);
  }

  std::unique_ptr<ConvMac> getConvMacFromAddOp(arith::AddIOp addOp) {
    // Make sure at least one of them is a multiplication, and the other one
    // is the accumulator coming form upchain.
    auto mulOp = addOp.getLhs().getDefiningOp<arith::MulIOp>();
    Value acc = addOp.getRhs();
    if (!mulOp) {
      mulOp = addOp.getRhs().getDefiningOp<arith::MulIOp>();
      acc = addOp.getLhs();
    }
    if (!mulOp)
      return nullptr;

    // Get the parameters of the convolution from the operands of the MulIOp
    auto convMac = getConvMacFromMulOp(mulOp);
    if (!convMac)
      return nullptr;

    // If both sides are MulIOp, we might be at the top of the chain
    auto upChainAccMulOp = acc.getDefiningOp<arith::MulIOp>();
    if (upChainAccMulOp) {
      auto convMac2 = getConvMacFromMulOp(upChainAccMulOp);
      // XXX: We pre-sort the top two MACs to make sure that an undefined
      // XXX: accumulator ends up on top of the chain.
      // XXX: But it might not be necessary? CHECK!
      if (convMac2 && convMac->lhs == convMac2->lhs &&
          convMac->rhs == convMac->rhs) {
        if (convMac->bcastIdx < convMac2->bcastIdx &&
            convMac->shift < convMac2->shift) {
          convMac2->topOfChainMulConv = std::move(convMac);
          convMac2->acc = acc;
          return convMac2;
        } else if (convMac->bcastIdx > convMac2->bcastIdx &&
                   convMac->shift > convMac2->shift) {
          convMac->topOfChainMulConv = std::move(convMac2);
          convMac->acc = acc;
          return convMac;
        } else {
          // WARNING: In this situation, the chain is ambiguous and picking one
          // WARNING: option over the other may result in a successful
          // WARNING: and/or better replacement. Here, we are assuming that
          // WARNING: is going to be either one or the other, or it won't
          // WARNING: matter.
        }
      } else {
        convMac->topOfChainMulConv = std::move(convMac2);
      }
    }
    convMac->acc = acc;
    return convMac;
  }

  LongestConvMACChainAnalysis(arith::AddIOp addOp) {
    std::unique_ptr<ConvMac> macConvChainElem = getConvMacFromAddOp(addOp);
    if (!macConvChainElem)
      return;

    if (macConvChainElem->acc) {
      auto upChainAddOp = macConvChainElem->acc.getDefiningOp<arith::AddIOp>();
      if (upChainAddOp) {
        auto &upChainChainAnalysis =
            am->getChildAnalysis<LongestConvMACChainAnalysis>(upChainAddOp);
        if (upChainChainAnalysis.convMacChain) {
          convMacChain = std::move(upChainChainAnalysis.convMacChain);
          convMacChain->push_back(std::move(macConvChainElem));
          return;
        }
      }
    }
    assert(!convMacChain && "Convolution MAC chain unexpectedly not empty");
    convMacChain = std::make_unique<ConvMacChain>();
    if (macConvChainElem->topOfChainMulConv)
      convMacChain->push_back(std::move(macConvChainElem->topOfChainMulConv));
    convMacChain->push_back(std::move(macConvChainElem));
  }
};
// HACK: For some reason, it's not possible to access the analysis manager from
// HACK: within an analysis, but we need it to build the analysis recursively.
// HACK: If there is a good reason not to do this, we should find an
// HACK: alternative way to build the MAC chain.
AnalysisManager *LongestConvMACChainAnalysis::am = nullptr;

// This conversion pattern folds a MAC chain into mul_conv and mac_conv
// ops. We can handle the mul MAC with a random order.
struct FoldMulAddChainToConvOpPattern
    : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  FoldMulAddChainToConvOpPattern(MLIRContext *context, AnalysisManager &am,
                                 unsigned shiftParam = 0)
      : OpConversionPattern<arith::AddIOp>(context), am(am),
        shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::AddIOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &convMacChainAnalysis =
        am.getChildAnalysis<LongestConvMACChainAnalysis>(srcOp);
    auto &convMacChain = convMacChainAnalysis.convMacChain;
    if (!convMacChain)
      return failure();

    auto loc = srcOp.getLoc();
    VectorType vecTy = cast<VectorType>(srcOp.getResult().getType());
    unsigned elemWidth = cast<IntegerType>(vecTy.getElementType()).getWidth();
    unsigned accWidth = elemWidth <= 8 ? 32 : 64;
    int32_t M = elemWidth == 8 ? 32 : 16;
    int32_t N = elemWidth == 8 ? 8 : 4;

    Type wideElemTy = IntegerType::get(getContext(), accWidth);
    Type accVecTy = VectorType::get(vecTy.getShape(), wideElemTy);

    const auto &groups = convMacChainAnalysis.getGroupsInChain();
    Value grpAcc = (*convMacChain)[groups[0].fromIdx]->acc;
    if (grpAcc)
      grpAcc = rewriter
                   .create<aievec::UPSOp>(srcOp.getLoc(), accVecTy, grpAcc,
                                          /*shift=*/0)
                   .getResult();
    for (const auto &group : groups) {
      Value grpLhs = (*convMacChain)[group.fromIdx]->lhs;
      Value grpRhs = (*convMacChain)[group.fromIdx]->rhs;
      auto filterVecTy = cast<VectorType>(grpRhs.getType());
      auto signalVecTy = cast<VectorType>(grpLhs.getType());
      // Sort out the vector used as filter
      // If the length of the filter is half that of the signal, concatenate
      // the filter with itself.
      if (2 * filterVecTy.getShape()[0] == signalVecTy.getShape()[0])
        grpRhs =
            rewriter
                .create<aievec::ConcatOp>(
                    loc, signalVecTy, SmallVector<Value, 2>({grpRhs, grpRhs}))
                .getResult();
      // If the filter has duplicate elements, pack them.
      if (group.bcastDist == 2)
        // NOTE: This shuffle mode works for `vector<64xi8>`
        grpRhs = rewriter
                     .create<aievec::ShuffleOp>(loc, signalVecTy, grpRhs,
                                                grpRhs, ShuffleMode::T8_64X2_LO)
                     .getResult();
      // If the first element of the filter to be used is not 0, shift the
      // filter to align the first element to the beginning.
      if (group.bcastShift) {
        int32_t shiftBytes =
            group.bcastShift * getElementSizeInBits(filterVecTy) >>
            (3 + group.bcastDist - 1);
        auto shiftBytesCst =
            rewriter
                .create<arith::ConstantOp>(
                    loc, rewriter.getI32IntegerAttr(shiftBytes))
                .getResult();
        grpRhs = rewriter
                     .create<aievec::ShiftOp>(grpRhs.getDefiningOp()->getLoc(),
                                              signalVecTy, grpRhs, grpRhs,
                                              shiftBytesCst)
                     .getResult();
      }
      // Sort out the vector used as signal
      // If the signal to be convolved doesn't start at element 0, shift the
      // signal to align the first element to the beginning.
      if (group.signalShift) {
        int32_t shiftBytes =
            group.signalShift * getElementSizeInBits(signalVecTy) >> 3;
        auto shiftBytesCst =
            rewriter
                .create<arith::ConstantOp>(
                    loc, rewriter.getI32IntegerAttr(shiftBytes))
                .getResult();
        grpLhs = rewriter
                     .create<aievec::ShiftOp>(loc, signalVecTy, grpLhs, grpLhs,
                                              shiftBytesCst)
                     .getResult();
      }
      // Generate a convolution operation for the group
      // If there is no upchain accumulator, use a mul_conv; use a mac_conv
      // otherwise.
      if (!grpAcc)
        grpAcc = rewriter
                     .create<aievec::MulConvOp>(srcOp.getLoc(), accVecTy,
                                                grpLhs, grpRhs, M, N)
                     .getResult();
      else
        grpAcc =
            rewriter
                .create<aievec::FMAConvOp>(srcOp.getLoc(), accVecTy, grpLhs,
                                           grpRhs, grpAcc, M, N, false)
                .getResult();
    }

    auto shiftParamOp = rewriter.create<arith::ConstantOp>(
        srcOp.getLoc(), rewriter.getI32IntegerAttr(shiftParam));
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(srcOp, vecTy, grpAcc,
                                               shiftParamOp.getResult());
    return success();
  }

  AnalysisManager &am;
  unsigned shiftParam;
};

namespace xilinx::aievec {

void configureAIEVecConvOpTransformationLegalizations(ConversionTarget &target,
                                                      AnalysisManager &am,
                                                      TargetBackend backend) {
  LongestConvMACChainAnalysis::am = &am;
  target.addLegalDialect<AIEVecDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<arith::AddIOp>([&am](arith::AddIOp op) {
    auto &convAnalysis = am.getChildAnalysis<LongestConvMACChainAnalysis>(op);
    return !convAnalysis.canChainBeReplacedWithConvOps();
  });
}

void populateAIEVecConvOpTransformationPatterns(RewritePatternSet &patterns,
                                                AnalysisManager &am,
                                                unsigned shiftParam,
                                                TargetBackend backend) {
  patterns.add<FoldMulAddChainToConvOpPattern>(patterns.getContext(), am,
                                               shiftParam);
}

struct AIEVecConvAnalysis : public AIEVecConvAnalysisBase<AIEVecConvAnalysis> {
  AIEVecConvAnalysis() = default;
  using ConvMacChain = LongestConvMACChainAnalysis::ConvMacChain;
  using ConvMacChainGroupList =
      LongestConvMACChainAnalysis::ConvMacChainGroupList;

  void runOnOperation() override {
    markAllAnalysesPreserved();
    AnalysisManager am = getAnalysisManager();
    LongestConvMACChainAnalysis::am = &am;
    Operation *op = getOperation();

    // Compute all the chains
    op->walk([&](arith::AddIOp addOp) {
      if (isa<VectorType>(addOp.getResult().getType()))
        am.getChildAnalysis<LongestConvMACChainAnalysis>(addOp);
    });

    // Sort the chains, ready to split by group
    op->walk([&](arith::AddIOp addOp) {
      if (isa<VectorType>(addOp.getResult().getType())) {
        auto &analysis =
            am.getChildAnalysis<LongestConvMACChainAnalysis>(addOp);
        if (analysis.convMacChain)
          analysis.sortChain();
      }
    });

    if (printResult) {
      op->walk([&](arith::AddIOp addOp) {
        if (isa<VectorType>(addOp.getResult().getType())) {
          auto &macChainAnalysis =
              am.getChildAnalysis<LongestConvMACChainAnalysis>(addOp);
          if (macChainAnalysis.canChainBeReplacedWithConvOps()) {
            addOp.print(llvm::outs());
            llvm::outs() << " is at the end of a convolution MAC Chain:\n";
            listChain(macChainAnalysis.convMacChain,
                      macChainAnalysis.getGroupsInChain());
          }
        }
      });
    }
  }

  void listChain(const std::unique_ptr<ConvMacChain> &chain,
                 const ConvMacChainGroupList &groups) const {
    uint64_t gIdx = 0;
    for (const auto &group : groups) {
      llvm::outs() << "-------------- GROUP " << std::to_string(gIdx)
                   << " --------------\n";
      llvm::outs() << "  Signal Shift: " << std::to_string(group.signalShift)
                   << "   Kernel Shift: " << std::to_string(group.bcastShift)
                   << "   Kernel Duplication: "
                   << std::to_string(group.bcastDist) << "\n";
      for (uint64_t i = group.fromIdx; i < group.toIdx; i++) {
        auto shift = (*chain)[i]->shift;
        auto bcastIdx = (*chain)[i]->bcastIdx;
        auto lhsOp = (*chain)[i]->lhs.getDefiningOp();
        auto rhsOp = (*chain)[i]->rhs.getDefiningOp();
        if (!(*chain)[i]->acc)
          llvm::outs() << "  [mul_conv]\n";
        llvm::outs() << "    [Shift: " << std::to_string(shift) << "]: ";
        lhsOp->print(llvm::outs());
        llvm::outs() << "\n    [Bcast: " << std::to_string(bcastIdx) << "]: ";
        rhsOp->print(llvm::outs());
        llvm::outs() << "\n";
      }
      gIdx++;
    }
    llvm::outs() << "-------------------------------------\n";
  }
};

std::unique_ptr<Pass> createAIEVecConvolutionAnalysisPass() {
  return std::make_unique<AIEVecConvAnalysis>();
}

} // namespace xilinx::aievec
