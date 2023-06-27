//===-ConvertVectorToAIEVec.cpp - Lower Vector to AIE vector ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This is the implementation of the lowering pass from standard Vector
// dialect to AIEVec, compatible with the AIE vector architecture.
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <optional>
#include <tuple>

#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/Transforms/IntervalReuse.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"

#include "FoldMulAddChainToConvOp.h"

namespace xilinx::aievec {
#define GEN_PASS_DEF_LOWERVECTORTOAIEVEC
#define GEN_PASS_DEF_CANONICALIZEFORAIEVEC
#define GEN_PASS_DEF_REDUNDANTLOADSTOREOPTIMIZATION
#define GEN_PASS_DEF_AIEVECTRANSFORMATION
#define GEN_PASS_DEF_AIEVECCONVOPTRANSFORMATION

#include "aie/Dialect/AIEVec/Transforms/Passes.h.inc"
} // namespace xilinx::aievec

namespace xilinx {
enum class AIEArch {
  AIE,    // Original AIE
  AIE_ML, // ML/V2 version of AIE
};
} // namespace xilinx

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

#define DEBUG_TYPE "aievec-lowering"

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Given the LHS and RHS of an `arith::AddIOp`, if one of them is defined by an
// `arith::MulIOp`, return a tuple with the `lhs`, `rhs`, and `acc` of the MAC
// operation that can replace them.
static std::optional<std::tuple<Value, Value, Value>>
extractMACOperandsFromAddOperands(Value addLhs, Value addRhs) {
  auto lhsDefOp = addLhs.getDefiningOp();
  auto rhsDefOp = addRhs.getDefiningOp();
  arith::MulIOp mulOp = nullptr;
  Value acc;
  if (lhsDefOp) {
    mulOp = dyn_cast<arith::MulIOp>(lhsDefOp);
    acc = addRhs;
  }
  if (!mulOp && rhsDefOp) {
    mulOp = dyn_cast<arith::MulIOp>(rhsDefOp);
    acc = addLhs;
  }
  if (!mulOp)
    return {};
  return std::make_tuple(mulOp.getLhs(), mulOp.getRhs(), acc);
}

//===----------------------------------------------------------------------===//
// Analyses
//===----------------------------------------------------------------------===//

// Calculates the effective size of the load operation (in bits).
// If a long UPD is followed by another one with an offset, we count
// its effective size as the number of bits loaded up to that offset.
// E.g.:
//  As is, the effective size of:
//     %0 = aievec.upd %m[%i] {index = 0 : i8, offset = 0 : si32}
//                            : memref<256xi32>, vector<32xi32>
//  would be `8 * sizeof(i32) * 32` (i.e: 1024 bits).
//  On the other, for two arranged like so:
//     %0 = aievec.upd %m[%i] {index = 0 : i8, offset = 0 : si32}
//                            : memref<256xi32>, vector<32xi32>
//     %1 = aievec.upd %m[%i], %1 {index = 1 : i8, offset = 512 : si32}
//                                : memref<256xi32>, vector<32xi32>
// it would be `8 * sizeof(i32) * 32 - 512` (i.e.: 512 bits) each.
struct UPDOpEffectiveAccessSizeAnalysis {
  UPDOpEffectiveAccessSizeAnalysis(aievec::UPDOp updOp) {
    auto vecType = cast<VectorType>(updOp.getResult().getType());
    unsigned sizeInBits =
        cast<ShapedType>(vecType).getSizeInBits() - updOp.getOffset();
    for (Operation *user : updOp->getUsers()) {
      auto userUpdOp = dyn_cast<xilinx::aievec::UPDOp>(user);
      if (userUpdOp)
        sizeInBits -= userUpdOp.getOffset();
    }
    effectiveSize = sizeInBits;
  }

  unsigned effectiveSize;
};

//===----------------------------------------------------------------------===//
// Lowering patterns
//===----------------------------------------------------------------------===//
// This pattern fold `vector.extract` and `vector.broadcast` into
// `aievec.broadcast` for aie-ml
struct FoldVectorExtractAndBroadcastToAIEBroadcast
    : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp bcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto extOp =
        dyn_cast<vector::ExtractOp>(bcastOp.getSource().getDefiningOp());

    if (!extOp)
      return failure();

    auto src = extOp.getVector();
    auto pos = extOp.getPosition();
    VectorType resultType = bcastOp.getResult().getType().cast<VectorType>();

    rewriter.replaceOpWithNewOp<aievec::BroadcastOp>(
        bcastOp, resultType, src, cast<IntegerAttr>(pos[0]).getInt());

    return success();
  }
};

static bool canFoldAIEShiftAndBroadcast(aievec::BroadcastOp op,
                                        aievec::ShiftOp &shiftOp,
                                        int32_t &idx) {
  if (!op.getSource().getDefiningOp())
    return false;

  shiftOp = dyn_cast<aievec::ShiftOp>(op.getSource().getDefiningOp());

  if (!shiftOp)
    return false;

  VectorType vType = shiftOp->getResult(0).getType().cast<VectorType>();
  int32_t elemSize = getElementSizeInBits(vType);
  auto constOp = cast<arith::ConstantOp>(shiftOp.getShift().getDefiningOp());
  int32_t shiftBytes = cast<IntegerAttr>(constOp.getValue()).getInt();
  idx = shiftBytes * 8 / elemSize + op.getIdx();

  if (idx <= 0 || idx >= (int32_t)getVectorLaneSize(vType)) {
    return false;
  }

  return true;
}

// Create MulElemOp for i8 and bf16 types in aie-ml. The corresponding intrinsic
// is mul_elem_16_2, which indicates that we need to concatenate zero vectors
// for both mul operands before creating MulElemOp.
static aievec::MulElemOp createMulElemAieML(ConversionPatternRewriter &rewriter,
                                            Value lval, Value rval,
                                            VectorType srcType,
                                            unsigned bitWidth, Location loc) {
  Type accType = getVectorOpDestType(srcType, /*AIEML =*/true);
  VectorType vecType =
      createVectorType(512 / bitWidth, srcType.getElementType());

  arith::ConstantOp zeroConstOp = nullptr;
  if (bitWidth == 8) {
    zeroConstOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(srcType.getElementType(), 0));
  } else {
    zeroConstOp =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF16FloatAttr(0));
  }
  auto broadcastZeroOp = rewriter.create<aievec::BroadcastScalarOp>(
      loc, vecType, zeroConstOp->getResult(0));
  auto extOp = rewriter.create<aievec::ExtOp>(loc, srcType,
                                              broadcastZeroOp.getResult(), 0);

  SmallVector<Value> lSources = {lval, extOp->getResult(0)};
  SmallVector<Value> rSources = {rval, extOp->getResult(0)};
  auto lConcatOp = rewriter.create<aievec::ConcatOp>(loc, vecType, lSources);
  auto rConcatOp = rewriter.create<aievec::ConcatOp>(loc, vecType, rSources);

  auto mulElemOp = rewriter.create<aievec::MulElemOp>(
      loc, accType, lConcatOp->getResult(0), rConcatOp->getResult(0));
  return mulElemOp;
}

struct FoldAIEShiftAndBroadcast
    : public OpConversionPattern<aievec::BroadcastOp> {
  using OpConversionPattern<aievec::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aievec::BroadcastOp bcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    aievec::ShiftOp shiftOp = nullptr;
    int32_t idx = 0;

    if (!canFoldAIEShiftAndBroadcast(bcastOp, shiftOp, idx)) {
      return failure();
    }

    VectorType resultType = bcastOp.getResult().getType().cast<VectorType>();

    rewriter.replaceOpWithNewOp<aievec::BroadcastOp>(bcastOp, resultType,
                                                     shiftOp.getLhs(), idx);

    return success();
  }
};

// This pattern replaces `arith.muli`+`arith.addi` on vectors with
// `aievec.mac_elem`. This pattern works for aie-ml.
struct ConvertMulAddToAIEVecFMAElemOpPattern
    : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  ConvertMulAddToAIEVecFMAElemOpPattern(MLIRContext *context,
                                        unsigned shiftParam = 0)
      : OpConversionPattern<arith::AddIOp>(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::AddIOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    VectorType resultType = dyn_cast<VectorType>(addOp.getType());
    if (!resultType)
      return failure();

    // Verify it can be replaced by a MAC
    auto res =
        extractMACOperandsFromAddOperands(adaptor.getLhs(), adaptor.getRhs());
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    // Verify the vector type is supported by AIEML
    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if ((laneSize != 32 || resultElWidth != 16) &&
        (laneSize != 16 || resultElWidth != 32))
      return failure();

    Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
                                       /*AIEML =*/true);
    auto upsOp = rewriter.create<aievec::UPSOp>(addOp.getLoc(), accType, acc,
                                                shiftParam);
    auto fmaElemOp = rewriter.create<aievec::FMAElemOp>(
        addOp.getLoc(), accType, lhs, rhs, upsOp.getResult(),
        /*fmsub=*/false);
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(
        addOp, resultType, fmaElemOp.getResult(), shiftParam);

    return success();
  }

  unsigned shiftParam;
};

// This pattern replaces `arith.mulf` on vectors with
// `aievec.mul_elem`. This pattern works for aie-ml.
struct ConvertMulFToAIEVecMulElemOpPattern
    : public OpConversionPattern<arith::MulFOp> {
  using OpConversionPattern<arith::MulFOp>::OpConversionPattern;

  ConvertMulFToAIEVecMulElemOpPattern(MLIRContext *context,
                                      unsigned shiftParam = 0)
      : OpConversionPattern<arith::MulFOp>(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::MulFOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    VectorType resultType = dyn_cast<VectorType>(mulOp.getType());
    if (!resultType)
      return failure();

    auto isAddOp = [&](Operation *op) { return isa<arith::AddFOp>(op); };
    // Verify it is not a part of FMA
    if (mulOp->hasOneUse() && llvm::any_of(mulOp->getUsers(), isAddOp))
      return failure();

    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();

    unsigned laneSize = getVectorLaneSize(resultType);

    // bfloat16 type
    if (laneSize != 16 || (resultElWidth != 16 && resultElWidth != 32))
      return failure();

    aievec::MulElemOp mulElemOp = nullptr;

    if (resultElWidth == 16) {
      mulElemOp =
          createMulElemAieML(rewriter, adaptor.getLhs(), adaptor.getRhs(),
                             resultType, resultElWidth, mulOp.getLoc());
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          mulOp, resultType, mulElemOp.getResult(), shiftParam);
    }
    // float type
    else {
      auto lhs = dyn_cast<arith::ExtFOp>(adaptor.getLhs().getDefiningOp());
      auto rhs = dyn_cast<arith::ExtFOp>(adaptor.getRhs().getDefiningOp());

      if (!lhs || !rhs)
        return failure();

      auto lval = lhs->getOperand(0);
      auto rval = rhs->getOperand(0);

      VectorType lSrcType = cast<VectorType>(lval.getType());
      VectorType rSrcType = cast<VectorType>(rval.getType());

      unsigned lBitWidth = lSrcType.getElementType().getIntOrFloatBitWidth();
      unsigned rBitWidth = rSrcType.getElementType().getIntOrFloatBitWidth();

      if (lBitWidth != 16 || rBitWidth != 16)
        return failure();

      mulElemOp = createMulElemAieML(rewriter, lval, rval, lSrcType, lBitWidth,
                                     mulOp.getLoc());
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          mulOp, resultType, mulElemOp.getResult(), /*isResAcc*/ false);
    }
    return success();
  }
  unsigned shiftParam;
};

// This pattern replaces `arith.muli` on vectors with
// `aievec.mul_elem`. This pattern works for aie-ml.
struct ConvertMulIToAIEVecMulElemOpPattern
    : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  ConvertMulIToAIEVecMulElemOpPattern(MLIRContext *context,
                                      unsigned shiftParam = 0)
      : OpConversionPattern<arith::MulIOp>(context), shiftParam(shiftParam) {}

  LogicalResult
  matchAndRewrite(arith::MulIOp mulOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    VectorType resultType = dyn_cast<VectorType>(mulOp.getType());
    if (!resultType)
      return failure();

    auto isAddOp = [&](Operation *op) { return isa<arith::AddIOp>(op); };
    // Verify it is not a part of MAC
    if (mulOp->hasOneUse() && llvm::any_of(mulOp->getUsers(), isAddOp))
      return failure();

    // Verify the vector type is supported by AIEML
    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if ((laneSize != 32 || (resultElWidth != 16 && resultElWidth != 8)) &&
        ((laneSize != 16 && laneSize != 32) || resultElWidth != 32))
      return failure();

    // Deal with the case with sext op for i8 and i16:
    // Case 1:
    // Transfer -
    // %1 = arith.extsi %a : vector<32xi8> to vector<32xi32>
    // %2 = arith.extsi %b : vector<32xi8> to vector<32xi32>
    // %3 = arith.muli %1, %2 : vector<32xi32>
    // to -
    // aievec.mul_elem(%a, %b) : vector<64xi8>, vector<64xi8>, vector<32xi32>
    //
    // Case 2:
    // Transfer -
    // %1 = arith.extsi %a : vector<32xi16> to vector<32xi32>
    // %2 = arith.extsi %b : vector<32xi16> to vector<32xi32>
    // %3 = arith.muli %1, %2 : vector<32xi32>
    // to -
    // aievec.mul_elem(%a, %b) : vector<32xi16>, vector<32xi16>, vector<32xi32>
    if (laneSize == 32 && (resultElWidth == 32 || resultElWidth == 8)) {
      if (resultElWidth == 32) {
        auto lhs = dyn_cast<arith::ExtSIOp>(adaptor.getLhs().getDefiningOp());
        auto rhs = dyn_cast<arith::ExtSIOp>(adaptor.getRhs().getDefiningOp());

        if (!lhs || !rhs)
          return failure();

        auto lval = lhs->getOperand(0);
        auto rval = rhs->getOperand(0);

        VectorType lSrcType = cast<VectorType>(lval.getType());
        VectorType rSrcType = cast<VectorType>(rval.getType());

        unsigned lBitWidth = lSrcType.getElementType().getIntOrFloatBitWidth();
        unsigned rBitWidth = rSrcType.getElementType().getIntOrFloatBitWidth();

        if ((lBitWidth != 8 || rBitWidth != 8) &&
            (lBitWidth != 16 || rBitWidth != 16))
          return failure();

        aievec::MulElemOp mulElemOp = nullptr;
        if (lBitWidth == 8) {
          mulElemOp = createMulElemAieML(rewriter, lval, rval, lSrcType,
                                         lBitWidth, mulOp.getLoc());
        } else {
          Type accType = getVectorOpDestType(lSrcType, /*AIEML =*/true);
          mulElemOp = rewriter.create<aievec::MulElemOp>(mulOp.getLoc(),
                                                         accType, lval, rval);
        }
        rewriter.replaceOpWithNewOp<aievec::CastOp>(
            mulOp, resultType, mulElemOp.getResult(), /*isResAcc*/ false);
        // Case 3:
        // Transfer -
        // %1 = arith muli %a, %b : vector<32xi8>
        // to -
        // aievec.mul_elem(%a, %b) : vector<64xi8>, vector<64xi8>,
        // vector<32xi32>
      } else {
        auto lval = adaptor.getLhs();
        auto rval = adaptor.getRhs();
        VectorType srcType = cast<VectorType>(lval.getType());
        unsigned bitWidth = srcType.getElementType().getIntOrFloatBitWidth();
        auto mulElemOp = createMulElemAieML(rewriter, lval, rval, srcType,
                                            bitWidth, mulOp.getLoc());
        rewriter.replaceOpWithNewOp<aievec::SRSOp>(
            mulOp, srcType, mulElemOp.getResult(), shiftParam);
      }
    } else {
      Type accType = getVectorOpDestType(cast<VectorType>(mulOp.getType()),
                                         /*AIEML =*/true);

      auto mulElemOp = rewriter.create<aievec::MulElemOp>(
          mulOp.getLoc(), accType, adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          mulOp, resultType, mulElemOp.getResult(), shiftParam);
    }
    return success();
  }

  unsigned shiftParam;
};

// Return the offset of a given transfer read operation with regards to the
// specified vector type. If the read is aligned size of the vector type, then
// the offset is 0. Otherwise, the offset is the number of elements past the
// immediately preceding aligned address.
template <
    typename TransferReadLikeOp,
    typename = std::enable_if_t<
        std::is_same_v<TransferReadLikeOp, vector::TransferReadOp> ||
        std::is_same_v<TransferReadLikeOp, vector::TransferReadOp::Adaptor>>>
static unsigned getTransferReadAlignmentOffset(TransferReadLikeOp op,
                                               VectorType vType) {
  // TODO: Add support for cases where the index is not comming from an
  // TODO: `affine.apply` op. E.g.: when the index is a constant.
  auto innerMostIndex = op.getIndices().back();
  auto vectorLength = vType.getShape().back();
  if (auto defOp = innerMostIndex.getDefiningOp())
    if (auto applyOp = dyn_cast<AffineApplyOp>(defOp))
      if (applyOp.getAffineMap().getNumDims() == 1)
        return applyOp.getAffineMap().compose(ArrayRef<int64_t>{0})[0] %
               vectorLength;
  return 0;
}

// This pattern converts a `vector.transfer_read` with an unaligned access
// into an aligned `vector.transfer_read` twice as long, followed by a
// `vector.extract_strided_slice` selecting the subvector matching the
// original `vector.transfer_read`.
struct SplitUnalignedTransferReadPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check that it's not a splat transfer read.
    if (adaptor.getPermutationMap().isConstant())
      return failure();

    // Check if the transfer is unaligned.
    auto vType = readOp.getVectorType();
    unsigned offset = getTransferReadAlignmentOffset(adaptor, vType);
    if (offset == 0)
      return failure();

    // Create an aligned transfer read

    // Calculate the aligned indices for the lower and higher parts.
    // TODO: Add support for cases where the offset is greater than the
    // TODO: vector length.
    auto lowIdx =
        dyn_cast<AffineApplyOp>(adaptor.getIndices().back().getDefiningOp())
            .getMapOperands()[0];
    auto vLen = vType.getShape().back();
    auto longVecTy = VectorType::get(2 * vLen, vType.getElementType());
    SmallVector<Value, 8> alignedIdx;
    alignedIdx.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
    alignedIdx[alignedIdx.size() - 1] = lowIdx;

    // Create the aligned transfer read for a vector 2x as long that covers the
    // elements of the unaligned vector.
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), longVecTy, adaptor.getSource(), alignedIdx,
        adaptor.getPadding());

    // Create a `vector.extract_strided_slice` to extract the unaligned vector.
    rewriter.replaceOpWithNewOp<vector::ExtractStridedSliceOp>(
        readOp, newReadOp.getResult(), offset, vLen, 1);

    return success();
  }
};

// This pattern converts a `vector.transfer_read` with a splat permutation map
// into a contiguous `vector.transfer_read` followed by a `vector.extract` to
// obtain the splat value and a `vector.broadcast` to broadcast it into a
// vector of the right size.
struct ConvertSplatTransferReadToBroadcastPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AffineMap map = readOp.getPermutationMap();
    if (!map.isConstant())
      return failure();

    // If the innermost index comes from an `affine.apply` op, take the base
    // as the new innermost index for the new `vector.transfer_read`, and the
    // offset as the index for the `aievec.broadcast` op.
    SmallVector<Value, 8> indices;
    indices.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
    Value innerMostIdx = indices[indices.size() - 1];
    Value newIdx = innerMostIdx;
    int64_t offset = 0;
    if (auto defOp = innerMostIdx.getDefiningOp())
      if (auto applyOp = dyn_cast<AffineApplyOp>(defOp))
        if (applyOp.getAffineMap().getNumDims() == 1) {
          newIdx = applyOp.getMapOperands()[0];
          offset = applyOp.getAffineMap().compose(ArrayRef<int64_t>{0})[0];
        }
    // XXX: We assume we are reading 1D vectors
    int64_t vlen = readOp.getVector().getType().getShape()[0];
    if (offset >= vlen) {
      // If the splat element is beyond the first vector, we calculate the
      // address of the vector containing the element.
      int64_t numElemsToSkip = vlen * (offset / vlen);
      offset = offset % vlen;
      auto newAddrMap = AffineMap::get(
          1, 0, getAffineDimExpr(0, readOp.getContext()) + numElemsToSkip);
      newIdx = rewriter
                   .create<AffineApplyOp>(readOp.getLoc(), newAddrMap,
                                          SmallVector<Value, 1>({newIdx}))
                   .getResult();
    }
    indices[indices.size() - 1] = newIdx;
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), readOp.getVector().getType(), adaptor.getSource(),
        indices, adaptor.getPadding());
    auto extractOp = rewriter.create<vector::ExtractOp>(
        readOp.getLoc(), newReadOp.getResult(), ArrayRef<int64_t>{offset});
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        readOp, newReadOp.getVector().getType(), extractOp.getResult());
    return success();
  }
};

static SmallVector<NamedAttribute>
buildFMAOpSplatAttrForElemTy(aievec::FMAOp fmaOp, int64_t bcastPos,
                             int64_t step = 1) {
  unsigned width = 0;
  auto elemTy = fmaOp.getLhs().getType().getElementType();
  auto intTy = dyn_cast<IntegerType>(elemTy);
  if (intTy)
    width = intTy.getWidth();
  auto ctx = fmaOp.getContext();
  switch (width) {
  case 16:
    // NOTE: The pattern is:
    //       acc[0]  = x[0]  * z[bcastPos] + x[16] * z[bcastPos+step]
    //       acc[1]  = x[1]  * z[bcastPos] + x[17] * z[bcastPos+step]
    //       acc[2]  = x[2]  * z[bcastPos] + x[18] * z[bcastPos+step]
    //       acc[3]  = x[3]  * z[bcastPos] + x[19] * z[bcastPos+step]
    //       acc[4]  = x[4]  * z[bcastPos] + x[20] * z[bcastPos+step]
    //       acc[5]  = x[5]  * z[bcastPos] + x[21] * z[bcastPos+step]
    //       acc[6]  = x[6]  * z[bcastPos] + x[22] * z[bcastPos+step]
    //       acc[7]  = x[7]  * z[bcastPos] + x[23] * z[bcastPos+step]
    //       acc[8]  = x[8]  * z[bcastPos] + x[24] * z[bcastPos+step]
    //       acc[9]  = x[9]  * z[bcastPos] + x[25] * z[bcastPos+step]
    //       acc[10] = x[10] * z[bcastPos] + x[26] * z[bcastPos+step]
    //       acc[11] = x[11] * z[bcastPos] + x[27] * z[bcastPos+step]
    //       acc[12] = x[12] * z[bcastPos] + x[28] * z[bcastPos+step]
    //       acc[13] = x[13] * z[bcastPos] + x[29] * z[bcastPos+step]
    //       acc[14] = x[14] * z[bcastPos] + x[30] * z[bcastPos+step]
    //       acc[15] = x[15] * z[bcastPos] + x[31] * z[bcastPos+step]
    return SmallVector<NamedAttribute, 11>(
        {{fmaOp.getXstartAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getXoffsetsAttrName(), StringAttr::get(ctx, "0x73727170")},
         {fmaOp.getXoffsetsHiAttrName(), StringAttr::get(ctx, "0x77767574")},
         {fmaOp.getXstepAttrName(), fmaOp.getXstepAttr()},
         {fmaOp.getXsquareAttrName(), StringAttr::get(ctx, "0x3120")},
         {fmaOp.getZstartAttrName(),
          StringAttr::get(ctx, std::to_string(bcastPos))},
         {fmaOp.getZoffsetsAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getZoffsetsHiAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getZstepAttrName(), StringAttr::get(ctx, std::to_string(step))},
         {fmaOp.getZsquareAttrName(), fmaOp.getZsquareAttr()},
         {fmaOp.getFmsubAttrName(), fmaOp.getFmsubAttr()}});
  case 32:
    return SmallVector<NamedAttribute, 11>(
        {{fmaOp.getXstartAttrName(), StringAttr::get(ctx, "0")},
         {fmaOp.getXoffsetsAttrName(), StringAttr::get(ctx, "0x76543210")},
         {fmaOp.getXoffsetsHiAttrName(), fmaOp.getXoffsetsHiAttr()},
         {fmaOp.getXstepAttrName(), fmaOp.getXstepAttr()},
         {fmaOp.getXsquareAttrName(), fmaOp.getXsquareAttr()},
         {fmaOp.getZstartAttrName(),
          StringAttr::get(ctx, std::to_string(bcastPos))},
         {fmaOp.getZoffsetsAttrName(), StringAttr::get(ctx, "0x00000000")},
         {fmaOp.getZoffsetsHiAttrName(), fmaOp.getZoffsetsHiAttr()},
         {fmaOp.getZstepAttrName(), fmaOp.getZstepAttr()},
         {fmaOp.getZsquareAttrName(), fmaOp.getZsquareAttr()},
         {fmaOp.getFmsubAttrName(), fmaOp.getFmsubAttr()}});
  }
  return {};
}

// template <typename T>
// concept AIEv1MACLikeOp = std::same_as<T, aievec::FMAOp> || std::same_as<T,
// aievec::FMAOp::Adaptor>;

template <typename AIEv1MACLikeOp,
          typename = std::enable_if_t<
              std::is_same_v<AIEv1MACLikeOp, aievec::FMAOp> ||
              std::is_same_v<AIEv1MACLikeOp, aievec::FMAOp::Adaptor>>>
static bool isSingleColumnInt16VectorTimesScalarMac(AIEv1MACLikeOp fmaOp) {
  // lhs is a 32xi16 vector
  VectorType lhsVTy = cast<VectorType>(fmaOp.getLhs().getType());
  auto intTy = dyn_cast<IntegerType>(lhsVTy.getElementType());
  if (!intTy || intTy.getWidth() != 16)
    return false;
  if (lhsVTy.getShape()[0] != 32)
    return false;
  // Attributes match a Vector x Scalar mac
  if (fmaOp.getXoffsets() != "0x73727170" ||
      fmaOp.getXoffsetsHi() != "0x77767574" || fmaOp.getXstart() != "0" ||
      fmaOp.getXsquare() != "0x3120" || fmaOp.getZoffsets() != "0" ||
      fmaOp.getZoffsetsHi() != "0" || fmaOp.getZstep() != "1")
    return false;
  // lhs op is a concat of a vector and a dense<0> constant vector
  if (!fmaOp.getLhs().getDefiningOp())
    return false;
  aievec::ConcatOp concatOp =
      dyn_cast<aievec::ConcatOp>(fmaOp.getLhs().getDefiningOp());
  if (!concatOp)
    return false;
  auto tailVec = concatOp.getSources()[1];
  if (!tailVec.getDefiningOp())
    return false;
  auto constOp = dyn_cast<arith::ConstantOp>(tailVec.getDefiningOp());
  if (!constOp)
    return false;
  auto cstDense = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
  if (!cstDense)
    return false;
  return llvm::all_of(cstDense, [](const APInt &val) { return val == 0; });
}

// This pattern folds an extract + broadcast feeding into an `aievec::FMAOp`
// into the op, using the shuffle attributes.
struct FoldBroadcastToFMAOp : public OpConversionPattern<aievec::FMAOp> {
  using OpConversionPattern<aievec::FMAOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aievec::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto concatOp =
        dyn_cast<aievec::ConcatOp>(adaptor.getLhs().getDefiningOp());
    if (!concatOp)
      return failure();
    vector::BroadcastOp bcastOp = nullptr;
    auto concatDefOp = concatOp.getSources()[0].getDefiningOp();
    if (concatDefOp)
      bcastOp = dyn_cast<vector::BroadcastOp>(concatDefOp);
    Value lhs = adaptor.getRhs();
    if (!bcastOp) {
      bcastOp = dyn_cast<vector::BroadcastOp>(adaptor.getRhs().getDefiningOp());
      if (!bcastOp)
        return failure();
      lhs = concatOp.getSources()[0];
    }
    auto extOp =
        dyn_cast<vector::ExtractOp>(bcastOp.getSource().getDefiningOp());
    if (!extOp)
      return failure();

    auto rhs = extOp.getVector();
    auto concatVecType = cast<VectorType>(concatOp.getResult().getType());
    auto zvec = rewriter.create<arith::ConstantOp>(
        concatOp.getLoc(), lhs.getType(), rewriter.getZeroAttr(lhs.getType()));
    auto lhsX2 =
        rewriter
            .create<aievec::ConcatOp>(concatOp.getLoc(), concatVecType,
                                      SmallVector<Value, 2>({lhs, zvec}))
            .getResult();
    // XXX: We assume a 1D vector
    auto pos = extOp.getPosition();
    int64_t zstart = cast<IntegerAttr>(pos[0]).getInt();
    auto fmaOpAttr = buildFMAOpSplatAttrForElemTy(fmaOp, zstart);
    rewriter.replaceOpWithNewOp<aievec::FMAOp>(
        fmaOp, TypeRange({fmaOp.getResult().getType()}),
        ValueRange({lhsX2, rhs, adaptor.getAcc()}), fmaOpAttr);

    return success();
  }
};

struct ConvertMulAddToAIEVecFMAOpPattern
    : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType vecType = dyn_cast<VectorType>(addOp.getType());
    if (!vecType)
      return failure();

    auto res =
        extractMACOperandsFromAddOperands(adaptor.getLhs(), adaptor.getRhs());
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    SmallVector<int64_t, 4> concatVecShape(vecType.getShape().begin(),
                                           vecType.getShape().end());
    concatVecShape[vecType.getRank() - 1] *= 2;
    auto concatVecType =
        VectorType::get(concatVecShape, vecType.getElementType());
    Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
                                       /*AIEML =*/false);
    auto lhsX2 = rewriter
                     .create<aievec::ConcatOp>(addOp.getLoc(), concatVecType,
                                               SmallVector<Value, 2>(2, lhs))
                     .getResult();
    auto upsOp = rewriter.create<aievec::UPSOp>(addOp.getLoc(), accType, acc);
    auto fmaOp = rewriter.create<aievec::FMAOp>(
        addOp.getLoc(), accType, lhsX2, rhs, upsOp.getResult(),
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xstep=*/"",
        /*xsquare=*/"", /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"",
        /*zstep=*/"", /*zsquare=*/"", /*fmsub=*/false);
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(addOp, vecType,
                                               fmaOp.getResult());
    return success();
  }
};

// This pattern replaces `vector.transfer_read` with `aievec.upd`. Right now,
// it performs a naïve direct translation. This needs to be expanded to
// support more complex scenarios.
struct LowerVectorTransferReadToAIEUPD
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LowerVectorTransferReadToAIEUPD(MLIRContext *context, AnalysisManager &am,
                                  int32_t maxVectorSize = 256)
      : OpConversionPattern<vector::TransferReadOp>(context), am(am),
        maxVectorSize(maxVectorSize) {}

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // == Handle invalid read operations ==
    // Masked loads
    if (readOp.getMask())
      return readOp.emitError() << "AIE doesn't support masked loads.";

    // Non-contiguous loads
    AffineMap map = readOp.getPermutationMap();
    if (!map.isMinorIdentity())
      return failure();

    // Splats
    if (map.isConstant())
      return failure();

    // When a transfer read with a constant innermost index is not aligned, we
    // get the corresponding aligned load followed by an aievec.shift op.
    // Example:
    // Convert -
    // %0 = vector.transfer_read %arg1[16] : vector<32xi8>
    // %1 = vector.transfer_read %arg1[34] : vector<32xi8>
    //
    // to -
    //
    // %0 = aievec.upd %arg1[0] : vector<32xi8>
    // %1 = aievec.upd %arg1[32] : vector<32xi8>
    // %2 = aievec.shift %0, %1 {shift = 16 : i32} : vector<32xi8>
    // %3 = aievec.upd %arg1[64] : vector<32xi8>
    // %4 = aievec.shift %2, %3 {shift = 2 : i32} : vector<32xi8>
    //
    SmallVector<Value, 4> indices(adaptor.getIndices().begin(),
                                  adaptor.getIndices().end());
    Value innerMostIdx = indices[indices.size() - 1];
    Value newIdx = innerMostIdx;
    VectorType vType = readOp.getVector().getType().cast<VectorType>();
    int32_t lanes = getVectorLaneSize(vType);

    if (auto defOp = innerMostIdx.getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        int64_t val = constOp.getValue().cast<IntegerAttr>().getInt();
        if (val) {
          int64_t offset = val % lanes;
          int64_t idx = val / lanes * lanes;
          newIdx = rewriter.create<arith::ConstantOp>(
              constOp.getLoc(),
              rewriter.getIntegerAttr(constOp.getType(), idx));
          indices[indices.size() - 1] = newIdx;
          int32_t shiftBytes = offset * getElementSizeInBits(vType) / 8;

          if (shiftBytes) {
            auto updOp = rewriter.create<xilinx::aievec::UPDOp>(
                readOp.getLoc(), vType, adaptor.getSource(), indices, 0, 0,
                TypedValue<VectorType>(nullptr));
            newIdx = rewriter.create<arith::ConstantOp>(
                constOp.getLoc(),
                rewriter.getIntegerAttr(constOp.getType(), idx + lanes));
            indices[indices.size() - 1] = newIdx;
            // Load the next vector lanes
            auto nextUpdOp = rewriter.create<xilinx::aievec::UPDOp>(
                readOp.getLoc(), vType, adaptor.getSource(), indices, 0, 0,
                TypedValue<VectorType>(nullptr));

            arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(
                readOp.getLoc(), rewriter.getI32IntegerAttr(shiftBytes));
            rewriter.replaceOpWithNewOp<xilinx::aievec::ShiftOp>(
                readOp, vType, updOp->getResult(0), nextUpdOp->getResult(0),
                constOp.getResult());
          } else {
            rewriter.replaceOpWithNewOp<xilinx::aievec::UPDOp>(
                readOp, vType, adaptor.getSource(), indices, 0, 0,
                TypedValue<VectorType>(nullptr));
          }
          return success();
        }
      }
    }
    rewriter.replaceOpWithNewOp<xilinx::aievec::UPDOp>(
        readOp, vType, adaptor.getSource(), indices, 0, 0,
        TypedValue<VectorType>(nullptr));
    return success();
  }

  AnalysisManager &am;
  int32_t maxVectorSize;
};

// XXX: Notice that this template doesn't verify that the vector element type
// XXX: is supported by the target architecture.
template <typename SrcOpTy, typename DstOpTy>
struct OneToOneVectorOpToAIEVecOpPattern : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOpTy>(
        srcOp, srcOp.getResult().getType(), adaptor.getLhs(), adaptor.getRhs(),
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xsquare=*/"",
        /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"", /*zsquare=*/"");
    return success();
  }
};

struct LowerVectorAddIOpToAIEVecAddOp
    : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddIOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = addOp.getType();
    if (!isa<VectorType>(resType))
      return failure();

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsDefOp = lhs.getDefiningOp();
    auto rhsDefOp = rhs.getDefiningOp();
    if ((lhsDefOp && isa<arith::MulIOp>(lhsDefOp)) ||
        (rhsDefOp && isa<arith::MulIOp>(rhsDefOp)))
      return failure();

    rewriter.replaceOpWithNewOp<aievec::AddOp>(
        addOp, resType, lhs, rhs,
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xsquare=*/"",
        /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"", /*zsquare=*/"");
    return success();
  }
};

using LowerVectorAddFOpToAIEVecAddOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::AddFOp, aievec::AddOp>;
using LowerVectorMulIOpToAIEVecMulOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::MulIOp, aievec::MulOp>;
using LowerVectorMulFOpToAIEVecMulOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::MulFOp, aievec::MulOp>;
using LowerVectorSubIOpToAIEVecSubOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::SubIOp, aievec::SubOp>;
using LowerVectorSubFOpToAIEVecSubOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::SubFOp, aievec::SubOp>;

template <typename SrcOpTy, typename AIEv2ElemOp>
static LogicalResult genAddElemAieML(ConversionPatternRewriter &rewriter,
                                     Value lval, Value rval, VectorType srcType,
                                     SrcOpTy srcOp) {
  auto lCastOp = rewriter.create<aievec::CastOp>(srcOp.getLoc(), srcType, lval,
                                                 /*isResAcc*/ true);
  auto rCastOp = rewriter.create<aievec::CastOp>(srcOp.getLoc(), srcType, rval,
                                                 /*isResAcc*/ true);
  auto elemOp = rewriter.create<AIEv2ElemOp>(
      srcOp.getLoc(), lCastOp->getResult(0).getType(), lCastOp->getResult(0),
      rCastOp->getResult(0));
  rewriter.replaceOpWithNewOp<aievec::CastOp>(
      srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);
  return success();
}

template <typename SrcOpTy, typename DstOpTy>
struct LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp
    : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp(MLIRContext *context)
      : OpConversionPattern<SrcOpTy>(context) {}

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    // A set recording the vector lane size and element width we are supporting
    // for aie-ml.
    llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
    laneSizeElWidthPairSet.insert({64, 8});
    laneSizeElWidthPairSet.insert({32, 16});
    laneSizeElWidthPairSet.insert({16, 32});
    laneSizeElWidthPairSet.insert({32, 32});

    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsDefOp = lhs.getDefiningOp();
    auto rhsDefOp = rhs.getDefiningOp();
    if ((lhsDefOp && isa<arith::MulIOp>(lhsDefOp)) ||
        (rhsDefOp && isa<arith::MulIOp>(rhsDefOp)) ||
        (lhsDefOp && isa<arith::MulFOp>(lhsDefOp)) ||
        (rhsDefOp && isa<arith::MulFOp>(rhsDefOp)))
      return failure();

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (scalarType.isa<IntegerType>()) {
      if (!laneSizeElWidthPairSet.count(
              std::make_pair(laneSize, resultElWidth)))
        return failure();

      if (laneSize == 32 && resultElWidth == 32) {
        if (!lhsDefOp || !rhsDefOp) {
          return genAddElemAieML<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                   resultType, srcOp);
        }
        auto lhsExt = dyn_cast<arith::ExtSIOp>(lhsDefOp);
        auto rhsExt = dyn_cast<arith::ExtSIOp>(rhsDefOp);
        if (!lhsExt || !rhsExt) {
          return genAddElemAieML<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                   resultType, srcOp);
        }
        auto lval = lhsExt->getOperand(0);
        auto rval = rhsExt->getOperand(0);

        VectorType lSrcType = cast<VectorType>(lval.getType());
        VectorType rSrcType = cast<VectorType>(rval.getType());

        unsigned lBitWidth = lSrcType.getElementType().getIntOrFloatBitWidth();
        unsigned rBitWidth = rSrcType.getElementType().getIntOrFloatBitWidth();

        if ((lBitWidth != 8 || rBitWidth != 8) &&
            (lBitWidth != 16 || rBitWidth != 16)) {
          return genAddElemAieML<SrcOpTy, DstOpTy>(rewriter, lhs, rhs,
                                                   resultType, srcOp);
        }

        Type accType = getVectorOpDestType(lSrcType, /*AIEML =*/true);
        auto lUpsOp =
            rewriter.create<aievec::UPSOp>(srcOp.getLoc(), accType, lval);
        auto rUpsOp =
            rewriter.create<aievec::UPSOp>(srcOp.getLoc(), accType, rval);
        auto elemOp = rewriter.create<DstOpTy>(
            srcOp.getLoc(), lUpsOp->getResult(0).getType(),
            lUpsOp->getResult(0), rUpsOp->getResult(0));
        rewriter.replaceOpWithNewOp<aievec::CastOp>(
            srcOp, srcOp.getType(), elemOp.getResult(), /*isResAcc*/ false);
        return success();
      } else {
        rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(), lhs, rhs);
        return success();
      }
    } // Float types
    else {
      if (laneSize != 16)
        return failure();

      // v16float
      if (resultElWidth == 32) {
        return genAddElemAieML<SrcOpTy, DstOpTy>(rewriter, lhs, rhs, resultType,
                                                 srcOp);
      }
      // v16bfloat16
      Type accType = getVectorOpDestType(resultType, /*AIEML =*/true);
      auto lUpsOp =
          rewriter.create<aievec::UPSOp>(srcOp.getLoc(), accType, lhs);
      auto rUpsOp =
          rewriter.create<aievec::UPSOp>(srcOp.getLoc(), accType, rhs);
      auto elemOp = rewriter.create<DstOpTy>(
          srcOp.getLoc(), lUpsOp->getResult(0).getType(), lUpsOp->getResult(0),
          rUpsOp->getResult(0));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(srcOp, srcOp.getType(),
                                                 elemOp.getResult());
      return success();
    }
    return failure();
  }
};

using LowerVectorAddIOpToAIEVecAddElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::AddIOp,
                                                    aievec::AddElemOp>;
using LowerVectorSubIOpToAIEVecSubElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::SubIOp,
                                                    aievec::SubElemOp>;
using LowerVectorAddFOpToAIEVecAddElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::AddFOp,
                                                    aievec::AddElemOp>;
using LowerVectorSubFOpToAIEVecSubElemOp =
    LowerVectorAddOrSubOpToAIEVecAddElemOrSubElemOp<arith::SubFOp,
                                                    aievec::SubElemOp>;

template <typename SrcOpTy, typename DstOpTy>
struct LowerVectorMinMaxOpToAIEVecMinMaxOp
    : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LowerVectorMinMaxOpToAIEVecMinMaxOp(MLIRContext *context)
      : OpConversionPattern<SrcOpTy>(context) {}

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    // A set recording the element width we are supporting for aie-ml.
    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (!(elWidthSet.count(resultElWidth) && laneSize * resultElWidth == 512))
      return failure();

    rewriter.replaceOpWithNewOp<DstOpTy>(srcOp, srcOp.getType(),
                                         adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

using LowerVectorMinSIOpToAIEVecMinOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MinSIOp, aievec::MinOp>;
using LowerVectorMaxSIOpToAIEVecMaxOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MaxSIOp, aievec::MaxOp>;
using LowerVectorMinFOpToAIEVecMinOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MinFOp, aievec::MinOp>;
using LowerVectorMaxFOpToAIEVecMaxOp =
    LowerVectorMinMaxOpToAIEVecMinMaxOp<arith::MaxFOp, aievec::MaxOp>;

static arith::CmpIPredicate
convertToIntegerPredicate(arith::CmpFPredicate pred) {
  switch (pred) {
  case CmpFPredicate::UEQ:
  case CmpFPredicate::OEQ:
    return CmpIPredicate::eq;
  case CmpFPredicate::UGT:
    return CmpIPredicate::ugt;
  case CmpFPredicate::OGT:
    return CmpIPredicate::sgt;
  case CmpFPredicate::UGE:
    return CmpIPredicate::uge;
  case CmpFPredicate::OGE:
    return CmpIPredicate::sge;
  case CmpFPredicate::ULT:
    return CmpIPredicate::ult;
  case CmpFPredicate::OLT:
    return CmpIPredicate::slt;
  case CmpFPredicate::ULE:
    return CmpIPredicate::ule;
  case CmpFPredicate::OLE:
    return CmpIPredicate::sle;
  case CmpFPredicate::UNE:
  case CmpFPredicate::ONE:
    return CmpIPredicate::ne;
  default:
    llvm_unreachable("Unexpected predicate!");
  }
}

static arith::CmpIPredicate
convertToIntegerPredicate(arith::CmpIPredicate pred) {
  return pred;
}

static aievec::CmpOp createCmpOpAieML(ConversionPatternRewriter &rewriter,
                                      CmpIPredicate pred, Location loc,
                                      Type type, Value lhs, Value rhs) {
  switch (pred) {
  case CmpIPredicate::eq:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "eq");
  case CmpIPredicate::ne:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "ne");
  case CmpIPredicate::slt:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "slt");
  case CmpIPredicate::ult:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "ult");
  case CmpIPredicate::sle:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "sle");
  case CmpIPredicate::ule:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "ule");
  case CmpIPredicate::sgt:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "sgt");
  case CmpIPredicate::ugt:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "ugt");
  case CmpIPredicate::sge:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "sge");
  case CmpIPredicate::uge:
    return rewriter.create<aievec::CmpOp>(loc, type, lhs, rhs, "uge");
  }
  return nullptr;
}

template <typename SrcOpTy, typename CmpTy>
struct LowerVectorCmpOpToAIEVecCmpOp : public OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LowerVectorCmpOpToAIEVecCmpOp(MLIRContext *context)
      : OpConversionPattern<SrcOpTy>(context) {}

  LogicalResult
  matchAndRewrite(SrcOpTy srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType lhsType = dyn_cast<VectorType>(srcOp.getLhs().getType());
    if (!lhsType)
      return failure();

    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = lhsType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);

    if (!(elWidthSet.count(elWidth) && laneSize * elWidth == 512))
      return failure();

    // Unsigned int and unsigned long long are acceptable type.
    Type type =
        mlir::IntegerType::get(srcOp.getContext(), laneSize <= 32 ? 32 : 64,
                               mlir::IntegerType::Unsigned);

    Location loc = srcOp.getLoc();
    Value lhs = srcOp.getLhs();
    Value rhs = srcOp.getRhs();
    CmpTy pred = srcOp.getPredicate();

    arith::CmpIPredicate ipred = convertToIntegerPredicate(pred);

    aievec::CmpOp aieCmpOp =
        createCmpOpAieML(rewriter, ipred, loc, type, lhs, rhs);

    if (!aieCmpOp)
      return failure();

    VectorType resultType = dyn_cast<VectorType>(srcOp.getResult().getType());
    // Convert vector i1 type to unsigned interger type by built-in unrealized
    // conversion cast op.
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        srcOp, resultType, aieCmpOp.getResult());

    return success();
  }
};

using LowerVectorCmpIOpToAIEVecCmpOp =
    LowerVectorCmpOpToAIEVecCmpOp<arith::CmpIOp, CmpIPredicate>;
using LowerVectorCmpFOpToAIEVecCmpOp =
    LowerVectorCmpOpToAIEVecCmpOp<arith::CmpFOp, CmpFPredicate>;

struct LowerVectorSelectOpToAIEVecSelOp
    : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = dyn_cast<VectorType>(srcOp.getType());
    if (!resultType)
      return failure();

    llvm::SmallSet<unsigned, 16> elWidthSet;
    elWidthSet.insert(8);
    elWidthSet.insert(16);
    elWidthSet.insert(32);

    Type scalarType = resultType.getElementType();
    unsigned resultElWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (!(elWidthSet.count(resultElWidth) && laneSize * resultElWidth == 512))
      return failure();

    Type type =
        mlir::IntegerType::get(srcOp.getContext(), laneSize <= 32 ? 32 : 64,
                               mlir::IntegerType::Unsigned);

    auto convertOp = rewriter.create<UnrealizedConversionCastOp>(
        srcOp.getLoc(), type, adaptor.getCondition());

    rewriter.replaceOpWithNewOp<aievec::SelOp>(
        srcOp, srcOp.getResult().getType(), srcOp.getTrueValue(),
        srcOp.getFalseValue(), convertOp.getResult(0));

    return success();
  }
};

template <typename DstOpTy>
static void generateAIEVecOpsForReductionOp(ConversionPatternRewriter &rewriter,
                                            vector::ReductionOp srcOp,
                                            int shiftIndex) {
  assert(shiftIndex > 0 && (shiftIndex & (shiftIndex - 1)) == 0 &&
         "shiftIndex must be power of 2");

  Location loc = srcOp.getLoc();
  Value curValue = srcOp.getVector();
  VectorType vType = dyn_cast<VectorType>(curValue.getType());
  Type scalarType = vType.getElementType();
  Type vecType = curValue.getType();
  DstOpTy curOp = nullptr;
  unsigned elWidth = scalarType.getIntOrFloatBitWidth();

  for (int id = shiftIndex; id > 0; id /= 2) {
    arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(id * elWidth / 8));

    auto shiftBytesOp = rewriter.create<aievec::ShiftOp>(
        loc, vecType, curValue, curValue, constOp.getResult());

    curOp = rewriter.create<DstOpTy>(loc, vecType, curValue,
                                     shiftBytesOp.getResult());

    curValue = curOp.getResult();
  }

  arith::ConstantOp zeroConstOp =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(srcOp, scalarType, curOp,
                                                 zeroConstOp.getResult());
  return;
}

static void generateReductionOpForFloat(ConversionPatternRewriter &rewriter,
                                        vector::ReductionOp srcOp,
                                        int shiftIndex) {
  assert(shiftIndex > 0 && (shiftIndex & (shiftIndex - 1)) == 0 &&
         "shiftIndex must be power of 2");

  Location loc = srcOp.getLoc();
  Value curValue = srcOp.getVector();
  VectorType vType = dyn_cast<VectorType>(curValue.getType());

  Type scalarType = vType.getElementType();
  aievec::CastOp curOp = nullptr;
  unsigned elWidth = scalarType.getIntOrFloatBitWidth();
  assert(elWidth == 32 && "scalar type should be float");

  for (int id = shiftIndex; id > 0; id /= 2) {
    arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(id * elWidth / 8));

    auto shiftBytesOp = rewriter.create<aievec::ShiftOp>(
        loc, vType, curValue, curValue, constOp.getResult());

    auto lCastOp = rewriter.create<aievec::CastOp>(loc, vType, curValue,
                                                   /*isResAcc*/ true);
    auto rCastOp =
        rewriter.create<aievec::CastOp>(loc, vType, shiftBytesOp.getResult(),
                                        /*isResAcc*/ true);
    auto elemOp = rewriter.create<aievec::AddElemOp>(
        loc, lCastOp.getResult().getType(), lCastOp.getResult(),
        rCastOp.getResult());
    curOp = rewriter.create<aievec::CastOp>(loc, vType, elemOp.getResult(),
                                            /*isResAcc*/ false);

    curValue = curOp.getResult();
  }

  arith::ConstantOp zeroConstOp =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(srcOp, scalarType, curOp,
                                                 zeroConstOp.getResult());
}

static void generateReductionOpForBFloat16(ConversionPatternRewriter &rewriter,
                                           vector::ReductionOp srcOp,
                                           int shiftIndex) {
  assert(shiftIndex > 0 && (shiftIndex & (shiftIndex - 1)) == 0 &&
         "shiftIndex must be power of 2");

  Value curValue = srcOp.getVector();
  VectorType vType = dyn_cast<VectorType>(curValue.getType());

  Type scalarType = vType.getElementType();
  Location loc = srcOp.getLoc();
  Type accType = getVectorOpDestType(vType, /*AIEML =*/true);
  unsigned accWidth =
      dyn_cast<VectorType>(accType).getElementType().getIntOrFloatBitWidth();

  auto upsOp = rewriter.create<aievec::UPSOp>(loc, accType, srcOp.getVector());

  curValue = upsOp.getResult();
  unsigned laneSize = getVectorLaneSize(vType);

  VectorType vecType = createVectorType(2 * laneSize, scalarType);
  aievec::AddElemOp curOp = nullptr;

  for (int id = shiftIndex; id > 0; id /= 2) {
    arith::ConstantOp constOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(id * accWidth / 8));
    auto shiftBytesOp = rewriter.create<aievec::ShiftOp>(
        loc, accType, curValue, curValue, constOp, true);
    curOp = rewriter.create<aievec::AddElemOp>(loc, accType, curValue,
                                               shiftBytesOp.getResult());
    curValue = curOp.getResult();
  }

  auto srsOp = rewriter.create<aievec::SRSOp>(loc, vType, curOp.getResult());
  SmallVector<Value> concatSources = {srsOp.getResult(), srsOp.getResult()};
  auto concatOp =
      rewriter.create<aievec::ConcatOp>(loc, vecType, concatSources);

  arith::ConstantOp zeroConstOp =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  rewriter.replaceOpWithNewOp<aievec::ExtElemOp>(srcOp, scalarType, concatOp,
                                                 zeroConstOp.getResult());
}

struct LowerVectorReductionOp
    : public OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern<vector::ReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::ReductionOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto kind = srcOp.getKind();

    if (kind != vector::CombiningKind::ADD &&
        kind != vector::CombiningKind::MINSI &&
        kind != vector::CombiningKind::MINUI &&
        kind != vector::CombiningKind::MINF &&
        kind != vector::CombiningKind::MAXSI &&
        kind != vector::CombiningKind::MAXUI &&
        kind != vector::CombiningKind::MAXF) {
      return failure();
    }

    VectorType vType = dyn_cast<VectorType>(srcOp.getVector().getType());
    if (!vType)
      return failure();

    // A set recording the vector lane size and element width we are
    // supporting for aie-ml.
    llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
    laneSizeElWidthPairSet.insert({64, 8});
    laneSizeElWidthPairSet.insert({32, 16});
    laneSizeElWidthPairSet.insert({16, 32});

    Type scalarType = vType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(vType);

    if (scalarType.isa<IntegerType>() &&
        !laneSizeElWidthPairSet.count(std::make_pair(laneSize, elWidth)))
      return failure();

    if (scalarType.isa<FloatType>() && laneSize != 16 && laneSize != 32)
      return failure();

    int shiftIndex = laneSize / 2;

    // Reduction minimum
    if (kind == vector::CombiningKind::MINSI ||
        kind == vector::CombiningKind::MINUI ||
        kind == vector::CombiningKind::MINF) {
      generateAIEVecOpsForReductionOp<aievec::MinOp>(rewriter, srcOp,
                                                     shiftIndex);
      return success();
    }

    // Reduction maximum
    if (kind == vector::CombiningKind::MAXSI ||
        kind == vector::CombiningKind::MAXUI ||
        kind == vector::CombiningKind::MAXF) {
      generateAIEVecOpsForReductionOp<aievec::MaxOp>(rewriter, srcOp,
                                                     shiftIndex);
      return success();
    }

    // Reduction add for i32, i16 and i8 types
    if (kind == vector::CombiningKind::ADD && isa<IntegerType>(scalarType)) {
      generateAIEVecOpsForReductionOp<aievec::AddElemOp>(rewriter, srcOp,
                                                         shiftIndex);
      return success();
    }

    // Reduction add for float and bfloat16
    if (kind == vector::CombiningKind::ADD && isa<FloatType>(scalarType)) {
      // float type
      if (elWidth == 32 && laneSize == 16) {
        generateReductionOpForFloat(rewriter, srcOp, shiftIndex);
        return success();
      }

      // bfloat16 type
      if (elWidth == 16 && laneSize == 16) {
        generateReductionOpForBFloat16(rewriter, srcOp, shiftIndex);
        return success();
      }
    }
    return failure();
  }
};

// If a UPD op is loading a vector twice the size of the architecture
// vector size, split it into a high and low load into the accumulator.
// TODO: This is a process we may want to include as part of the
// TODO: legalization of `vector.transfer_read`.
struct SplitUPDOpOnAccPattern : public OpConversionPattern<aievec::UPDOp> {
  using OpConversionPattern<aievec::UPDOp>::OpConversionPattern;

  SplitUPDOpOnAccPattern(MLIRContext *context, AnalysisManager &am,
                         int32_t maxVectorSize = 256)
      : OpConversionPattern<aievec::UPDOp>(context), am(am),
        maxVectorSize(maxVectorSize) {}

  LogicalResult
  matchAndRewrite(aievec::UPDOp updOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (am.getChildAnalysis<UPDOpEffectiveAccessSizeAnalysis>(updOp)
            .effectiveSize < 2 * static_cast<unsigned>(maxVectorSize))
      return failure();

    auto updOp0 = rewriter.create<aievec::UPDOp>(
        updOp.getLoc(), updOp.getResult().getType(), adaptor.getSource(),
        adaptor.getIndices(), 0, 0);
    rewriter.replaceOpWithNewOp<aievec::UPDOp>(
        updOp, updOp.getResult().getType(), adaptor.getSource(),
        adaptor.getIndices(), 2 * maxVectorSize, 1, updOp0.getResult());
    return success();
  }

  AnalysisManager &am;
  int32_t maxVectorSize;
};

template <typename OpTy>
struct SetInboundsToReadStoreOpPattern : public RewritePattern {
  SetInboundsToReadStoreOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    OpTy writeOrReadOp = cast<OpTy>(op);

    // TODO: We are currently setting all `vector.transfer_read` and
    // TODO: `vector.transfer_write` as "in bounds". We need to add
    // TODO: an analysis to verify that this is true before doing so.
    if (writeOrReadOp.getInBounds() || writeOrReadOp.getTransferRank() == 0) {
      return failure();
    }

    SmallVector<bool, 4> bools(writeOrReadOp.getTransferRank(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(bools);
    rewriter.updateRootInPlace(writeOrReadOp, [&]() {
      writeOrReadOp->setAttr(writeOrReadOp.getInBoundsAttrName(), inBoundsAttr);
    });
    return success();
  }
};

using SetInboundsToReadOp = SetInboundsToReadStoreOpPattern<TransferReadOp>;
using SetInboundsToWriteOp = SetInboundsToReadStoreOpPattern<TransferWriteOp>;

struct MergeSingleColumnI16FMAOpPattern
    : public OpConversionPattern<aievec::FMAOp> {
  using OpConversionPattern<aievec::FMAOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aievec::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isSingleColumnInt16VectorTimesScalarMac(adaptor))
      return failure();
    auto accProdOp = adaptor.getAcc().getDefiningOp();
    if (!accProdOp)
      return failure();
    auto accFmaOp = dyn_cast<aievec::FMAOp>(accProdOp);
    if (!accFmaOp)
      return failure();
    if (!isSingleColumnInt16VectorTimesScalarMac(accFmaOp))
      return failure();
    if (adaptor.getRhs() != accFmaOp.getRhs())
      return failure();
    auto accConcatOp =
        cast<aievec::ConcatOp>(accFmaOp.getLhs().getDefiningOp());
    auto fmaConcatOp = cast<aievec::ConcatOp>(adaptor.getLhs().getDefiningOp());
    unsigned fmaZstart, accFmaZstart;
    if (adaptor.getZstart().getAsInteger(10, fmaZstart) ||
        accFmaOp.getZstart().getAsInteger(10, accFmaZstart))
      return failure();
    auto start = std::min(fmaZstart, accFmaZstart);
    auto step = std::max(fmaZstart, accFmaZstart) - start;
    auto lowV = accConcatOp.getSources()[0];
    auto hiV = fmaConcatOp.getSources()[0];
    if (accFmaZstart > fmaZstart)
      std::swap(lowV, hiV);
    auto newConcatOp = rewriter.create<aievec::ConcatOp>(
        fmaOp.getLoc(), adaptor.getLhs().getType(),
        SmallVector<Value, 2>({lowV, hiV}));
    auto newFmaOpAttr = buildFMAOpSplatAttrForElemTy(fmaOp, start, step);
    rewriter.replaceOpWithNewOp<aievec::FMAOp>(
        fmaOp, TypeRange({fmaOp.getResult().getType()}),
        ValueRange({newConcatOp, adaptor.getRhs(), accFmaOp.getAcc()}),
        newFmaOpAttr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern collection
//===----------------------------------------------------------------------===//

static void populateAIEVecV1ConversionPatterns(RewritePatternSet &patterns,
                                               AnalysisManager &am) {
  patterns.add<LowerVectorTransferReadToAIEUPD, SplitUPDOpOnAccPattern>(
      patterns.getContext(), am, 256);
  patterns
      .add<LowerVectorAddFOpToAIEVecAddOp, LowerVectorSubIOpToAIEVecSubOp,
           LowerVectorSubFOpToAIEVecSubOp, ConvertMulAddToAIEVecFMAOpPattern,
           FoldBroadcastToFMAOp, LowerVectorAddIOpToAIEVecAddOp>(
          patterns.getContext());
}

static void populateAIEVecV2ConversionPatterns(RewritePatternSet &patterns,
                                               AnalysisManager &am) {
  patterns.add<LowerVectorTransferReadToAIEUPD, SplitUPDOpOnAccPattern>(
      patterns.getContext(), am, 512);

  patterns.add<
      LowerVectorAddIOpToAIEVecAddElemOp, LowerVectorAddFOpToAIEVecAddElemOp,
      LowerVectorSubIOpToAIEVecSubElemOp, LowerVectorSubFOpToAIEVecSubElemOp,
      LowerVectorMinSIOpToAIEVecMinOp, LowerVectorMaxSIOpToAIEVecMaxOp,
      LowerVectorMinFOpToAIEVecMinOp, LowerVectorMaxFOpToAIEVecMaxOp,
      LowerVectorCmpIOpToAIEVecCmpOp, LowerVectorCmpFOpToAIEVecCmpOp,
      LowerVectorSelectOpToAIEVecSelOp, LowerVectorReductionOp,
      FoldVectorExtractAndBroadcastToAIEBroadcast,
      ConvertMulAddToAIEVecFMAElemOpPattern,
      ConvertMulIToAIEVecMulElemOpPattern, ConvertMulFToAIEVecMulElemOpPattern>(
      patterns.getContext());
}

static void
populateAIEVecV1TransformationPatterns(RewritePatternSet &patterns) {
  patterns.add<MergeSingleColumnI16FMAOpPattern>(patterns.getContext());
}

static void
populateAIEVecV2TransformationPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldAIEShiftAndBroadcast>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Legalizations
//===----------------------------------------------------------------------===//

// TODO: Review the validity of these legalizations beyond basic cases.

static void configureAIEVecCommonLegalizations(ConversionTarget &target,
                                               AnalysisManager &am) {
  target.addLegalDialect<xilinx::aievec::AIEVecDialect, arith::ArithDialect>();
  target.addIllegalOp<vector::TransferReadOp>();
  target.addDynamicallyLegalOp<arith::AddIOp>(
      [](arith::AddIOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::AddFOp>(
      [](arith::AddFOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::SubIOp>(
      [](arith::SubIOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::SubFOp>(
      [](arith::SubFOp op) { return !isa<VectorType>(op.getType()); });
}

static void configureAIEVecV1Legalizations(ConversionTarget &target,
                                           AnalysisManager &am) {
  target.addDynamicallyLegalOp<aievec::UPDOp>([&am](xilinx::aievec::UPDOp op) {
    return am.getChildAnalysis<UPDOpEffectiveAccessSizeAnalysis>(op)
               .effectiveSize <= 512;
  });
  target.addDynamicallyLegalOp<aievec::FMAOp>([](xilinx::aievec::FMAOp op) {
    auto lhsDefOp = op.getLhs().getDefiningOp();
    aievec::ConcatOp concatOp = nullptr;
    if (lhsDefOp)
      concatOp = dyn_cast<aievec::ConcatOp>(op.getLhs().getDefiningOp());
    if (!concatOp)
      return true;
    vector::BroadcastOp srcBcast = nullptr;
    auto lhsOp = concatOp.getSources()[0].getDefiningOp();
    if (lhsOp)
      srcBcast = dyn_cast<vector::BroadcastOp>(lhsOp);
    if (!srcBcast) {
      auto rhsOp = op.getRhs().getDefiningOp();
      if (!rhsOp)
        return true;
      srcBcast = dyn_cast<vector::BroadcastOp>(rhsOp);
    }
    if (srcBcast) {
      auto srcOp = srcBcast.getSource().getDefiningOp();
      if (srcOp)
        return !isa<vector::ExtractOp>(srcOp);
    }
    return true;
  });
  target.addLegalDialect<memref::MemRefDialect>();
}

static void configureAIEVecV2Legalizations(ConversionTarget &target,
                                           AnalysisManager &am) {
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addDynamicallyLegalOp<aievec::UPDOp>([&am](aievec::UPDOp op) {
    return am.getChildAnalysis<UPDOpEffectiveAccessSizeAnalysis>(op)
               .effectiveSize <= 1024;
  });

  // A set recording the vector lane size and element width supported
  llvm::SmallSet<std::pair<unsigned, unsigned>, 16> laneSizeElWidthPairSet;
  laneSizeElWidthPairSet.insert({64, 8});
  laneSizeElWidthPairSet.insert({32, 16});
  laneSizeElWidthPairSet.insert({16, 32});
  laneSizeElWidthPairSet.insert({32, 32});

  // A set recording the element width supported
  llvm::SmallSet<unsigned, 16> elWidthSet;
  elWidthSet.insert(8);
  elWidthSet.insert(16);
  elWidthSet.insert(32);

  target.addDynamicallyLegalOp<arith::AddIOp>([=](arith::AddIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !laneSizeElWidthPairSet.count(
        std::make_pair(laneSize, resultElWidth));
  });

  target.addDynamicallyLegalOp<arith::SubIOp>([=](arith::SubIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !laneSizeElWidthPairSet.count(
        std::make_pair(laneSize, resultElWidth));
  });

  target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    unsigned laneSize = getVectorLaneSize(resultType);
    return laneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::SubFOp>([](arith::SubFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    unsigned laneSize = getVectorLaneSize(resultType);
    return laneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::MulIOp>([](arith::MulIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto isAddOp = [&](Operation *op) { return isa<arith::AddIOp>(op); };
    // Verify it is not a part of MAC
    if (op->hasOneUse() && llvm::any_of(op->getUsers(), isAddOp))
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return (laneSize != 32 || (resultElWidth != 16 && resultElWidth != 8)) &&
           ((laneSize != 16 && laneSize != 32) || resultElWidth != 32);
  });

  target.addDynamicallyLegalOp<arith::MulFOp>([](arith::MulFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto isAddOp = [&](Operation *op) { return isa<arith::AddFOp>(op); };
    // Verify it is not a part of FMA
    if (op->hasOneUse() && llvm::any_of(op->getUsers(), isAddOp))
      return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return (laneSize != 16 || (resultElWidth != 16 && resultElWidth != 32));
  });

  target.addDynamicallyLegalOp<arith::MinSIOp>([=](arith::MinSIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !(elWidthSet.count(resultElWidth) &&
             laneSize * resultElWidth == 512);
  });

  target.addDynamicallyLegalOp<arith::MaxSIOp>([=](arith::MaxSIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !(elWidthSet.count(resultElWidth) &&
             laneSize * resultElWidth == 512);
  });

  target.addDynamicallyLegalOp<arith::MinFOp>([=](arith::MinFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !(elWidthSet.count(resultElWidth) &&
             laneSize * resultElWidth == 512);
  });

  target.addDynamicallyLegalOp<arith::MaxFOp>([=](arith::MaxFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    return !(elWidthSet.count(resultElWidth) &&
             laneSize * resultElWidth == 512);
  });

  target.addDynamicallyLegalOp<arith::CmpIOp>([=](arith::CmpIOp op) {
    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    if (!lhsType) {
      return true;
    }
    auto lhsElWidth = lhsType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);

    if (!(elWidthSet.count(lhsElWidth) && laneSize * lhsElWidth == 512)) {
      return true;
    }

    return false;
  });

  target.addDynamicallyLegalOp<arith::CmpFOp>([=](arith::CmpFOp op) {
    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    if (!lhsType) {
      return true;
    }
    auto lhsElWidth = lhsType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(lhsType);

    if (!(elWidthSet.count(lhsElWidth) && laneSize * lhsElWidth == 512)) {
      return true;
    }

    return false;
  });

  target.addDynamicallyLegalOp<arith::SelectOp>([=](arith::SelectOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) {
      return true;
    }
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if (!(elWidthSet.count(resultElWidth) && laneSize * resultElWidth == 512)) {
      return true;
    }

    return false;
  });

  target.addDynamicallyLegalOp<vector::ReductionOp>(
      [=](vector::ReductionOp op) {
        auto kind = op.getKind();

        if (kind != vector::CombiningKind::ADD &&
            kind != vector::CombiningKind::MINSI &&
            kind != vector::CombiningKind::MINUI &&
            kind != vector::CombiningKind::MINF &&
            kind != vector::CombiningKind::MAXSI &&
            kind != vector::CombiningKind::MAXUI &&
            kind != vector::CombiningKind::MAXF) {
          return true;
        }

        VectorType vType = dyn_cast<VectorType>(op.getVector().getType());
        if (!vType)
          return true;

        llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
        laneSizeElWidthPairSet.insert({64, 8});
        laneSizeElWidthPairSet.insert({32, 16});
        laneSizeElWidthPairSet.insert({16, 32});

        Type scalarType = vType.getElementType();
        unsigned elWidth = scalarType.getIntOrFloatBitWidth();
        unsigned laneSize = getVectorLaneSize(vType);

        if (scalarType.isa<IntegerType>() &&
            !laneSizeElWidthPairSet.count(std::make_pair(laneSize, elWidth)))
          return true;

        if (scalarType.isa<FloatType>() && laneSize != 16 && laneSize != 32)
          return true;

        return false;
      });
}

static bool singleColumnFMAOpCanFold(aievec::FMAOp fmaOp) {
  auto accProdOp = fmaOp.getAcc().getDefiningOp();
  if (!accProdOp)
    return false;
  auto accFmaOp = dyn_cast<aievec::FMAOp>(accProdOp);
  if (!accFmaOp)
    return false;
  if (!isSingleColumnInt16VectorTimesScalarMac(accFmaOp))
    return false;
  return fmaOp.getRhs() == accFmaOp.getRhs() &&
         !singleColumnFMAOpCanFold(accFmaOp);
}

static void
configureAIEVecV1TransformationLegalizations(ConversionTarget &target) {
  target.addLegalDialect<aievec::AIEVecDialect>();
  target.addDynamicallyLegalOp<aievec::FMAOp>([](aievec::FMAOp fmaOp) {
    if (isSingleColumnInt16VectorTimesScalarMac(fmaOp))
      return !singleColumnFMAOpCanFold(fmaOp);
    return true;
  });
}

static void
configureAIEVecV2TransformationLegalizations(ConversionTarget &target) {
  target.addDynamicallyLegalOp<xilinx::aievec::BroadcastOp>(
      [](xilinx::aievec::BroadcastOp op) {
        aievec::ShiftOp shiftOp = nullptr;
        int32_t idx = 0;
        return !canFoldAIEShiftAndBroadcast(op, shiftOp, idx);
      });
}

//===----------------------------------------------------------------------===//
// Lowering passes
//===----------------------------------------------------------------------===//

// TODO: For more complex conversion from Vector to AIEVec, we may want to
// make
// TODO: this into a pipeline where:
// TODO:     1. If the operands of a vector op are too long, split it down to
// TODO:        right-sized vectors.
// TODO:     2. Unroll vector ops when the vector type is unsupported.
// TODO:     3. Perform the dialect conversion legalizations.
struct LowerVectorToAIEVec
    : public aievec::impl::LowerVectorToAIEVecBase<LowerVectorToAIEVec> {
  using Base::Base;

  void runOnOperation() override;
};

/// Lower incoming vector operations into their corresponding AIE vector
/// intrinsics.
void LowerVectorToAIEVec::runOnOperation() {
  auto func = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  AIEArch aieVersion = AIEArch::AIE;
  if (!aieTarget.empty()) {
    std::string target = aieTarget;
    if (target == "aieml") {
      aieVersion = AIEArch::AIE_ML;
    } else if (target != "aie") {
      func.emitError() << "unknown AIE target '" << aieTarget << "'";
      signalPassFailure();
      return;
    }
  }

  AnalysisManager am = getAnalysisManager();
  configureAIEVecCommonLegalizations(target, am);
  if (aieVersion == AIEArch::AIE) {
    populateAIEVecV1ConversionPatterns(patterns, am);
    configureAIEVecV1Legalizations(target, am);
  } else {
    populateAIEVecV2ConversionPatterns(patterns, am);
    configureAIEVecV2Legalizations(target, am);
  }

  if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

// This pass converts standard vector ops into a subset of `Vector` ops more
// amenable to being converted to `AIEVec`. So far, this process consists of
// one steps:
//    1) Replace splat transfer reads with contiguous transfer reads followed
//       by `extract` + `broadcast` operations.
struct CanonicalizeForAIEVecPass
    : public aievec::impl::CanonicalizeForAIEVecBase<
          CanonicalizeForAIEVecPass> {
  using Base::Base;

  void runOnOperation() override;
};

static void
configureCommonAIECanonicalizeLegalizations(ConversionTarget &target) {
  target.addLegalDialect<vector::VectorDialect>();
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<aievec::AIEVecDialect>();
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant();
      });
}

static void configureAIEv1CanonicalizeLegalizations(ConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant() &&
               getTransferReadAlignmentOffset(op, op.getVectorType()) == 0;
      });
}

static void configureAIEMLCanonicalizeLegalizations(ConversionTarget &target) {}

static void
populateCommonAIECanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertSplatTransferReadToBroadcastPattern>(
      patterns.getContext());
}

static void
populateAIEv1CanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<SplitUnalignedTransferReadPattern>(patterns.getContext());
}

static void
populateAIEMLCanonicalizeConversionPatterns(RewritePatternSet &patterns) {}

void CanonicalizeForAIEVecPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  AIEArch aieVersion = AIEArch::AIE;
  if (!aieTarget.empty()) {
    std::string target = aieTarget;
    if (target == "aieml") {
      aieVersion = AIEArch::AIE_ML;
    } else if (target != "aie") {
      funcOp.emitError() << "unknown AIE target '" << aieTarget << "'";
      signalPassFailure();
      return;
    }
  }

  populateCommonAIECanonicalizeConversionPatterns(patterns);
  configureCommonAIECanonicalizeLegalizations(target);
  if (aieVersion == AIEArch::AIE) {
    populateAIEv1CanonicalizeConversionPatterns(patterns);
    configureAIEv1CanonicalizeLegalizations(target);
  } else {
    populateAIEMLCanonicalizeConversionPatterns(patterns);
    configureAIEMLCanonicalizeLegalizations(target);
  }

  if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

struct RedundantLoadStoreOptimizationPass
    : public PassWrapper<RedundantLoadStoreOptimizationPass,
                         OperationPass<func::FuncOp>> {
  void runOnOperation() override;
};

void RedundantLoadStoreOptimizationPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

  patterns.add<SetInboundsToReadOp, SetInboundsToWriteOp>(
      patterns.getContext());

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  transferOpflowOpt(funcOp);
}

std::unique_ptr<::mlir::Pass> createRedundantLoadStoreOptimizationPass() {
  return std::make_unique<RedundantLoadStoreOptimizationPass>();
}

struct AIEVecTransformationPass
    : public aievec::impl::AIEVecTransformationBase<AIEVecTransformationPass> {
  using Base::Base;
  void runOnOperation() override;
};

void AIEVecTransformationPass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  AIEArch aieVersion = AIEArch::AIE;
  if (!aieTarget.empty()) {
    std::string target = aieTarget;
    if (target == "aieml") {
      aieVersion = AIEArch::AIE_ML;
    } else if (target != "aie") {
      func.emitError() << "unknown AIE target '" << aieTarget << "'";
      signalPassFailure();
      return;
    }
  }
  if (aieVersion == AIEArch::AIE) {
    populateAIEVecV1TransformationPatterns(patterns);
    configureAIEVecV1TransformationLegalizations(target);
  } else {
    populateAIEVecV2TransformationPatterns(patterns);
    configureAIEVecV2TransformationLegalizations(target);
  }

  if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

struct AIEVecConvOpTransformationPass
    : public aievec::impl::AIEVecConvOpTransformationBase<
          AIEVecConvOpTransformationPass> {
  using Base::Base;
  void runOnOperation() override;
};

void AIEVecConvOpTransformationPass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  AIEArch aieVersion = AIEArch::AIE;
  if (!aieTarget.empty()) {
    std::string target = aieTarget;
    if (target == "aieml") {
      aieVersion = AIEArch::AIE_ML;
    } else if (target != "aie") {
      func.emitError() << "unknown AIE target '" << aieTarget << "'";
      signalPassFailure();
      return;
    }
  }

  AnalysisManager am = getAnalysisManager();
  if (aieVersion == AIEArch::AIE_ML) {
    populateAIEVecConvOpTransformationPatterns(patterns, am, shiftParam);
    configureAIEVecConvOpTransformationLegalizations(target, am);
  }

  if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    signalPassFailure();
  }
}
//===---------------------------------------------------------------------------
// Pipeline implementations
//===---------------------------------------------------------------------------
void xilinx::aievec::buildConvertVectorToAIEVec(
    OpPassManager &pm, const ConvertVectorToAIEVecOptions &options) {
  pm.addPass(createCanonicalizerPass());

  pm.addPass(createRedundantLoadStoreOptimizationPass());

  // Add `Vector` code canonicalization passes
  // TODO: Add passes to unroll vector with unsupported types
  // TODO: Add passes to split vectors that won't fit in registers
  pm.addPass(
      createCanonicalizeForAIEVec(options.getCanonicalizeForAIEVecOptions()));
  // Add lowering from `Vector` to `AIEVec`
  pm.addPass(
      createLowerVectorToAIEVec(options.getLowerVectorToAIEVecOptions()));
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // Add AIEVec transformation pass
  pm.addPass(
      createAIEVecTransformation(options.getAIEVecTransformationOptions()));

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Add generating aievec convolution ops pass
  pm.addPass(createAIEVecConvOpTransformation(
      options.getAIEVecConvOpTransformationOptions()));

  // Add post-lowering canonicalization passes
  pm.addPass(createCSEPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
}

//===---------------------------------------------------------------------------
// Pipeline registration
//===---------------------------------------------------------------------------
void xilinx::aievec::registerAIEVecPipelines() {
  PassPipelineRegistration<ConvertVectorToAIEVecOptions>(
      "convert-vector-to-aievec",
      "This pass pipeline takes standard \"Vector\" code and converts it to "
      "\"AIEVec\" code targeting the selected Xilinx AIE vector "
      "architecture.",
      buildConvertVectorToAIEVec);
}
