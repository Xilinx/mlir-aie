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
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"

namespace xilinx::aievec {
#define GEN_PASS_DEF_LOWERVECTORTOAIEVEC
#define GEN_PASS_DEF_CANONICALIZEFORAIEVEC
#define GEN_PASS_DEF_REDUNDANTLOADSTOREOPTIMIZATION
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

// Given an AIEOp, determines if an operation writes to an accumulator
// based on operation type and operand types
static bool writesToAccumulator(Operation *op) {
  // Integer muls and FMAs write to accumulator
  if (!isAIEOp(op))
    return false;
  if (auto mulOp = dyn_cast<aievec::MulOp>(op))
    return mulOp.getResult()
        .getType()
        .cast<VectorType>()
        .getElementType()
        .isa<IntegerType>();
  if (auto fmaOp = dyn_cast<aievec::FMAOp>(op))
    return fmaOp.getResult()
        .getType()
        .cast<VectorType>()
        .getElementType()
        .isa<IntegerType>();

  return isa<aievec::FMAElemOp, aievec::MulElemOp, aievec::FMAConvOp,
             aievec::MulConvOp, aievec::UPSOp>(op);
}

//
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

// This pattern replaces `vector.fma` with `aievec.mac_elem` for aie-ml.
struct LowerVectorFMAToAIEVecFMAElem
    : public OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern<vector::FMAOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    VectorType resultType = fmaOp.getResult().getType().cast<VectorType>();

    if (!resultType.getElementType().isa<IntegerType>())
      return failure();

    IntegerType resultElType = resultType.getElementType().cast<IntegerType>();
    unsigned resultElWidth = resultElType.getWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if ((laneSize != 32 || resultElWidth != 16) &&
        (laneSize != 16 || resultElWidth != 32))
      return failure();

    Value acc = adaptor.getAcc();
    Type accType = getVectorOpDestType(acc.getType().cast<VectorType>(), true);
    auto accDefOp = acc.getDefiningOp();
    if (!writesToAccumulator(accDefOp)) {
      acc = rewriter.create<aievec::UPSOp>(fmaOp->getLoc(), accType, acc);
      LLVM_DEBUG(llvm::dbgs()
                 << "\n\nCreated UPS op " << acc << " to move the output of "
                 << fmaOp << " into accumulator");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto fmaElemOp = rewriter.create<aievec::FMAElemOp>(
        fmaOp->getLoc(), lhs, rhs, acc, /*fmsub=*/false);

    VectorType fmaElemResultType =
        fmaElemOp.getResult().getType().cast<VectorType>();
    IntegerType fmaElemResultElType =
        fmaElemResultType.getElementType().cast<IntegerType>();
    unsigned fmaResultElWidth = fmaElemResultElType.getWidth();

    if (fmaResultElWidth > resultElWidth) {
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(fmaOp, resultType,
                                                 fmaElemOp.getResult());
    } else {
      rewriter.replaceOpWithNewOp<aievec::FMAElemOp>(fmaOp, lhs, rhs, acc,
                                                     /*fmsub=*/false);
    }
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
          offset = applyOp.getAffineMap().compose({0})[0];
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

// This pattern folds an extract + broadcast feeding into an `aievec::FMAOp`
// into the op, using the shuffle attributes.
struct FoldBroadcastToFMAOp : public OpConversionPattern<aievec::FMAOp> {
  using OpConversionPattern<aievec::FMAOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aievec::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bcastOp =
        dyn_cast<vector::BroadcastOp>(adaptor.getLhs().getDefiningOp());
    Value rhs = adaptor.getRhs();
    if (!bcastOp) {
      bcastOp = dyn_cast<vector::BroadcastOp>(adaptor.getRhs().getDefiningOp());
      rhs = adaptor.getLhs();
      if (!bcastOp)
        return failure();
    }
    auto extOp =
        dyn_cast<vector::ExtractOp>(bcastOp.getSource().getDefiningOp());
    if (!extOp)
      return failure();

    auto newLhs = extOp.getVector();
    // XXX: We assume a 1D vector
    auto pos = extOp.getPosition();
    int64_t zstart = cast<IntegerAttr>(pos[0]).getInt();
    rewriter.replaceOpWithNewOp<aievec::FMAOp>(
        fmaOp, fmaOp.getResult().getType(), newLhs, rhs, adaptor.getAcc(),
        /*xstart =*/"0", /*xoffsets =*/"0x76543210", adaptor.getXoffsetsHi(),
        adaptor.getXstep(), adaptor.getXsquare(),
        /*zstart =*/std::to_string(zstart), adaptor.getZoffsets(),
        adaptor.getZoffsetsHi(), adaptor.getZstep(), adaptor.getZsquare(),
        adaptor.getFmsub());
    return success();
  }
};

// This pattern converts a vector `arith.add` following a vector `arith.mul`
// into a `vector.fma`.
template <typename MulOpTy, typename AddOpTy>
struct FoldMulAddToFMAPattern : public OpConversionPattern<AddOpTy> {
  using OpConversionPattern<AddOpTy>::OpConversionPattern;
  using OpAdaptor = typename AddOpTy::Adaptor;

  LogicalResult
  matchAndRewrite(AddOpTy addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<VectorType>(addOp.getType()))
      return failure();
    auto lhsDefOp = adaptor.getLhs().getDefiningOp();
    auto rhsDefOp = adaptor.getRhs().getDefiningOp();
    MulOpTy lhsOp = nullptr, rhsOp = nullptr;
    if (lhsDefOp)
      lhsOp = dyn_cast<MulOpTy>(lhsDefOp);
    if (rhsDefOp)
      rhsOp = dyn_cast<MulOpTy>(rhsDefOp);
    if (!(lhsOp || rhsOp))
      return failure();
    MulOpTy mulOp = lhsOp;
    auto acc = adaptor.getRhs();
    if (!mulOp) {
      mulOp = rhsOp;
      acc = adaptor.getLhs();
    }
    rewriter.replaceOpWithNewOp<vector::FMAOp>(
        addOp, addOp.getType(), mulOp.getLhs(), mulOp.getRhs(), acc);
    return success();
  }
};

using FoldMulAddIntToFMAPattern =
    FoldMulAddToFMAPattern<arith::MulIOp, arith::AddIOp>;
using FoldMulAddFloatToFMAPattern =
    FoldMulAddToFMAPattern<arith::MulFOp, arith::AddFOp>;

struct LowerVectorFMAOpPattern : public OpConversionPattern<vector::FMAOp> {
  using OpConversionPattern<vector::FMAOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::FMAOp fmaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    VectorType fmaVecTy = cast<VectorType>(fmaOp.getType());
    auto fmaVecElemTy = fmaVecTy.getElementType();
    // TODO: Select the appropriate wide type for the accumulator
    //    unsigned vecElemWidth = fmaVecElemTy.getIntOrFloatBitWidth();
    unsigned wideElemWidth = 80;
    Type wideElemTy;
    if (isa<IntegerType>(fmaVecElemTy))
      wideElemTy = rewriter.getIntegerType(wideElemWidth);
    else
      wideElemTy = rewriter.getF64Type();
    VectorType accVecTy = VectorType::get(fmaVecTy.getShape(), wideElemTy);
    auto upsOp = rewriter.create<aievec::UPSOp>(fmaOp.getLoc(), accVecTy,
                                                fmaOp.getAcc());
    auto aiFmaOp = rewriter.create<aievec::FMAOp>(
        fmaOp.getLoc(), accVecTy, adaptor.getLhs(), adaptor.getRhs(),
        upsOp.getResult(),
        /*xstart=*/"", /*xoffsets=*/"", /*xoffsets_hi=*/"", /*xstep=*/"",
        /*xsquare=*/"", /*zstart=*/"", /*zoffsets=*/"", /*zoffsets_hi=*/"",
        /*zstep=*/"", /*zsquare=*/"", /*fmsub=*/false);
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(fmaOp, fmaVecTy,
                                               aiFmaOp.getResult());
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

    // TODO: Verify alignment
    // TODO: Extend this function so it can address more than the trivial
    // case.
    SmallVector<Value, 4> indices(adaptor.getIndices().begin(),
                                  adaptor.getIndices().end());
    rewriter.replaceOpWithNewOp<xilinx::aievec::UPDOp>(
        readOp, readOp.getVector().getType(), adaptor.getSource(), indices, 0,
        0, TypedValue<VectorType>(nullptr));
    return success();
  }

  AnalysisManager &am;
  int32_t maxVectorSize;
};

// XXX: Notice that this template doesn't verify that the vector element type
// is
// XXX: supported by the target architecture.
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

using LowerVectorAddIOpToAIEVecAddOp =
    OneToOneVectorOpToAIEVecOpPattern<arith::AddIOp, aievec::AddOp>;
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
struct SetInboundsToReadStoreOpPattern : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if ((!op.getInBounds()) && op.getTransferRank() != 0) {
      SmallVector<bool, 4> bools(op.getTransferRank(), true);
      auto inBoundsAttr = rewriter.getBoolArrayAttr(bools);
      rewriter.updateRootInPlace(
          op, [&]() { op->setAttr(op.getInBoundsAttrName(), inBoundsAttr); });
      return success();
    }
    return failure();
  }
};

using SetInboundsToReadOp = SetInboundsToReadStoreOpPattern<TransferReadOp>;
using SetInboundsToWriteOp = SetInboundsToReadStoreOpPattern<TransferWriteOp>;

//===----------------------------------------------------------------------===//
// Pattern collection
//===----------------------------------------------------------------------===//

static void populateAIEVecCommonConversionPatterns(RewritePatternSet &patterns,
                                                   AnalysisManager &am) {
  patterns.add<LowerVectorAddIOpToAIEVecAddOp, LowerVectorAddFOpToAIEVecAddOp,
               LowerVectorSubIOpToAIEVecSubOp, LowerVectorSubFOpToAIEVecSubOp>(
      patterns.getContext());
}

static void populateAIEVecV1ConversionPatterns(RewritePatternSet &patterns,
                                               AnalysisManager &am) {
  patterns.add<LowerVectorTransferReadToAIEUPD, SplitUPDOpOnAccPattern>(
      patterns.getContext(), am, 256);
  patterns.add<LowerVectorFMAOpPattern, FoldBroadcastToFMAOp>(
      patterns.getContext());
}

static void populateAIEVecV2ConversionPatterns(RewritePatternSet &patterns,
                                               AnalysisManager &am) {
  patterns.add<LowerVectorTransferReadToAIEUPD, SplitUPDOpOnAccPattern>(
      patterns.getContext(), am, 512);

  patterns.add<FoldVectorExtractAndBroadcastToAIEBroadcast,
               LowerVectorFMAToAIEVecFMAElem>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Legalizations
//===----------------------------------------------------------------------===//

// TODO: Review the validity of these legalizations beyond basic cases.

static void configureAIEVecCommonLegalizations(ConversionTarget &target,
                                               AnalysisManager &am) {
  target.addLegalDialect<xilinx::aievec::AIEVecDialect>();
  target.addLegalDialect<arith::ArithDialect>();
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
  target.addIllegalOp<vector::FMAOp>();
  target.addDynamicallyLegalOp<aievec::UPDOp>([&am](xilinx::aievec::UPDOp op) {
    return am.getChildAnalysis<UPDOpEffectiveAccessSizeAnalysis>(op)
               .effectiveSize <= 512;
  });
  target.addDynamicallyLegalOp<aievec::FMAOp>([](xilinx::aievec::FMAOp op) {
    auto srcBcast = dyn_cast<vector::BroadcastOp>(op.getLhs().getDefiningOp());
    if (!srcBcast)
      srcBcast = dyn_cast<vector::BroadcastOp>(op.getRhs().getDefiningOp());
    if (srcBcast)
      return !isa<vector::ExtractOp>(srcBcast.getSource().getDefiningOp());
    return true;
  });
  target.addLegalDialect<memref::MemRefDialect>();
}

static void configureAIEVecV2Legalizations(ConversionTarget &target,
                                           AnalysisManager &am) {
  target.addDynamicallyLegalOp<aievec::UPDOp>([&am](aievec::UPDOp op) {
    return am.getChildAnalysis<UPDOpEffectiveAccessSizeAnalysis>(op)
               .effectiveSize <= 1024;
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
  populateAIEVecCommonConversionPatterns(patterns, am);
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

// Check whether one of the operands of the AddOp is defined by a MulOp
template <typename AddOpTy> static bool canFoldToFMA(AddOpTy addOp) {
  if (isa<VectorType>(addOp.getType())) {
    auto lhsOp = addOp.getLhs().getDefiningOp();
    if (lhsOp && isa<arith::MulIOp, arith::MulFOp>(lhsOp))
      return true;
    auto rhsOp = addOp.getRhs().getDefiningOp();
    if (rhsOp && isa<arith::MulIOp, arith::MulFOp>(rhsOp))
      return true;
  }
  return false;
}

// This pass converts standard vector ops into a subset of `Vector` ops more
// amenable to being converted to `AIEVec`. So far, this process consists of
// two steps:
//    1) Merge `arith::mul` followed by `arith::add` into `vector::fma`
//    2) Replace splat transfer reads with contiguous transfer reads followed
//    by
//       `extract` + `broadcast` operations.
struct CanonicalizeForAIEVecPass
    : public aievec::impl::CanonicalizeForAIEVecBase<
          CanonicalizeForAIEVecPass> {
  using Base::Base;

  void runOnOperation() override;
};

static void
configureCommonAIECanonicalizeLegalizations(ConversionTarget &target) {
  target.addLegalDialect<vector::VectorDialect>();
  target.addDynamicallyLegalOp<arith::AddIOp>(
      [](arith::AddIOp op) { return !canFoldToFMA(op); });
  target.addDynamicallyLegalOp<arith::AddFOp>(
      [](arith::AddFOp op) { return !canFoldToFMA(op); });
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant();
      });
}

static void configureAIEv1CanonicalizeLegalizations(ConversionTarget &target) {}

static void configureAIEMLCanonicalizeLegalizations(ConversionTarget &target) {}

static void
populateCommonAIECanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldMulAddIntToFMAPattern, FoldMulAddFloatToFMAPattern,
               ConvertSplatTransferReadToBroadcastPattern>(
      patterns.getContext());
}

static void
populateAIEv1CanonicalizeConversionPatterns(RewritePatternSet &patterns) {}

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
    : public aievec::impl::RedundantLoadStoreOptimizationBase<
          RedundantLoadStoreOptimizationPass> {
  using Base::Base;

  void runOnOperation() override;
};

static void
configureRedundantLoadStoreOptimizationLegalizations(ConversionTarget &target) {
  target.addLegalDialect<vector::VectorDialect>();
  target.addDynamicallyLegalOp<TransferReadOp>(
      [](vector::TransferReadOp op) { return !(!op.getInBounds()); });
  target.addDynamicallyLegalOp<TransferWriteOp>(
      [](vector::TransferWriteOp op) { return !(!op.getInBounds()); });
}

static void populateRedundantLoadStoreOptimizationConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SetInboundsToReadOp, SetInboundsToWriteOp>(
      patterns.getContext());
}

void RedundantLoadStoreOptimizationPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  populateRedundantLoadStoreOptimizationConversionPatterns(patterns);
  configureRedundantLoadStoreOptimizationLegalizations(target);

  transferOpflowOpt(funcOp);

  if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

//===---------------------------------------------------------------------------
// Pipeline implementations
//===---------------------------------------------------------------------------
void xilinx::aievec::buildConvertVectorToAIEVec(
    OpPassManager &pm, const ConvertVectorToAIEVecOptions &options) {
  pm.addPass(createCanonicalizerPass());

  pm.addPass(createRedundantLoadStoreOptimization());

  // Add `Vector` code canonicalization passes
  // TODO: Add passes to unroll vector with unsupported types
  // TODO: Add passes to split vectors that won't fit in registers
  pm.addPass(
      createCanonicalizeForAIEVec(options.getCanonicalizeForAIEVecOptions()));
  // Add lowering from `Vector` to `AIEVec`
  pm.addPass(
      createLowerVectorToAIEVec(options.getLowerVectorToAIEVecOptions()));
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
