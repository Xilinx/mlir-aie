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

typedef std::tuple<int8_t, aievec::UPDOp, arith::MulIOp> MulDefTupleTy;
using MulDefTupleVecTy = SmallVector<MulDefTupleTy, 8>;
using MulDefMapTy = DenseMap<Value, MulDefTupleVecTy>;

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
  idx = shiftOp.getShift() * 8 / elemSize + op.getIdx();

  if (idx <= 0 || idx >= (int32_t)getVectorLaneSize(vType)) {
    return false;
  }

  return true;
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

    SmallVector<Value> sources = shiftOp.getSources();

    VectorType resultType = bcastOp.getResult().getType().cast<VectorType>();

    rewriter.replaceOpWithNewOp<aievec::BroadcastOp>(bcastOp, resultType,
                                                     sources[0], idx);

    return success();
  }
};

// This pattern replaces `arith.muli`+`arith.addi` on vectors with
// `aievec.mac_elem`. This pattern works for aie-ml.
struct ConvertMulAddToAIEVecFMAElemOpPattern
    : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

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
    IntegerType resultElType = cast<IntegerType>(resultType.getElementType());
    unsigned resultElWidth = resultElType.getWidth();
    unsigned laneSize = getVectorLaneSize(resultType);

    if ((laneSize != 32 || resultElWidth != 16) &&
        (laneSize != 16 || resultElWidth != 32))
      return failure();

    Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
                                       /*AIEML =*/true);
    auto upsOp = rewriter.create<aievec::UPSOp>(addOp.getLoc(), accType, acc);
    auto fmaElemOp = rewriter.create<aievec::FMAElemOp>(
        addOp.getLoc(), accType, lhs, rhs, upsOp.getResult(),
        /*fmsub=*/false);
    rewriter.replaceOpWithNewOp<aievec::SRSOp>(addOp, resultType,
                                               fmaElemOp.getResult());

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

    Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
                                       /*AIEML =*/false);
    auto upsOp = rewriter.create<aievec::UPSOp>(addOp.getLoc(), accType, acc);
    auto fmaOp = rewriter.create<aievec::FMAOp>(
        addOp.getLoc(), accType, lhs, rhs, upsOp.getResult(),
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

            SmallVector<Value> sources = {updOp->getResult(0),
                                          nextUpdOp->getResult(0)};
            rewriter.replaceOpWithNewOp<xilinx::aievec::ShiftOp>(
                readOp, vType, sources, shiftBytes);
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

// Linearize the exprVec as a strided access, but do not simplify
static AffineExpr makeFlattenedStridedExpr(ArrayRef<int64_t> sizes,
                                           ArrayRef<AffineExpr> exprs,
                                           MLIRContext *context) {
  assert(!sizes.empty() && !exprs.empty() &&
         "expected non-empty sizes and exprs");
  if (llvm::is_contained(sizes, 0))
    return getAffineConstantExpr(0, context);

  auto maps = AffineMap::inferFromExprList(exprs);
  assert(!maps.empty() && "Expected one non-empty map");
  unsigned nSymbols = maps[0].getNumSymbols();

  AffineExpr expr;
  bool dynamicPoisonBit = false;
  int64_t runningSize = 1;
  for (auto en : llvm::zip(llvm::reverse(exprs), llvm::reverse(sizes))) {
    int64_t size = std::get<1>(en);

    if (size == 0)
      continue;
    AffineExpr dimExpr = std::get<0>(en);
    AffineExpr stride = dynamicPoisonBit
                            ? getAffineSymbolExpr(nSymbols++, context)
                            : getAffineConstantExpr(runningSize, context);
    expr = expr ? expr + dimExpr * stride : dimExpr * stride;
    if (size > 0) {
      runningSize *= size;
      assert(runningSize > 0 && "integer overflow in size computation");
    } else {
      dynamicPoisonBit = true;
    }
  }
  return expr;
}

// Construct a linearized affine expression for the transfer_read op.
static AffineExpr constructLinearizedAffineExpr(aievec::UPDOp updOp) {
  SmallVector<Value, 4> indices(updOp.getIndices().begin(),
                                updOp.getIndices().end());
  MemRefType memRefType = updOp.getSource().getType().cast<MemRefType>();
  MLIRContext *context = memRefType.getContext();

  SmallVector<AffineExpr, 8> exprVec;
  DenseMap<Value, AffineExpr> indexToExprDimMap;
  for (auto idxAndValue : llvm::enumerate(indices)) {
    auto value = idxAndValue.value();
    if (AffineApplyOp apOf = value.getDefiningOp<AffineApplyOp>()) {
      AffineMap map = apOf.getAffineMap();
      assert(map.getNumResults() == 1 &&
             "Failed to create linearized affineExpr for complicated index");
      SmallVector<AffineExpr, 4> indexExprs;

      for (auto index : apOf.getMapOperands()) {
        if (auto cIdx = index.getDefiningOp<arith::ConstantOp>()) {
          auto idxVal = cIdx.getValue().cast<IntegerAttr>().getValue();
          unsigned idx = idxVal.getSExtValue();
          indexExprs.push_back(getAffineConstantExpr(idx, context));
        } else {
          if (!indexToExprDimMap.count(index))
            indexToExprDimMap[index] =
                getAffineDimExpr(indexToExprDimMap.size(), context);
          indexExprs.push_back(indexToExprDimMap[index]);
        }
      }

      exprVec.push_back(map.getResult(0).replaceDims(indexExprs));
    } else if (auto cOp = value.getDefiningOp<arith::ConstantOp>()) {
      auto idxVal = cOp.getValue().cast<IntegerAttr>().getValue();
      unsigned idx = idxVal.getSExtValue();
      exprVec.push_back(getAffineConstantExpr(idx, context));
    } else {
      if (!indexToExprDimMap.count(value))
        indexToExprDimMap[value] =
            getAffineDimExpr(indexToExprDimMap.size(), context);
      exprVec.push_back(indexToExprDimMap[value]);
    }
  }

  assert(!exprVec.empty() && "Could not construct linearized affineExpr");

  auto ret = makeFlattenedStridedExpr(memRefType.getShape(), exprVec,
                                      memRefType.getContext());

  return ret;
}

// From a linearized affine expression, compute the base and the constant
// offset. If the access is A[i][j+2] for an N*N array A, the linearized
// expression will be A[i*N+j+2]. The base in this case will be (i*N+j), and the
// offset will be 2.
static std::pair<AffineExpr, int32_t> getBaseAndOffset(AffineExpr expr) {
  AffineExpr base = expr;
  int32_t offset = 0;

  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    base = nullptr;
    offset += constExpr.getValue();
  } else if (auto binopExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (binopExpr.getKind() == AffineExprKind::Add) {
      AffineExpr lhs = binopExpr.getLHS(), rhs = binopExpr.getRHS();
      if (auto constExpr = lhs.dyn_cast<AffineConstantExpr>()) {
        base = rhs;
        offset += constExpr.getValue();
      }
      if (auto constExpr = rhs.dyn_cast<AffineConstantExpr>()) {
        base = base == rhs ? nullptr : lhs;
        offset += constExpr.getValue();
      }
    }
  }
  return std::make_pair(base, offset);
}

static arith::MulIOp getDefMulOp(arith::AddIOp addOp) {
  arith::MulIOp defLhs =
      dyn_cast<arith::MulIOp>(addOp->getOperand(0).getDefiningOp());
  arith::MulIOp defRhs =
      dyn_cast<arith::MulIOp>(addOp->getOperand(1).getDefiningOp());
  if (!defLhs && !defRhs) {
    return nullptr;
  }
  if (defLhs && defRhs) {
    return defLhs->isBeforeInBlock(defRhs) ? defRhs : defLhs;
  }
  return defLhs ? defLhs : defRhs;
}

static arith::AddIOp getDefAddOp(arith::AddIOp addOp) {
  arith::AddIOp defLhs =
      dyn_cast<arith::AddIOp>(addOp->getOperand(0).getDefiningOp());
  arith::AddIOp defRhs =
      dyn_cast<arith::AddIOp>(addOp->getOperand(1).getDefiningOp());
  if ((!defLhs && !defRhs) || (defLhs && defRhs)) {
    return nullptr;
  }
  return defLhs ? defLhs : defRhs;
}

static bool checkLegalityForChain(arith::MulIOp mulOp,
                                  MulDefMapTy &macChainMap) {
  // Get the mul op's lhs and rhs defining ops. We keep splat op at
  // rhs.
  if (isa<aievec::BroadcastOp>(mulOp.getOperand(0).getDefiningOp())) {
    Value left = mulOp->getOperand(0);
    Value right = mulOp->getOperand(1);
    mulOp->setOperand(0, right);
    mulOp->setOperand(1, left);
  }

  if (!isa<aievec::UPDOp>(mulOp->getOperand(0).getDefiningOp())) {
    return false;
  }

  aievec::BroadcastOp bcastOp =
      dyn_cast<aievec::BroadcastOp>(mulOp->getOperand(1).getDefiningOp());

  if (!isa<aievec::UPDOp>(bcastOp.getSource().getDefiningOp())) {
    return false;
  }

  aievec::UPDOp updOp =
      cast<aievec::UPDOp>(mulOp->getOperand(0).getDefiningOp());

  if (!macChainMap.count(bcastOp.getSource())) {
    MulDefTupleVecTy tupleVec;
    tupleVec.push_back(std::make_tuple(bcastOp.getIdx(), updOp, mulOp));
    macChainMap.insert(std::make_pair(bcastOp.getSource(), tupleVec));
  } else {
    macChainMap[bcastOp.getSource()].push_back(
        std::make_tuple(bcastOp.getIdx(), updOp, mulOp));
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
  if (!usrOp || isa<arith::AddIOp>(usrOp) || isa<arith::MulIOp>(usrOp)) {
    return false;
  }

  arith::AddIOp curAddOp = addOp;
  // Build a mul add Chain map by recording the def of mul ops.
  // Identify the chain by checking the legality of their ops.
  while (true) {
    arith::MulIOp defLhs =
        dyn_cast<arith::MulIOp>(curAddOp->getOperand(0).getDefiningOp());
    arith::MulIOp defRhs =
        dyn_cast<arith::MulIOp>(curAddOp->getOperand(1).getDefiningOp());

    if (!defLhs && !defRhs) {
      break;
      // If both ops of add op are mul ops, this will reach the top of the
      // chain. Check the legality for both mul op and add them to the chain
      // map.
    } else if (defLhs && defRhs) {
      if (!checkLegalityForChain(defLhs, macChainMap) ||
          !checkLegalityForChain(defRhs, macChainMap)) {
        break;
      }
    } else {
      arith::MulIOp curMulOp = defLhs ? defLhs : defRhs;
      if (!checkLegalityForChain(curMulOp, macChainMap)) {
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

  if (macChainMap.empty()) {
    return false;
  }

  if (std::any_of(macChainMap.begin(), macChainMap.end(),
                  [](const auto &p) { return p.second.size() < 2; }))
    return false;

  for (auto item : macChainMap) {
    auto macChain = item.second;
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

      AffineExpr curLinearAccess = constructLinearizedAffineExpr(curUpdOp);
      AffineExpr nextLinearAccess = constructLinearizedAffineExpr(nextUpdOp);
      AffineExpr curBase, nextBase;
      int32_t curOffset, nextOffset;

      // Get the base and offset from linear access expr
      std::tie(curBase, curOffset) = getBaseAndOffset(curLinearAccess);
      std::tie(nextBase, nextOffset) = getBaseAndOffset(nextLinearAccess);
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

      if (fusedOps.size() > (resultElWidth == 16 ? 4 : 8)) {
        groupFusedOps.push_back(fusedOps);
        fusedOps.clear();
        fusedOps.push_back(nextMulOp);
        std::tie(curIdx, curUpdOp, curMulOp) = *it;
        continue;
      }
      std::tie(curIdx, curUpdOp, curMulOp) = *it;
    }
    groupFusedOps.push_back(fusedOps);
  }

  for (auto fusedOps : groupFusedOps) {
    unsigned numFusedOps = fusedOps.size();
    if (numFusedOps < 2) {
      return false;
    }
  }

  return true;
}

struct FoldMulAddChainToConvOpPattern
    : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  FoldMulAddChainToConvOpPattern(MLIRContext *context, AnalysisManager &am)
      : OpConversionPattern<arith::AddIOp>(context), am(am) {}

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

    arith::MulIOp mOp = rewriter.replaceOpWithNewOp<arith::MulIOp>(
        srcOp, adaptor.getLhs(), adaptor.getLhs());

    //  arith::AddIOp aOp= rewriter.replaceOpWithNewOp<arith::AddIOp>(
    //        srcOp, mOp->getOperand(0), mOp->getOperand(0));

    /*
        for (auto fusedOps : groupFusedOps) {
          arith::MulIOp curMulOp = (*fusedOps.begin());
          arith::AddIOp curAddOp =
              cast<arith::AddIOp>(*curMulOp->getUsers().begin());
          Operation *addLhs =
              dyn_cast<arith::MulIOp>(curAddOp->getOperand(0).getDefiningOp());
          Operation *addRhs =
              dyn_cast<arith::MulIOp>(curAddOp->getOperand(1).getDefiningOp());
          bool isMulConv = false;
          Value acc = nullptr;

          if (addLhs && addRhs) {
            isMulConv = true;
          } else {
            acc = addLhs ? curAddOp->getOperand(1) : curAddOp->getOperand(0);
          }

          Value lhs = curMulOp->getOperand(0);
          Value rhs = curMulOp->getOperand(1);
          VectorType vType = curMulOp.getResult().getType().cast<VectorType>();
          Type sType = vType.getElementType();
          IntegerType iType = sType.cast<IntegerType>();
          unsigned width = iType.getWidth() <= 8 ? 32 : 64;
          int32_t M = iType.getWidth() == 8 ? 32 : 16;
          int32_t N = iType.getWidth() == 8 ? 8 : 4;

          Type ctype = mlir::IntegerType::get(iType.getContext(), width);
          Type opType = VectorType::get(vType.getShape(), ctype);

          aievec::BroadcastOp bcastOp =
       cast<aievec::BroadcastOp>(rhs.getDefiningOp()); aievec::UPDOp bcastUPDOp
       = cast<aievec::UPDOp>(bcastOp.getSource().getDefiningOp());
          SmallVector<Value, 4> indices(bcastUPDOp.getIndices().begin(),
                                        bcastUPDOp.getIndices().end());
          unsigned lanes = 512 / getElementSizeInBits(vType);
          VectorType resType = createVectorType(lanes, sType);
          Value innerMostIdx = indices[indices.size() - 1];
          Value newIdx = innerMostIdx;
          int64_t val = -1;
          int64_t defIdx = -1;
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

          if (vType != resType) {
            newBcastOp = rewriter.replaceOpWithNewOp<aievec::UPDOp>(
                bcastUPDOp, resType, bcastUPDOp.getSource(), indices, 0, 0,
                TypedValue<VectorType>(nullptr));
          }

          Operation *shuffleOp = newBcastOp;
          if (dupFactor != 1) {
            shuffleOp = rewriter.create<aievec::ShuffleOp>(
                newBcastOp.getLoc(), resType, newBcastOp.getResult(), 0);
          }

          int32_t shiftBytes = (bcastOp.getIdx() + val) *
                               getElementSizeInBits(vType) / 8 / dupFactor;

          rhs = shuffleOp->getResult(0);

          if (shiftBytes) {
            SmallVector<Value> sources = {shuffleOp->getResult(0)};

            rhs = rewriter.create<aievec::ShiftOp>(
                shuffleOp->getLoc(),
       sources.back().getType().cast<VectorType>(), sources, shiftBytes);
          }

          aievec::UPDOp lUPDOp = cast<aievec::UPDOp>(lhs.getDefiningOp());
          SmallVector<Value, 8> lIndices;
          lIndices.append(lUPDOp.getIndices().begin(),
       lUPDOp.getIndices().end());

          lhs = rewriter.replaceOpWithNewOp<aievec::UPDOp>(
              lUPDOp, resType, lUPDOp.getSource(), lIndices, 0, 0,
              TypedValue<VectorType>(nullptr));

          auto notMulOrAddOp = [&](Operation *op) {
            return !isa<arith::MulIOp, arith::AddIOp>(op);
          };

          arith::AddIOp lastAddOp =
              cast<arith::AddIOp>(*(fusedOps.back()->getUsers().begin()));

          aievec::MulConvOp convOp =
       rewriter.create<aievec::MulConvOp>(srcOp.getLoc(), opType, lhs, rhs, M,
       N); rewriter.replaceOpWithNewOp<aievec::SRSOp>( srcOp, vType,
       convOp.getResult());
        }*/

    return success();
  }
  AnalysisManager &am;
};

//===----------------------------------------------------------------------===//
// Pattern collection
//===----------------------------------------------------------------------===//

static void populateAIEVecCommonConversionPatterns(RewritePatternSet &patterns,
                                                   AnalysisManager &am) {
  patterns.add<LowerVectorAddFOpToAIEVecAddOp, LowerVectorSubIOpToAIEVecSubOp,
               LowerVectorSubFOpToAIEVecSubOp>(patterns.getContext());
}

static void populateAIEVecV1ConversionPatterns(RewritePatternSet &patterns,
                                               AnalysisManager &am) {
  patterns.add<LowerVectorTransferReadToAIEUPD, SplitUPDOpOnAccPattern>(
      patterns.getContext(), am, 256);
  patterns.add<ConvertMulAddToAIEVecFMAOpPattern, FoldBroadcastToFMAOp,
               LowerVectorAddIOpToAIEVecAddOp>(patterns.getContext());
}

static void populateAIEVecV2ConversionPatterns(RewritePatternSet &patterns,
                                               AnalysisManager &am) {
  patterns.add<LowerVectorTransferReadToAIEUPD, SplitUPDOpOnAccPattern>(
      patterns.getContext(), am, 512);

  patterns.add<LowerVectorAddIOpToAIEVecAddOp,
               FoldVectorExtractAndBroadcastToAIEBroadcast,
               ConvertMulAddToAIEVecFMAElemOpPattern>(patterns.getContext());
}

static void
populateAIEVecV1TransformationPatterns(RewritePatternSet &patterns) {}

static void
populateAIEVecV2TransformationPatterns(RewritePatternSet &patterns) {
  patterns.add<FoldAIEShiftAndBroadcast>(patterns.getContext());
}

static void
populateAIEVecConvOpTransformationPatterns(RewritePatternSet &patterns,
                                           AnalysisManager &am) {
  patterns.add<FoldMulAddChainToConvOpPattern>(patterns.getContext(), am);
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
  // target.addDynamicallyLegalOp<arith::AddIOp>(
  //    [](arith::AddIOp op) { return !isa<VectorType>(op.getType()); });
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
    vector::BroadcastOp srcBcast = nullptr;
    auto lhsOp = op.getLhs().getDefiningOp();
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
  target.addDynamicallyLegalOp<aievec::UPDOp>([&am](aievec::UPDOp op) {
    return am.getChildAnalysis<UPDOpEffectiveAccessSizeAnalysis>(op)
               .effectiveSize <= 1024;
  });
}

static void
configureAIEVecV1TransformationLegalizations(ConversionTarget &target) {}

static void
configureAIEVecV2TransformationLegalizations(ConversionTarget &target) {
  target.addDynamicallyLegalOp<xilinx::aievec::BroadcastOp>(
      [](xilinx::aievec::BroadcastOp op) {
        aievec::ShiftOp shiftOp = nullptr;
        int32_t idx = 0;
        return !canFoldAIEShiftAndBroadcast(op, shiftOp, idx);
      });
}

static void
configureAIEVecConvOpTransformationLegalizations(ConversionTarget &target,
                                                 AnalysisManager &am) {
  target.addDynamicallyLegalOp<arith::AddIOp>([](arith::AddIOp op) {
    SmallVector<SmallVector<arith::MulIOp, 8>, 8> groupFusedOps;
    MulDefMapTy macChainMap;
    unsigned dupFactor = 1;
    return !canFoldMulAddChainToConvOp(op, macChainMap, groupFusedOps,
                                       dupFactor);
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
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant();
      });
}

static void configureAIEv1CanonicalizeLegalizations(ConversionTarget &target) {}

static void configureAIEMLCanonicalizeLegalizations(ConversionTarget &target) {}

static void
populateCommonAIECanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertSplatTransferReadToBroadcastPattern>(
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
    populateAIEVecConvOpTransformationPatterns(patterns, am);
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

  // Add AIEVec transformation pass
  pm.addPass(
      createAIEVecTransformation(options.getAIEVecTransformationOptions()));

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

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
