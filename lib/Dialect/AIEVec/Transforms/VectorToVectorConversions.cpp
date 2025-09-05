//===-VectorToVectorConversions.cpp - Conversions within Vector -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// This file contains conversions and rewrites to the Vector dialect to make
// it compatible with the available vector instructions in AIE architectures
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/Utils/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>

#define DEBUG_TYPE "aievec-canonicalization"

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

//============================================================================//
//================== Common AIE canonicalization analysis ====================//
//============================================================================//

static TargetBackend decodeTargetBackend(const std::string &backend) {
  if (!backend.empty()) {
    if (backend == "llvmir")
      return TargetBackend::LLVMIR;
    if (backend != "cpp")
      return TargetBackend::UNKNOWN;
  }
  return TargetBackend::CPP;
}

static AIEArch decodeAIETarget(const std::string &target) {
  if (!target.empty()) {
    if (target == "aieml" || target == "aie2" || target == "aie2p")
      return AIEArch::AIE2;
    if (target != "aie")
      return AIEArch::UNKNOWN;
  }
  return AIEArch::AIE;
}

//============================================================================//
//================== Common AIE canonicalization analysis ====================//
//============================================================================//

static bool isGemmBTransposedContractionOp(vector::ContractionOp op) {
  if (op.getKind() != vector::CombiningKind::ADD)
    return false;

  // Get and check shape of operands
  auto lhsShape = op.getLhsType().getShape();
  auto rhsShape = op.getRhsType().getShape();
  auto accShape = cast<ShapedType>(op.getAccType()).getShape();
  if (lhsShape.size() < 2 || rhsShape.size() < 2 || accShape.size() < 2)
    return false;

  // Check that the innermost iterators match gemm-like iterators
  SmallVector<vector::IteratorType> iterators = op.getIteratorTypesArray();
  if (iterators.size() < 3)
    return false;
  auto innerMostIterators =
      SmallVector<vector::IteratorType>(iterators.end() - 3, iterators.end());
  if (vector::IteratorType::parallel != innerMostIterators[0] ||
      vector::IteratorType::parallel != innerMostIterators[1] ||
      vector::IteratorType::reduction != innerMostIterators[2])
    return false;

  // Get indexing maps of iterators for operands
  SmallVector<AffineMap, 4> indexingMaps(op.getIndexingMapsArray());
  SmallVector<int64_t> outerMostResults;
  for (int64_t i = 0; i < indexingMaps[0].getNumResults() - 2; i++)
    outerMostResults.push_back(i);

  auto innerLhsMap = indexingMaps[0].dropResults(outerMostResults);
  auto innerRhsMap = indexingMaps[1].dropResults(outerMostResults);
  auto innerAccMap = indexingMaps[2].dropResults(outerMostResults);

  // Check whether they conform to a "transposed B" gemm
  auto *ctx = op.getContext();
  auto mmAidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0, 2}, ctx)
          .dropResults(0);
  auto mmBidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{0, 1, 2}, ctx)
          .dropResults(0);
  auto mmCidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{2, 0, 1}, ctx)
          .dropResults(0);
  int64_t numOuterMostDims = indexingMaps[0].getNumDims() - 3;
  return innerLhsMap == mmAidxMap.shiftDims(numOuterMostDims) &&
         innerRhsMap == mmBidxMap.shiftDims(numOuterMostDims) &&
         innerAccMap == mmCidxMap.shiftDims(numOuterMostDims);
}

//============================================================================//
//============ Common AIE canonicalization conversion patterns ===============//
//============================================================================//

// This pattern converts a `vector.transfer_read` with an unaligned access
// into an aligned `vector.transfer_read` twice as long, followed by a
// `vector.extract_strided_slice` selecting the subvector matching the
// original `vector.transfer_read`.
struct SplitUnalignedTransferReadPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  SplitUnalignedTransferReadPattern(MLIRContext *context, int64_t maxVectorSize,
                                    int64_t alignment)
      : OpConversionPattern<vector::TransferReadOp>(context),
        maxVectorSize(maxVectorSize), vectorAlignment(alignment) {}

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check that it's not a splat transfer read.
    if (adaptor.getPermutationMap().isConstant())
      return failure();

    // Check if the transfer is unaligned.
    auto vType = readOp.getVectorType();
    int64_t offset =
        getTransferReadAlignmentOffset(adaptor, vType, vectorAlignment)
            .value_or(0);
    if (offset == 0)
      return failure();

    // Verify that we can load a vector 2x as long as the original
    auto vLen = vType.getShape().back();
    auto longVecTy = VectorType::get(2 * vLen, vType.getElementType());
    auto longVecSize = getElementSizeInBits(vType) * 2 * vLen;
    if (longVecSize > maxVectorSize)
      return failure();

    // Calculate the aligned indices for the lower and higher parts.
    // TODO: Add support for cases where the offset is greater than the
    // TODO: vector length.
    auto loc = readOp.getLoc();
    Value oldInnerMostIdx = adaptor.getIndices().back();
    auto offsetCorrectionMap =
        AffineMap::get(1, 0, getAffineDimExpr(0, readOp.getContext()) - offset);
    Value newInnerMostIdx = rewriter
                                .create<affine::AffineApplyOp>(
                                    readOp.getLoc(), offsetCorrectionMap,
                                    SmallVector<Value, 1>({oldInnerMostIdx}))
                                .getResult();
    SmallVector<Value, 8> alignedIdx;
    alignedIdx.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
    alignedIdx[alignedIdx.size() - 1] = newInnerMostIdx;

    // Create the aligned transfer read for a vector 2x as long that covers the
    // elements of the unaligned vector.
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        loc, longVecTy, adaptor.getBase(), alignedIdx, adaptor.getPadding());

    // Create a `vector.extract_strided_slice` to extract the unaligned vector.
    rewriter.replaceOpWithNewOp<vector::ExtractStridedSliceOp>(
        readOp, newReadOp.getResult(), offset, vLen, 1);

    return success();
  }

  int64_t maxVectorSize;
  int64_t vectorAlignment;
};

// This pattern converts a `vector.transfer_read` with a splat permutation map
// into a contiguous `vector.transfer_read` followed by a `vector.extract` to
// obtain the splat value and a `vector.broadcast` to broadcast it into a
// vector of the right size.
struct ConvertSplatTransferReadToBroadcastPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  ConvertSplatTransferReadToBroadcastPattern(MLIRContext *context)
      : OpConversionPattern<vector::TransferReadOp>(context) {}

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AffineMap map = readOp.getPermutationMap();
    if (!map.isConstant())
      return failure();

    Value srcMemRef = adaptor.getBase();
    SmallVector<Value, 8> indices;
    Value newIdx;
    int64_t offset = 0;
    // If it's a zero-rank memory access
    if (cast<MemRefType>(srcMemRef.getType()).getRank() == 0) {
      srcMemRef = rewriter
                      .create<memref::ExpandShapeOp>(
                          readOp.getLoc(), SmallVector<int64_t, 1>({1}),
                          srcMemRef, SmallVector<ReassociationIndices, 1>({}))
                      .getResult();
      newIdx = rewriter.create<arith::ConstantOp>(readOp.getLoc(),
                                                  rewriter.getIndexAttr(0L));
      indices.push_back(newIdx);
    } else {
      indices.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
      newIdx = indices[indices.size() - 1];
      // If the innermost index comes from an `affine.apply` op, take the base
      // as the new innermost index for the new `vector.transfer_read`, and the
      // offset as the index for the `aievec.broadcast` op.
      if (auto applyOp = newIdx.getDefiningOp<affine::AffineApplyOp>())
        if (applyOp.getAffineMap().getNumDims() == 1) {
          newIdx = applyOp.getMapOperands()[0];
          offset = applyOp.getAffineMap().compose(ArrayRef<int64_t>{0})[0];
        }
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
      newIdx =
          rewriter
              .create<affine::AffineApplyOp>(readOp.getLoc(), newAddrMap,
                                             SmallVector<Value, 1>({newIdx}))
              .getResult();
    }
    indices[indices.size() - 1] = newIdx;
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), readOp.getVector().getType(), srcMemRef, indices,
        adaptor.getPadding());
    auto extractOp = rewriter.create<vector::ExtractOp>(
        readOp.getLoc(), newReadOp.getResult(), ArrayRef<int64_t>{offset});
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        readOp, newReadOp.getVector().getType(), extractOp.getResult());
    return success();
  }
};

// This pattern moves cast operations as close as possible to the source of
// the data. This helps to simplify dealing with patterns that may vary only
// by these sorts of casts between data manipulation operations and arithmetic
// ops.
// TODO: Generalize this op and instantiate for different types of cast ops.
struct HoistCastOpToDataSourcePattern : public RewritePattern {
  HoistCastOpToDataSourcePattern(MLIRContext *context)
      : RewritePattern(arith::ExtSIOp::getOperationName(), /*benefit=*/1,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    arith::ExtSIOp extOp = cast<arith::ExtSIOp>(op);
    Operation *defOp = extOp.getIn().getDefiningOp();
    // If it's a data source op, we're done.
    if (!defOp || isa<vector::TransferReadOp, memref::LoadOp,
                      affine::AffineLoadOp, func::CallOp>(defOp))
      return failure();

    // At the moment, we only accept ops we know we can swap with cast.
    if (!isa<vector::BroadcastOp, vector::ExtractOp,
             vector::ExtractStridedSliceOp>(defOp))
      return failure();

    Type extOpInTy = extOp.getIn().getType();
    SmallVector<Value, 4> inputs;
    for (Value operand : defOp->getOperands()) {
      Type operandTy = operand.getType();
      VectorType extOpInVecTy = dyn_cast<VectorType>(extOpInTy);
      VectorType operandVecTy = dyn_cast<VectorType>(operandTy);
      if (operandTy == extOpInTy) {
        Type outTy = extOp.getOut().getType();
        inputs.push_back(
            rewriter.create<arith::ExtSIOp>(extOp.getLoc(), outTy, operand)
                .getOut());
      } else if (extOpInVecTy && extOpInVecTy.getElementType() == operandTy) {
        // Promote from vector to scalar -> scalar conversion for this operand
        Type outTy =
            cast<VectorType>(extOp.getOut().getType()).getElementType();
        inputs.push_back(
            rewriter.create<arith::ExtSIOp>(extOp.getLoc(), outTy, operand)
                .getOut());
      } else if (operandVecTy && operandVecTy.getElementType() == extOpInTy) {
        // Promote from scalar to vector -> vector conversion for this operand
        Type outTy =
            VectorType::get(operandVecTy.getShape(), extOp.getOut().getType());
        inputs.push_back(
            rewriter.create<arith::ExtSIOp>(extOp.getLoc(), outTy, operand)
                .getOut());
      } else if (extOpInVecTy && operandVecTy &&
                 (extOpInVecTy.getElementType() ==
                  operandVecTy.getElementType())) {
        // Hoist through a vector shape change
        Type outTy = VectorType::get(
            operandVecTy.getShape(),
            cast<VectorType>(extOp.getOut().getType()).getElementType());
        inputs.push_back(
            rewriter.create<arith::ExtSIOp>(extOp.getLoc(), outTy, operand)
                .getOut());
      } else {
        inputs.push_back(operand);
      }
    }

    auto *newOp =
        rewriter.create(extOp->getLoc(), defOp->getName().getIdentifier(),
                        inputs, {extOp.getOut().getType()}, defOp->getAttrs());
    rewriter.replaceOp(extOp, newOp->getResult(0));
    return success();
  }
};

// This pattern swaps a UnaryOpA followed by UnaryOpB. This pattern can be used
// to improve pattern matching for mixed-type arithmetic ops, by getting sign
// extension ops closer to the single-type arithmetic operations.
template <class UnaryOpA, class UnaryOpB>
struct SwapUnaryOpsPattern : public OpRewritePattern<UnaryOpB> {
  using OpRewritePattern<UnaryOpB>::OpRewritePattern;
  // This function takes the chain of operations A->B, and returns the new type
  // between B and A after the swap.
  using InferTypeB2AFnTy = std::function<Type(UnaryOpA aOp, UnaryOpB bOp)>;
  InferTypeB2AFnTy inferTypeB2A = nullptr;

  SwapUnaryOpsPattern(MLIRContext *context, InferTypeB2AFnTy inferType)
      : OpRewritePattern<UnaryOpB>(context), inferTypeB2A(inferType) {}

  LogicalResult matchAndRewrite(UnaryOpB bOp,
                                PatternRewriter &rewriter) const override {
    static_assert(
        UnaryOpA::template hasTrait<OpTrait::OneOperand>(),
        "SwapUnaryOps can only be instantiated for single-operand ops");
    static_assert(
        UnaryOpB::template hasTrait<OpTrait::OneOperand>(),
        "SwapUnaryOps can only be instantiated for single-operand ops");
    UnaryOpA aOp = bOp.getOperand().template getDefiningOp<UnaryOpA>();
    if (!aOp)
      return rewriter.notifyMatchFailure(bOp, UnaryOpB::getOperationName() +
                                                  " not preceeded by " +
                                                  UnaryOpA::getOperationName());

    Type newA2BTy = inferTypeB2A(aOp, bOp);

    auto newA =
        rewriter.create<UnaryOpB>(bOp->getLoc(), SmallVector<Type>({newA2BTy}),
                                  aOp->getOperands(), bOp->getAttrs());
    auto newB = rewriter.create<UnaryOpA>(
        bOp->getLoc(), SmallVector<Type>({bOp.getResult().getType()}),
        newA->getResults(), aOp->getAttrs());
    rewriter.replaceOp(bOp, newB.getResult());
    return success();
  }
};

static SmallVector<Value> collapseInnerMostDimIndices(PatternRewriter &b,
                                                      Location loc, int numDims,
                                                      ValueRange indices,
                                                      ArrayRef<int64_t> shape,
                                                      AffineMap layout) {
  // TODO: Don't assume trivial layout
  assert(layout.isMinorIdentity() &&
         "dimension collapse in non-identity layout is not implemented");
  auto newIdxExpr = b.getAffineDimExpr(numDims - 1);
  int64_t stride = 1;
  for (int64_t dim = numDims - 2; dim >= 0; dim--) {
    stride *= shape[shape.size() - (numDims - dim - 1)];
    newIdxExpr = newIdxExpr + b.getAffineDimExpr(dim) * stride;
  }
  auto newIndexMap = AffineMap::get(numDims, 0, newIdxExpr);
  Value newInnerMostIdxValue =
      b.create<affine::AffineApplyOp>(loc, newIndexMap,
                                      indices.take_back(numDims))
          .getResult();
  SmallVector<Value> newIdxRange;
  for (auto idx : indices.drop_back(numDims))
    newIdxRange.push_back(idx);
  newIdxRange.push_back(newInnerMostIdxValue);
  return newIdxRange;
}

static Value collapseInnerMostShapeDims(PatternRewriter &b, Location loc,
                                        int numDims, Value val) {
  auto memRefTy = cast<MemRefType>(val.getType());
  auto shape = memRefTy.getShape();
  int64_t newInnerMostDim = std::accumulate(shape.end() - numDims, shape.end(),
                                            1, std::multiplies<>());
  SmallVector<int64_t, 4> newShape{shape.begin(), shape.end() - numDims + 1};
  newShape[shape.size() - numDims] = newInnerMostDim;
  auto newNumDims = newShape.size();
  auto *ctx = b.getContext();
  auto newMemRefTy = MemRefType::get(
      newShape, memRefTy.getElementType(),
      AffineMap::getMinorIdentityMap(newNumDims, newNumDims, ctx),
      memRefTy.getMemorySpace());
  auto reassocIndices =
      getReassociationIndicesForCollapse(shape, newShape).value();
  return b
      .create<memref::CollapseShapeOp>(loc, newMemRefTy, val, reassocIndices)
      .getResult();
}

// This pattern flatten multidimensional `vector.transfer_read` operations
// replacing them with a `memref.collapse_shape`, a 1D `vector.transfer_read`,
// and a `vector.shape_cast`.
struct FlattenMultDimTransferReadPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We can only deal with unmasked transfer ops with an identity permutation
    // map.
    if (!adaptor.getPermutationMap().isMinorIdentity() || adaptor.getMask())
      return failure();
    VectorType vectorTy = readOp.getVector().getType();
    if (vectorTy.getRank() < 2)
      return failure();
    // Work only on bufferized reads
    MemRefType memRefTy = dyn_cast<MemRefType>(adaptor.getBase().getType());
    if (!memRefTy)
      return failure();
    auto memRefShape = memRefTy.getShape();
    auto vecShape = vectorTy.getShape();

    auto newVectorTy =
        VectorType::get({std::accumulate(vecShape.begin(), vecShape.end(), 1,
                                         std::multiplies<>())},
                        vectorTy.getElementType());
    AffineMap layout = memRefTy.getLayout().getAffineMap();
    auto newIndices =
        collapseInnerMostDimIndices(rewriter, readOp.getLoc(), vecShape.size(),
                                    adaptor.getIndices(), memRefShape, layout);
    auto newSource = collapseInnerMostShapeDims(
        rewriter, readOp.getLoc(), vecShape.size(), adaptor.getBase());
    auto newVector = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), newVectorTy, newSource, newIndices,
        /*padding*/
        arith::getZeroConstant(rewriter, readOp.getLoc(),
                               newVectorTy.getElementType()));

    auto inBoundsArrayAttrOpt = adaptor.getInBounds();
    if (inBoundsArrayAttrOpt) {
      SmallVector<bool> inBounds =
          llvm::to_vector(inBoundsArrayAttrOpt.getAsValueRange<BoolAttr>());
      SmallVector<bool> newInBounds({false});
      newInBounds[0] = std::all_of(inBounds.begin(), inBounds.end(),
                                   [](bool v) { return v; });
      newVector.getProperties().setInBounds(
          rewriter.getBoolArrayAttr(newInBounds));
    }

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(readOp, vectorTy,
                                                     newVector);

    return success();
  }
};

// This pattern flatten multidimensional `vector.transfer_write` operations
// replacing them with a `memref.collapse_shape`, a `vector.shape_cast`, and a
// 1D `vector.transfer_write`,
struct FlattenMultDimTransferWritePattern
    : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferWriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We can only deal with unmasked transfer ops with an identity permutation
    // map.
    if (!adaptor.getPermutationMap().isMinorIdentity() || adaptor.getMask())
      return failure();
    VectorType vectorTy = cast<VectorType>(adaptor.getValueToStore().getType());
    if (vectorTy.getRank() < 2)
      return failure();
    // Work only on bufferized reads
    MemRefType memRefTy = dyn_cast<MemRefType>(adaptor.getBase().getType());
    if (!memRefTy)
      return failure();
    auto memRefShape = memRefTy.getShape();
    auto vecShape = vectorTy.getShape();

    auto newVectorTy =
        VectorType::get({std::accumulate(vecShape.begin(), vecShape.end(), 1,
                                         std::multiplies<>())},
                        vectorTy.getElementType());
    AffineMap layout = memRefTy.getLayout().getAffineMap();
    auto newVector =
        rewriter
            .create<vector::ShapeCastOp>(writeOp.getLoc(), newVectorTy,
                                         adaptor.getValueToStore())
            .getResult();
    auto newIndices =
        collapseInnerMostDimIndices(rewriter, writeOp.getLoc(), vecShape.size(),
                                    adaptor.getIndices(), memRefShape, layout);
    auto newSource = collapseInnerMostShapeDims(
        rewriter, writeOp.getLoc(), vecShape.size(), adaptor.getBase());

    auto newOp = rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        writeOp, newVector, newSource, newIndices);

    auto inBoundsArrayAttrOpt = adaptor.getInBounds();
    if (inBoundsArrayAttrOpt) {
      SmallVector<bool> inBounds =
          llvm::to_vector(inBoundsArrayAttrOpt.getAsValueRange<BoolAttr>());
      SmallVector<bool> newInBounds({false});
      newInBounds[0] = std::all_of(inBounds.begin(), inBounds.end(),
                                   [](bool v) { return v; });
      newOp.getProperties().setInBounds(rewriter.getBoolArrayAttr(newInBounds));
    }

    return success();
  }
};

// This pattern extracts an implicit transposition of the 2 innermost
// dimensions of `rhs` in a gemm-like contraction op, making it an explicit
// `vector.transpose` op.
// If `rhs` is coming from a widening op (`extf`/`extsi`/`extui`), the
// transposition will be hoisted above the widening op.
struct ExtractTransposeFromContractionOp
    : public OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern<vector::ContractionOp>::OpConversionPattern;

  static VectorType getTransposedVectorType(VectorType vecTy) {
    SmallVector<int64_t> shape{vecTy.getShape()};
    auto nDim = shape.size();
    int64_t dimNm1 = shape[nDim - 1];
    shape[nDim - 1] = shape[nDim - 2];
    shape[nDim - 2] = dimNm1;
    auto elemTy = vecTy.getElementType();
    return VectorType::get(shape, elemTy);
  }

  LogicalResult
  matchAndRewrite(vector::ContractionOp contractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isGemmBTransposedContractionOp(contractOp))
      return failure();

    Location loc = contractOp.getLoc();
    auto *ctx = rewriter.getContext();

    Value rhsVal = adaptor.getRhs();
    VectorType rhsVecTy = contractOp.getRhsType();
    Type rhsElemTy = rhsVecTy.getElementType();

    bool doExtF = false, doExtSI = false, doExtUI = false;
    if (auto extfRhsOp = rhsVal.getDefiningOp<arith::ExtFOp>()) {
      rhsVal = extfRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtF = true;
    } else if (auto extsiRhsOp = rhsVal.getDefiningOp<arith::ExtSIOp>()) {
      rhsVal = extsiRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtSI = true;
    } else if (auto extuiRhsOp = rhsVal.getDefiningOp<arith::ExtUIOp>()) {
      rhsVal = extuiRhsOp.getIn();
      rhsVecTy = cast<VectorType>(rhsVal.getType());
      doExtUI = true;
    }

    int64_t nDim = rhsVecTy.getShape().size();
    SmallVector<int64_t> rhsPermutation;
    for (int64_t i = 0; i < nDim - 2; i++)
      rhsPermutation.push_back(i);
    rhsPermutation.push_back(nDim - 1);
    rhsPermutation.push_back(nDim - 2);
    auto transpRhsVecTy = getTransposedVectorType(rhsVecTy);
    rhsVal = rewriter
                 .create<vector::TransposeOp>(loc, transpRhsVecTy, rhsVal,
                                              rhsPermutation)
                 .getResult();

    if (doExtF)
      rhsVal =
          rewriter
              .create<arith::ExtFOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();
    if (doExtSI)
      rhsVal =
          rewriter
              .create<arith::ExtSIOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();
    if (doExtUI)
      rhsVal =
          rewriter
              .create<arith::ExtUIOp>(
                  loc, VectorType::get(transpRhsVecTy.getShape(), rhsElemTy),
                  rhsVal)
              .getOut();

    SmallVector<AffineMap, 4> oldIdxMaps(contractOp.getIndexingMapsArray());

    nDim = oldIdxMaps[1].getNumDims();
    SmallVector<int64_t> innerDimPerm;
    for (int64_t i = 0; i < nDim - 2; i++)
      innerDimPerm.push_back(i);
    innerDimPerm.push_back(nDim - 1);
    innerDimPerm.push_back(nDim - 2);
    auto transpPermMap = AffineMap::getPermutationMap(innerDimPerm, ctx);

    auto newIdxMaps = rewriter.getAffineMapArrayAttr(
        {oldIdxMaps[0], oldIdxMaps[1].compose(transpPermMap), oldIdxMaps[2]});

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, contractOp.getResult().getType(), adaptor.getLhs(), rhsVal,
        adaptor.getAcc(), newIdxMaps, contractOp.getIteratorTypes());

    return success();
  }
};

/// Utility function to check if all provided indices are constant zero values.
/// @return success() if all indices are constant zeros, failure() otherwise
static LogicalResult isAllZeroOffsetAccess(mlir::OperandRange indices) {
  if (!llvm::all_of(indices, [](Value val) {
        IntegerAttr attr;
        if (!matchPattern(val, m_Constant(&attr)))
          return false;
        return attr.getInt() == 0;
      })) {
    return failure();
  }
  return success();
}

/// Utility function to convert OpFoldResult offsets from a SubView operation
/// into a vector of Values.
static SmallVector<Value> opFoldResultsToValues(PatternRewriter &rewriter,
                                                Location loc,
                                                memref::SubViewOp subViewOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(subViewOp);
  SmallVector<Value> newIndices;
  for (OpFoldResult offset : subViewOp.getMixedOffsets()) {
    Value indexVal;
    if (auto attr = dyn_cast<Attribute>(offset)) {
      indexVal = rewriter.create<arith::ConstantIndexOp>(
          loc, cast<IntegerAttr>(attr).getInt());
    } else {
      indexVal = cast<Value>(offset);
    }
    newIndices.push_back(indexVal);
  }
  return newIndices;
}

/// Pattern to canonicalize trivial vector.transfer_read operations on subviews.
///
/// This pattern eliminates unnecessary memref.subview operations when the
/// transfer_read accesses the subview with all-zero indices. It transforms:
///
/// INPUT:
///   %subview = memref.subview %memref [offset0, offset1, ...]
///   %result = vector.transfer_read %subview[0, 0, ...]
///
/// OUTPUT:
///   %result = vector.transfer_read %memref[offset0, offset1, ...]
///
/// The pattern only matches when:
/// - The base of transfer_read is defined by a memref.subview operation
/// - All indices in the transfer_read are constant zeros
struct CanonicalizeTrivialReadAccessSubviewOpPattern
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    // Check if the base memref comes from a subview operation
    auto subViewOp = dyn_cast_if_present<memref::SubViewOp>(
        readOp.getBase().getDefiningOp());
    if (!subViewOp)
      return failure();

    // Verify that all access indices are zero
    if (failed(isAllZeroOffsetAccess(readOp.getIndices())))
      return failure();

    // Convert subview offsets to explicit index values
    SmallVector<Value> newIndices =
        opFoldResultsToValues(rewriter, readOp.getLoc(), subViewOp);

    // Replace with direct access to the original memref using subview offsets
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        readOp, readOp.getType(), subViewOp.getSource(), newIndices,
        readOp.getPadding(), readOp.getInBoundsValues());
    return success();
  }
};

/// Pattern to canonicalize trivial vector.transfer_write operations on
/// subviews.
///
/// This pattern eliminates unnecessary memref.subview operations when the
/// transfer_write accesses the subview with all-zero indices. It transforms:
///
/// INPUT:
///   %subview = memref.subview %memref [offset0, offset1, ...]
///   vector.transfer_write %value, %subview[0, 0, ...]
///
/// OUTPUT:
///   vector.transfer_write %value, %memref[offset0, offset1, ...]
///
/// The pattern only matches when:
/// - The base of transfer_write is defined by a memref.subview operation
/// - All indices in the transfer_write are constant zeros
struct CanonicalizeTrivialWriteAccessSubviewOpPattern
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the base memref comes from a subview operation
    auto subViewOp = dyn_cast_if_present<memref::SubViewOp>(
        writeOp.getBase().getDefiningOp());
    if (!subViewOp)
      return failure();

    // Verify that all access indices are zero
    if (failed(isAllZeroOffsetAccess(writeOp.getIndices())))
      return failure();

    // Convert subview offsets to explicit index values
    SmallVector<Value> newIndices =
        opFoldResultsToValues(rewriter, writeOp.getLoc(), subViewOp);

    // Create new transfer_write with direct access to original memref
    rewriter.create<vector::TransferWriteOp>(
        writeOp.getLoc(), writeOp.getVector(), subViewOp.getSource(),
        newIndices, writeOp.getInBoundsValues());

    // Remove the original transfer_write operation
    rewriter.eraseOp(writeOp);
    return success();
  }
};

//============================================================================//
//============ AIE2 canonicalization conversion patterns ===============//
//============================================================================//

struct ConvertLeadingUnitDimInsertToReshapePattern
    : public OpRewritePattern<vector::InsertOp> {

  using OpRewritePattern<vector::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertOp insOp,
                                PatternRewriter &rewriter) const override {
    auto insSrcTy = dyn_cast<VectorType>(insOp.getValueToStoreType());
    if (!insSrcTy)
      return failure();

    auto srcShape = insSrcTy.getShape();
    auto dstShape = insOp.getDestVectorType().getShape();

    unsigned long numLeadUnitDimDst = 0;
    while (numLeadUnitDimDst < dstShape.size() &&
           dstShape[numLeadUnitDimDst] == 1)
      numLeadUnitDimDst++;

    if (!numLeadUnitDimDst)
      return failure();

    unsigned long numLeadUnitDimSrc = 0;
    while (numLeadUnitDimSrc < srcShape.size() &&
           srcShape[numLeadUnitDimSrc] == 1)
      numLeadUnitDimSrc++;

    SmallVector<int64_t> nonLeadUnitDimDstShape(
        dstShape.begin() + numLeadUnitDimDst, dstShape.end());
    SmallVector<int64_t> nonLeadUnitDimSrcShape(
        srcShape.begin() + numLeadUnitDimSrc, srcShape.end());

    if (nonLeadUnitDimSrcShape != nonLeadUnitDimDstShape)
      return failure();

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        insOp, insOp.getDestVectorType(), insOp.getValueToStore());
    return success();
  }
};

//============================================================================//
//================ Common AIE canonicalization configuration =================//
//============================================================================//
static void
configureCommonAIECanonicalizeLegalizations(ConversionTarget &target,
                                            TargetBackend backend) {
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         memref::MemRefDialect, vector::VectorDialect,
                         ub::UBDialect>();
}

static void
populateCommonAIECanonicalizeConversionPatterns(RewritePatternSet &patterns,
                                                TargetBackend backend) {
  patterns.add<ConvertSplatTransferReadToBroadcastPattern>(
      patterns.getContext());
}

//============================================================================//
//============== AIEv1-specific canonicalization configuration ===============//
//============================================================================//

static void configureAIEv1CanonicalizeLegalizations(ConversionTarget &target,
                                                    TargetBackend backend) {
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant() &&
               getTransferReadAlignmentOffset(op, op.getVectorType(), 128)
                       .value_or(0) == 0;
      });
}

static void
populateAIEv1CanonicalizeConversionPatterns(RewritePatternSet &patterns,
                                            TargetBackend backend) {
  patterns.add<SplitUnalignedTransferReadPattern>(patterns.getContext(), 512,
                                                  128);
}

//============================================================================//
//============== AIE2-specific canonicalization configuration ===============//
//============================================================================//

static void configureAIE2CanonicalizeLegalizations(ConversionTarget &target,
                                                   TargetBackend backend) {
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant() &&
               getTransferReadAlignmentOffset(op, op.getVectorType(), 256)
                       .value_or(0) == 0 &&
               op.getVector().getType().getRank() < 2;
      });
  target.addDynamicallyLegalOp<vector::TransferWriteOp>(
      [](vector::TransferWriteOp op) {
        return cast<VectorType>(op.getVector().getType()).getRank() < 2;
      });
  target.addDynamicallyLegalOp<vector::ContractionOp>(
      [](vector::ContractionOp op) {
        return !isGemmBTransposedContractionOp(op);
      });
}

static void
populateAIE2CanonicalizeConversionPatterns(RewritePatternSet &patterns,
                                           TargetBackend backend) {
  patterns.add<SplitUnalignedTransferReadPattern>(patterns.getContext(), 1024,
                                                  256);
  patterns
      .add<ExtractTransposeFromContractionOp, FlattenMultDimTransferReadPattern,
           FlattenMultDimTransferWritePattern>(patterns.getContext());
}

//============================================================================//
//=================== Common AIE Canonicalization Passes =====================//
//============================================================================//

struct VectorBroadcastLoweringPass
    : public PassWrapper<VectorBroadcastLoweringPass, OperationPass<>> {

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateVectorBroadcastLoweringPatterns(patterns);
    patterns.add<ConvertLeadingUnitDimInsertToReshapePattern>(
        patterns.getContext());

    (void)applyPatternsGreedily(op, std::move(patterns));
  }
};

static std::unique_ptr<::mlir::Pass> createVectorBroadcastLoweringPass() {
  return std::make_unique<VectorBroadcastLoweringPass>();
}

// This pass converts standard vector ops into a subset of `Vector` ops more
// amenable to being converted to `AIEVec`. So far, this process consists of
// two steps:
//    1) Replace splat transfer reads with contiguous transfer reads followed
//       by `extract` + `splat` operations.
//    2) Split unaligned transfer reads into a wider aligned transfer read
//       followed by a `vector.extract_strided_slice` operation.
struct CanonicalizeVectorForAIEVecPass
    : public PassWrapper<CanonicalizeVectorForAIEVecPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeVectorForAIEVecPass)

  CanonicalizeVectorForAIEVecPass() = default;
  CanonicalizeVectorForAIEVecPass(const CanonicalizeVectorForAIEVecPass &pass)
      : PassWrapper(pass) {}

  CanonicalizeVectorForAIEVecPass(
      const CanonicalizeVectorForAIEVecOptions &options)
      : CanonicalizeVectorForAIEVecPass() {
    aieTarget = options.aieTarget;
    targetBackend = options.targetBackend;
  }

  // In case we want to register this pass as a standalone pass for test
  // purposes.
  StringRef getArgument() const final {
    return "test-canonicalize-vector-for-aievec";
  }

  StringRef getDescription() const final {
    return "Canonicalize vector operations for AIEVec conversion";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect,
                vector::VectorDialect, affine::AffineDialect, ub::UBDialect>();
  }

  Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc(
          "Select AIE version: \"aie\", \"aie2\", or \"aie2p\". This will "
          "determine the vector size and available operations."),
      llvm::cl::init("aie")};

  Option<std::string> targetBackend{
      *this, "target-backend",
      llvm::cl::desc("Select translation backend: \"cpp\" or \"llvmir\". This "
                     "will determine the aievec operations used to convert "
                     "from vector dialect."),
      llvm::cl::init("cpp")};

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    AIEArch aieVersion = decodeAIETarget(aieTarget);
    if (aieVersion == AIEArch::UNKNOWN) {
      op->emitError() << "unknown AIE target '" << aieTarget << "'";
      signalPassFailure();
      return;
    }

    TargetBackend backend = decodeTargetBackend(targetBackend);
    if (backend == TargetBackend::UNKNOWN) {
      op->emitError() << "unknown target backend '" << targetBackend << "'";
      signalPassFailure();
      return;
    }
    if (backend == TargetBackend::LLVMIR && aieVersion == AIEArch::AIE) {
      op->emitError() << "targetting LLVM IR is not supported for AIEv1";
      signalPassFailure();
      return;
    }

    populateCommonAIECanonicalizeConversionPatterns(patterns, backend);
    configureCommonAIECanonicalizeLegalizations(target, backend);
    if (aieVersion == AIEArch::AIE) {
      populateAIEv1CanonicalizeConversionPatterns(patterns, backend);
      configureAIEv1CanonicalizeLegalizations(target, backend);
    } else {
      populateAIE2CanonicalizeConversionPatterns(patterns, backend);
      configureAIE2CanonicalizeLegalizations(target, backend);
    }

    {
      RewritePatternSet patterns(context);
      patterns.add<CanonicalizeTrivialReadAccessSubviewOpPattern,
                   CanonicalizeTrivialWriteAccessSubviewOpPattern>(context);
      (void)applyPatternsGreedily(op, std::move(patterns));
    }

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static std::unique_ptr<::mlir::Pass> createCanonicalizeVectorForAIEVecPass(
    const CanonicalizeVectorForAIEVecOptions &options) {
  return std::make_unique<CanonicalizeVectorForAIEVecPass>(options);
}

struct HoistCastOpToDataSourcePass
    : public PassWrapper<HoistCastOpToDataSourcePass, OperationPass<>> {

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<HoistCastOpToDataSourcePattern>(patterns.getContext());

    (void)applyPatternsGreedily(op, std::move(patterns));
  }
};

static std::unique_ptr<::mlir::Pass> createHoistCastOpToDataSourcePass() {
  return std::make_unique<HoistCastOpToDataSourcePass>();
}

struct ReorderOperationsPass
    : public PassWrapper<ReorderOperationsPass, OperationPass<>> {

  void runOnOperation() override {
    auto *op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<SwapUnaryOpsPattern<arith::ExtSIOp, vector::BroadcastOp>>(
        patterns.getContext(),
        [](arith::ExtSIOp extOp, vector::BroadcastOp bcastOp) -> Type {
          Type extInElemTy = extOp.getIn().getType();
          auto extInVecTy = dyn_cast<VectorType>(extInElemTy);
          if (extInVecTy)
            extInElemTy = extInVecTy.getElementType();
          return VectorType::get(bcastOp.getResultVectorType().getShape(),
                                 extInElemTy);
        });

    (void)applyPatternsGreedily(op, std::move(patterns));
  }
};

static std::unique_ptr<::mlir::Pass> createReorderOperationsPass() {
  return std::make_unique<ReorderOperationsPass>();
}

//============================================================================//
//=============== Main Vector2Vector Pipeline Configuration ==================//
//============================================================================//

void xilinx::aievec::buildCanonicalizeVectorForAIEVec(
    OpPassManager &pm, const CanonicalizeVectorForAIEVecOptions &options) {
  // Add `Vector` code canonicalization passes
  // TODO: Add passes to unroll vector with unsupported types
  // TODO: Add passes to split vectors that won't fit in registers
  if (decodeTargetBackend(options.targetBackend) == TargetBackend::LLVMIR)
    pm.addPass(createReorderOperationsPass());
  pm.addPass(createCopyRemovalPass());
  pm.addPass(createVectorBroadcastLoweringPass());
  pm.addPass(createCanonicalizeVectorForAIEVecPass(options));
  if (decodeTargetBackend(options.targetBackend) == TargetBackend::CPP)
    pm.addPass(createHoistCastOpToDataSourcePass());
}
