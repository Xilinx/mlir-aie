//===- AIEVecTransformOps.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (c) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/TransformOps/AIEVecTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define DEBUG_TYPE "aievec-transforms"

//===----------------------------------------------------------------------===//
// VectorizeContractionOp
//===----------------------------------------------------------------------===//

// Emit IR to convert the given tensor in the form tensor<...xMxNxTy> into a
// tensor<...xvector<MxNxTy>>. It does so by bufferizing, casting, and
// tensorizing.
//
// E.g., for a `%t : tensor<64x64x8x4xf32>`, it will generate the following IR:
// ```
//     %0 = bufferization.to_memref %t : memref<64x64x8x4xf32>
//     %1 = vector.type_cast %0 : memref<64x64xvector<8x4xf32>>
//     %2 = bufferization.to_tensor %1 restrict : memref<64x64xvector<8x4xf32>>
// ```
static Value vectorizeTensor(OpBuilder &rewriter, Location loc, Value tensor) {
  auto opTy = tensor.getType();
  auto shapeTy = cast<ShapedType>(opTy);
  auto shape = shapeTy.getShape();
  auto elemTy = shapeTy.getElementType();
  auto toMemRefOp = rewriter.create<bufferization::ToMemrefOp>(
      loc, MemRefType::get(shape, elemTy), tensor);
  auto rank = shape.size();
  auto newShape = shape.slice(0, rank - 2);
  auto opVecElemTy = VectorType::get(shape.slice(rank - 2, 2), elemTy);
  auto opMemrefVecTy = MemRefType::get(newShape, opVecElemTy);
  auto typeCastOp =
      rewriter.create<vector::TypeCastOp>(loc, opMemrefVecTy, toMemRefOp);
  auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get(newShape, opVecElemTy), typeCastOp);
  toTensorOp.setRestrict(true);
  return toTensorOp.getResult();
}

// Emit IR to convert the given tensor in the form tensor<...xvector<MxNxTy>>
// into a tensor<...xMxNxTy>. It performs the inverse operation to
// `vectorizeTensor` above.
static Value scalarizeTensor(OpBuilder &rewriter, Location loc, Value tensor) {
  auto opTy = tensor.getType();
  auto shapeTy = cast<ShapedType>(opTy);

  auto vecShape = shapeTy.getShape();
  auto vecElemTy = cast<VectorType>(shapeTy.getElementType());
  auto elemTy = vecElemTy.getElementType();
  auto toMemRefVecTyOp = rewriter.create<bufferization::ToMemrefOp>(
      loc, MemRefType::get(vecShape, vecElemTy), tensor);

  SmallVector<int64_t> scalShape;
  for (auto d : shapeTy.getShape())
    scalShape.push_back(d);
  for (auto d : vecElemTy.getShape())
    scalShape.push_back(d);
  auto opMemrefScalTy = MemRefType::get(scalShape, elemTy);
  auto typeCastOp =
      rewriter.create<vector::TypeCastOp>(loc, opMemrefScalTy, toMemRefVecTyOp);

  auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get(scalShape, elemTy), typeCastOp);
  toTensorOp.setRestrict(true);
  return toTensorOp.getResult();
}

static bool vectorizeContractionOpBlock(OpBuilder &rewriter, Location loc,
                                        Block &srcBlock, Block &dstBlock) {
  auto ctx = rewriter.getContext();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(&dstBlock);
  auto baA = static_cast<Value>(dstBlock.getArgument(0));
  auto baB = static_cast<Value>(dstBlock.getArgument(1));
  auto baC = static_cast<Value>(dstBlock.getArgument(2));
  // Store vectorized values for op replacement
  llvm::DenseMap<Value, Value> convertedValues;
  convertedValues.try_emplace(srcBlock.getArgument(0), baA);
  convertedValues.try_emplace(srcBlock.getArgument(1), baB);
  convertedValues.try_emplace(srcBlock.getArgument(2), baC);
  auto indexingMaps = rewriter.getAffineMapArrayAttr(
      {AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0, 2}, ctx)
           .dropResults(0),
       AffineMap::getPermutationMap(ArrayRef<unsigned>{0, 2, 1}, ctx)
           .dropResults(0),
       AffineMap::getPermutationMap(ArrayRef<unsigned>{2, 0, 1}, ctx)
           .dropResults(0)});
  auto iteratorTypes = rewriter.getArrayAttr(
      {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
       vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
       vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});
  bool addOpFound = false, mulOpFound = false;
  WalkResult walkResult = srcBlock.walk([&](Operation *op) {
    return llvm::TypeSwitch<Operation *, WalkResult>(op)
        .Case<arith::AddIOp, arith::AddFOp>([&](auto addOp) {
          if (addOpFound)
            return WalkResult::interrupt();
          addOpFound = true;
          auto lhs = addOp->getOperand(0);
          auto rhs = addOp->getOperand(1);
          Value opA, opB, opC;
          auto lhsDefOp = lhs.getDefiningOp();
          auto rhsDefOp = rhs.getDefiningOp();
          if (lhsDefOp && isa<arith::MulIOp, arith::MulFOp>(lhsDefOp)) {
            opA = convertedValues[lhsDefOp->getOperand(0)];
            opB = convertedValues[lhsDefOp->getOperand(1)];
            opC = convertedValues[rhs];
          } else if (rhsDefOp && isa<arith::MulIOp, arith::MulFOp>(rhsDefOp)) {
            opA = convertedValues[rhsDefOp->getOperand(0)];
            opB = convertedValues[rhsDefOp->getOperand(1)];
            opC = convertedValues[lhs];
          } else
            return WalkResult::interrupt();
          auto conOp = rewriter.create<vector::ContractionOp>(
              loc, opA, opB, opC, indexingMaps, iteratorTypes);
          convertedValues.try_emplace(op->getResult(0), conOp.getResult());
          return WalkResult::advance();
        })
        .Case<arith::MulIOp, arith::MulFOp>([&](auto) {
          if (mulOpFound)
            return WalkResult::interrupt();
          mulOpFound = true;
          return WalkResult::skip();
        })
        .Case<linalg::YieldOp>([&](linalg::YieldOp yieldOp) {
          rewriter.create<linalg::YieldOp>(
              loc, convertedValues[yieldOp.getValues()[0]]);
          return WalkResult::advance(); // Or ::interrupt()
        })
        .Default([&](Operation *unaryOp) {
          if (unaryOp->getNumResults() != 1 || unaryOp->getNumOperands() != 1)
            return WalkResult::interrupt();
          auto srcOpIn = unaryOp->getOperand(0);
          auto srcOpInTy = srcOpIn.getType();
          auto srcOpTy = unaryOp->getResultTypes()[0];
          auto dstOpIn = convertedValues[srcOpIn];
          Type dstOpTy = dstOpIn.getType();
          if (srcOpInTy != srcOpTy) {
            auto vecElemTy = dyn_cast<VectorType>(dstOpTy);
            if (!vecElemTy)
              return WalkResult::interrupt();
            dstOpTy = VectorType::get(vecElemTy.getShape(), srcOpTy);
          }
          auto newOp =
              rewriter.create(loc, unaryOp->getName().getIdentifier(),
                              {dstOpIn}, {dstOpTy}, unaryOp->getAttrs());
          convertedValues.try_emplace(unaryOp->getResult(0),
                                      newOp->getResult(0));
          return WalkResult::advance();
        });
  });
  return mulOpFound && addOpFound && !walkResult.wasInterrupted();
}

DiagnosedSilenceableFailure transform::VectorizeContractionOp::applyToOne(
    TransformRewriter &rewriter, linalg::GenericOp target,
    ApplyToEachResultList &results, TransformState &state) {

  auto ctx = target.getContext();
  SmallVector<Value> inputs = target.getInputs();
  if (SmallVector<Value> outputs = target.getOutputs();
      inputs.size() != 2 || outputs.size() != 1)
    return emitSilenceableError() << "payload is not a contraction.";

  // Split the iterators in two: inner contraction + remaining
  SmallVector<utils::IteratorType> iterators = target.getIteratorTypesArray();
  auto innerMostIterators =
      SmallVector<utils::IteratorType>(iterators.end() - 3, iterators.end());
  auto outerMostIterators =
      SmallVector<utils::IteratorType>(iterators.begin(), iterators.end() - 3);

  if (!linalg::isParallelIterator(innerMostIterators[0]) ||
      !linalg::isParallelIterator(innerMostIterators[1]) ||
      !linalg::isReductionIterator(innerMostIterators[2]))
    return emitSilenceableError()
           << "linalg.generic op innermost iterators don't correspond with a "
              "gemm-like contraction.";

  auto indexingMaps = target.getIndexingMapsArray();
  //===
  // Verify that the innermost dimensions are a contraction
  //===

  // 1. Build the indexing maps for the operands of a GEMM contraction
  auto mmAidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0, 2}, ctx)
          .dropResults(0);
  auto mmBidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{0, 2, 1}, ctx)
          .dropResults(0);
  auto mmCidxMap =
      AffineMap::getPermutationMap(ArrayRef<unsigned>{2, 0, 1}, ctx)
          .dropResults(0);

  // 2. Get the indexing maps for the 2 innermost dimmensions of each operand
  SmallVector<int64_t> outerMostResults;
  for (int64_t i = 0; i < indexingMaps[0].getNumResults() - 2; i++)
    outerMostResults.push_back(i);

  auto innerMostA = indexingMaps[0].dropResults(outerMostResults);
  auto innerMostB = indexingMaps[1].dropResults(outerMostResults);
  auto innerMostC = indexingMaps[2].dropResults(outerMostResults);

  // 3. Compare the extended GEMM contraction indexing maps with the indexing
  //    maps of the innermost results.
  int64_t numOuterMostDims = indexingMaps[0].getNumDims() - 3;
  if (innerMostA != mmAidxMap.shiftDims(numOuterMostDims) ||
      innerMostB != mmBidxMap.shiftDims(numOuterMostDims) ||
      innerMostC != mmCidxMap.shiftDims(numOuterMostDims))
    return emitSilenceableError()
           << "linalg.generic op innermost indexing maps don't correspond with "
              "a gemm-like contraction.";

  //===
  // Create new indexing maps for the vectorized operation
  //===

  SmallVector<AffineExpr> remOuterDims;
  for (unsigned i = 0; i < numOuterMostDims; i++)
    remOuterDims.push_back(getAffineDimExpr(i, ctx));
  unsigned numResults = indexingMaps[0].getNumResults();
  SmallVector<int64_t> positions = {numResults - 2, numResults - 1};
  auto outerMostAidxMap =
      indexingMaps[0].dropResults(positions).replaceDimsAndSymbols(
          remOuterDims, {}, numOuterMostDims, 0);
  auto outerMostBidxMap =
      indexingMaps[1].dropResults(positions).replaceDimsAndSymbols(
          remOuterDims, {}, numOuterMostDims, 0);
  auto outerMostCidxMap =
      indexingMaps[2].dropResults(positions).replaceDimsAndSymbols(
          remOuterDims, {}, numOuterMostDims, 0);

  rewriter.setInsertionPoint(target);
  Location loc = target.getLoc();
  // Insert reshape ops for input operands
  auto opA = vectorizeTensor(rewriter, loc, target.getInputs()[0]);
  auto opB = vectorizeTensor(rewriter, loc, target.getInputs()[1]);
  auto opC = vectorizeTensor(rewriter, loc, target.getOutputs()[0]);

  // Create new linalg.generic with vector arguments and vectorized basic block
  auto newOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange({opC.getType()}), ValueRange({opA, opB}),
      ValueRange({opC}),
      SmallVector<AffineMap>(
          {outerMostAidxMap, outerMostBidxMap, outerMostCidxMap}),
      outerMostIterators);
  auto &opBody = newOp->getRegion(0);
  opBody.push_back(new Block());
  auto &opBlock = opBody.front();
  opBlock.addArguments({cast<TensorType>(opA.getType()).getElementType(),
                        cast<TensorType>(opB.getType()).getElementType(),
                        cast<TensorType>(opC.getType()).getElementType()},
                       {loc, loc, loc});
  if (!vectorizeContractionOpBlock(rewriter, loc, target->getRegion(0).front(),
                                   opBlock))
    return emitSilenceableError()
           << "linalg.generic op payload does not correspond with a "
              "vectorizable contraction.";

  // Insert reshape ops for output operand
  auto res = scalarizeTensor(rewriter, loc, newOp.getResults()[0]);
  rewriter.replaceOp(target, res);

  results.push_back(newOp);

  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/TransformOps/AIEVecTransformOps.cpp.inc"
