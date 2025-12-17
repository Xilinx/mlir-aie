//===- AIEHoistVectorTransferPointers.cpp -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass hoists vector transfer operations with IV-dependent pointers
// out of scf.for loops by using iter_args to track pointer updates. This
// optimization reduces address computation overhead in loops by maintaining
// a running pointer offset rather than recomputing addresses each iteration.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "aie-hoist-vector-transfer-pointers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if a value depends on the given loop induction variable
/// Uses a cache to avoid exponential recursion on complex dependency chains
static bool dependsOnLoopIVForHoist(Value val, Value loopIV,
                                    DenseMap<Value, bool> &cache) {
  // Check cache - return cached result if already computed
  auto it = cache.find(val);
  if (it != cache.end())
    return it->second;

  // Mark as being computed (assume false initially to handle recursion)
  // This prevents infinite recursion in case of cycles (though SSA shouldn't
  // have cycles)
  cache[val] = false;

  bool result = false;
  if (val == loopIV) {
    result = true;
  } else if (auto defOp = val.getDefiningOp()) {
    // Check for operations that use the loop IV in their operands
    for (Value operand : defOp->getOperands()) {
      if (dependsOnLoopIVForHoist(operand, loopIV, cache)) {
        result = true;
        break;
      }
    }
  }

  // Store the computed result in cache
  cache[val] = result;
  return result;
}

/// Wrapper for dependsOnLoopIVForHoist that manages the cache
static bool dependsOnLoopIVForHoist(Value val, Value loopIV) {
  DenseMap<Value, bool> cache;
  return dependsOnLoopIVForHoist(val, loopIV, cache);
}

/// Clone an operation and its operands (recursively) that don't depend on the
/// loop IV. Uses memoization via the mapping to avoid exponential recursion.
static Value cloneOpAndOperands(Operation *op, Value loopIV, OpBuilder &builder,
                                IRMapping &mapping) {
  // Only handle operations with exactly one result
  if (op->getNumResults() != 1)
    return Value();

  // If we've already cloned this operation, return the mapped result
  // This is critical for avoiding exponential recursion
  if (mapping.contains(op->getResult(0)))
    return mapping.lookup(op->getResult(0));

  // Check if this operation depends on the loop IV before trying to clone
  if (dependsOnLoopIVForHoist(op->getResult(0), loopIV))
    return Value();

  // Clone operands recursively
  SmallVector<Value> newOperands;
  for (Value operand : op->getOperands()) {
    if (auto defOp = operand.getDefiningOp()) {
      Value clonedOperand = cloneOpAndOperands(defOp, loopIV, builder, mapping);
      if (!clonedOperand)
        return Value(); // Failed to clone an operand
      newOperands.push_back(clonedOperand);
    } else {
      // Operand is a block argument or constant (guaranteed not to be the
      // loop IV due to the dependency check at line 91)
      newOperands.push_back(operand);
    }
  }

  // Clone the operation
  Operation *clonedOp = builder.clone(*op);
  clonedOp->setOperands(newOperands);

  // Map the result to enable memoization
  mapping.map(op->getResult(0), clonedOp->getResult(0));
  return clonedOp->getResult(0);
}

/// Get the total number of elements in a vector type
static int64_t getVectorNumElements(VectorType vectorType) {
  int64_t numElements = 1;
  for (int64_t dim : vectorType.getShape()) {
    numElements *= dim;
  }
  return numElements;
}

//===----------------------------------------------------------------------===//
// HoistVectorTransferPointers Pattern
//===----------------------------------------------------------------------===//

/// Information about a vector transfer operation
struct TransferOpInfo {
  Operation *op;
  Value base;
  MemRefType memrefType;
  VectorType vectorType;
  SmallVector<Value> indices;
  int64_t constantStride; // Total constant stride per iteration
  bool hasIVDependentIndices;
};

/// Pattern to hoist vector transfer operations with IV-dependent pointers
/// out of scf.for loops by using iter_args to track pointer updates
struct HoistVectorTransferPointersPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Value loopIV = forOp.getInductionVar();
    Location loc = forOp.getLoc();

    // Collect all vector transfer operations with IV-dependent indices
    SmallVector<TransferOpInfo> transferOps;

    for (Operation &op : forOp.getBody()->without_terminator()) {
      Value base;
      VectorType vectorType;
      SmallVector<Value> indices;

      if (auto readOp = dyn_cast<vector::TransferReadOp>(&op)) {
        base = readOp.getBase();
        vectorType = readOp.getVectorType();
        indices.assign(readOp.getIndices().begin(), readOp.getIndices().end());
      } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(&op)) {
        base = writeOp.getBase();
        vectorType = writeOp.getVectorType();
        indices.assign(writeOp.getIndices().begin(),
                       writeOp.getIndices().end());
      } else {
        continue;
      }

      auto memrefType = dyn_cast<MemRefType>(base.getType());
      if (!memrefType)
        continue;

      // Check if any indices depend on loop IV and compute constant stride
      bool hasIVDependentIndices = false;
      int64_t constantStride = 0;

      // Get the loop step to account for in stride calculation
      auto stepCst = forOp.getConstantStep();
      int64_t loopStep =
          stepCst.has_value() ? stepCst.value().getSExtValue() : 1;

      for (size_t dimIdx = 0; dimIdx < indices.size(); ++dimIdx) {
        Value idx = indices[dimIdx];
        if (dependsOnLoopIVForHoist(idx, loopIV)) {
          hasIVDependentIndices = true;

          // Calculate the stride for this dimension
          int64_t dimStride = 1;
          bool hasDynamicStride = false;
          for (size_t j = dimIdx + 1;
               j < static_cast<size_t>(memrefType.getRank()); ++j) {
            int64_t dimSize = memrefType.getShape()[j];
            if (dimSize == ShapedType::kDynamic) {
              hasDynamicStride = true;
              break;
            }
            dimStride *= dimSize;
          }

          // Multiply by loop step - the stride per iteration is:
          // (elements per dimension) * (loop step)
          if (!hasDynamicStride)
            constantStride += dimStride * loopStep;
          else
            hasIVDependentIndices = false; // Can't hoist if stride is dynamic
        }
      }

      transferOps.push_back({&op, base, memrefType, vectorType, indices,
                             constantStride, hasIVDependentIndices});
    }

    // If there are no transfer ops, don't modify
    if (transferOps.empty())
      return failure();

    // Prepare to add iter_args for each transfer operation with IV-dependent
    // indices
    SmallVector<Value> newInitArgs;
    SmallVector<Value> flatMemrefs;

    for (const auto &info : transferOps) {
      if (!info.hasIVDependentIndices)
        continue;

      // Flatten the memref if needed
      rewriter.setInsertionPoint(forOp);
      Value flatMemref = info.base;
      if (info.memrefType.getRank() > 1) {
        int64_t totalSize = 1;
        for (int64_t dim : info.memrefType.getShape()) {
          if (dim == ShapedType::kDynamic)
            return failure(); // Dynamic memref shapes not supported
          totalSize *= dim;
        }

        // Preserve strided layout if present
        MemRefType flatMemrefType;
        if (auto stridedLayout = dyn_cast_or_null<StridedLayoutAttr>(
                info.memrefType.getLayout())) {
          // The collapsed stride is the innermost stride (last element)
          int64_t collapsedStride = stridedLayout.getStrides().back();
          int64_t offset = stridedLayout.getOffset();

          auto newLayout = StridedLayoutAttr::get(rewriter.getContext(), offset,
                                                  {collapsedStride});
          flatMemrefType =
              MemRefType::get({totalSize}, info.memrefType.getElementType(),
                              newLayout, info.memrefType.getMemorySpace());
        } else {
          flatMemrefType =
              MemRefType::get({totalSize}, info.memrefType.getElementType(),
                              AffineMap(), info.memrefType.getMemorySpace());
        }

        SmallVector<ReassociationIndices> reassociation;
        ReassociationIndices allDims;
        for (size_t i = 0; i < static_cast<size_t>(info.memrefType.getRank());
             ++i) {
          allDims.push_back(i);
        }
        reassociation.push_back(allDims);

        flatMemref = memref::CollapseShapeOp::create(
            rewriter, loc, flatMemrefType, info.base, reassociation);
      }
      flatMemrefs.push_back(flatMemref);

      // Compute base pointer (with zeros for IV-dependent parts)
      int64_t rank = info.memrefType.getRank();
      AffineExpr linearExpr = rewriter.getAffineConstantExpr(0);
      int64_t stride = 1;
      for (int64_t i = rank - 1; i >= 0; --i) {
        linearExpr = linearExpr + rewriter.getAffineDimExpr(i) * stride;
        if (i > 0)
          stride *= info.memrefType.getShape()[i];
      }
      auto linearMap = AffineMap::get(rank, 0, linearExpr);

      // For IV-dependent indices, evaluate them at the loop's lower bound
      // to preserve constant offsets (e.g., %iv+1 becomes lowerBound+1)
      SmallVector<Value> evaluatedIndices;
      IRMapping indexMapping;
      for (Value idx : info.indices) {
        if (dependsOnLoopIVForHoist(idx, loopIV)) {
          // Clone the computation with the IV replaced by lower bound
          if (auto affineOp = idx.getDefiningOp<affine::AffineApplyOp>()) {
            SmallVector<Value> mappedOperands;
            for (Value operand : affineOp.getMapOperands()) {
              if (operand == loopIV)
                mappedOperands.push_back(forOp.getLowerBound());
              else
                mappedOperands.push_back(operand);
            }
            Value evaluatedIdx = affine::AffineApplyOp::create(
                rewriter, loc, affineOp.getAffineMap(), mappedOperands);
            evaluatedIndices.push_back(evaluatedIdx);
          } else {
            // Direct IV usage - just use lower bound
            evaluatedIndices.push_back(forOp.getLowerBound());
          }
        } else {
          // Index doesn't depend on IV, clone it
          if (auto defOp = idx.getDefiningOp()) {
            Value clonedIdx =
                cloneOpAndOperands(defOp, loopIV, rewriter, indexMapping);
            if (clonedIdx)
              evaluatedIndices.push_back(clonedIdx);
            else
              evaluatedIndices.push_back(idx);
          } else {
            evaluatedIndices.push_back(idx);
          }
        }
      }

      Value basePointer = affine::AffineApplyOp::create(
          rewriter, loc, linearMap, evaluatedIndices);

      newInitArgs.push_back(basePointer);
    }

    // If there are no IV-dependent transfers, just process them to flatten
    // vectors
    if (newInitArgs.empty()) {
      // Check if any transfer needs flattening (avoid infinite rewrites)
      bool needsFlattening = false;
      bool hasProcessableTransfers = false;
      for (const auto &info : transferOps) {
        // Skip if base is defined inside the loop (e.g., a subview)
        // We can't hoist these
        if (info.base.getDefiningOp() &&
            forOp->isProperAncestor(info.base.getDefiningOp()))
          continue;

        hasProcessableTransfers = true;

        // Check if this transfer has already been flattened
        // (flattened transfers use 1D identity map)
        if (auto readOp = dyn_cast<vector::TransferReadOp>(info.op)) {
          if (readOp.getPermutationMap().getNumDims() != 1)
            needsFlattening = true;
        } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(info.op)) {
          if (writeOp.getPermutationMap().getNumDims() != 1)
            needsFlattening = true;
        }
      }

      // If there are no processable transfers (all bases defined in loop)
      // or nothing needs flattening, bail out
      if (!hasProcessableTransfers || !needsFlattening)
        return failure();

      // First, create flattened memrefs outside the loop for bases not defined
      // inside
      DenseMap<Value, Value> baseFlatMemrefs;
      rewriter.setInsertionPoint(forOp);
      for (const auto &info : transferOps) {
        if (baseFlatMemrefs.count(info.base))
          continue;

        // Skip if base is defined inside the loop (e.g., a subview)
        if (info.base.getDefiningOp() &&
            forOp->isProperAncestor(info.base.getDefiningOp()))
          continue;

        Value flatMemref = info.base;
        if (info.memrefType.getRank() > 1) {
          int64_t totalSize = 1;
          for (int64_t dim : info.memrefType.getShape()) {
            totalSize *= dim;
          }

          // Preserve strided layout if present
          MemRefType flatMemrefType;
          if (auto stridedLayout = dyn_cast_or_null<StridedLayoutAttr>(
                  info.memrefType.getLayout())) {
            int64_t collapsedStride = stridedLayout.getStrides().back();
            int64_t offset = stridedLayout.getOffset();

            auto newLayout = StridedLayoutAttr::get(rewriter.getContext(),
                                                    offset, {collapsedStride});
            flatMemrefType =
                MemRefType::get({totalSize}, info.memrefType.getElementType(),
                                newLayout, info.memrefType.getMemorySpace());
          } else {
            flatMemrefType =
                MemRefType::get({totalSize}, info.memrefType.getElementType(),
                                AffineMap(), info.memrefType.getMemorySpace());
          }

          SmallVector<ReassociationIndices> reassociation;
          ReassociationIndices allDims;
          for (size_t i = 0; i < static_cast<size_t>(info.memrefType.getRank());
               ++i) {
            allDims.push_back(i);
          }
          reassociation.push_back(allDims);
          flatMemref = memref::CollapseShapeOp::create(
              rewriter, loc, flatMemrefType, info.base, reassociation);
        }
        baseFlatMemrefs[info.base] = flatMemref;
      }

      // Process all transfers without using iter_args
      bool madeChanges = false;
      for (const auto &info : transferOps) {
        // Skip if base is defined inside the loop
        if (info.base.getDefiningOp() &&
            forOp->isProperAncestor(info.base.getDefiningOp()))
          continue;

        // Skip if we don't have a flattened version
        if (!baseFlatMemrefs.count(info.base))
          continue;

        rewriter.setInsertionPoint(info.op);

        // Flatten vector type
        int64_t numElements = getVectorNumElements(info.vectorType);
        VectorType flatVectorType =
            VectorType::get({numElements}, info.vectorType.getElementType());

        // Get the flattened memref
        Value flatMemref = baseFlatMemrefs[info.base];

        // Compute pointer from indices
        int64_t rank = info.memrefType.getRank();
        AffineExpr linearExpr = rewriter.getAffineConstantExpr(0);
        int64_t stride = 1;
        for (int64_t i = rank - 1; i >= 0; --i) {
          linearExpr = linearExpr + rewriter.getAffineDimExpr(i) * stride;
          if (i > 0)
            stride *= info.memrefType.getShape()[i];
        }
        auto linearMap = AffineMap::get(rank, 0, linearExpr);

        Value currentPointer = affine::AffineApplyOp::create(
            rewriter, loc, linearMap, info.indices);

        // Transform the transfer operation
        AffineMap identityMap1D = AffineMap::get(
            1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());
        auto inBoundsAttr = rewriter.getBoolArrayAttr({true});

        if (auto readOp = dyn_cast<vector::TransferReadOp>(info.op)) {
          Value flatRead = vector::TransferReadOp::create(
              rewriter, loc, flatVectorType, flatMemref,
              ValueRange{currentPointer}, AffineMapAttr::get(identityMap1D),
              readOp.getPadding(),
              /*mask=*/Value(), inBoundsAttr);
          Value shapedRead = vector::ShapeCastOp::create(
              rewriter, loc, info.vectorType, flatRead);
          rewriter.replaceOp(readOp, shapedRead);
          madeChanges = true;
        } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(info.op)) {
          Value flatValue = vector::ShapeCastOp::create(
              rewriter, loc, flatVectorType, writeOp.getVector());
          rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
              writeOp, flatValue, flatMemref, ValueRange{currentPointer},
              AffineMapAttr::get(identityMap1D), /*mask=*/Value(),
              inBoundsAttr);
          madeChanges = true;
        }
      }
      return madeChanges ? success() : failure();
    }

    // Use replaceWithAdditionalYields to add pointer iter_args
    auto yieldValuesFn =
        [&](OpBuilder &b, Location yieldLoc,
            ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
      SmallVector<Value> yieldValues;

      // Process each transfer operation with IV-dependent indices
      size_t iterArgIdx = 0;
      for (size_t i = 0; i < transferOps.size(); ++i) {
        const auto &info = transferOps[i];
        if (!info.hasIVDependentIndices)
          continue;

        BlockArgument ptrIterArg =
            newBbArgs[newBbArgs.size() - newInitArgs.size() + iterArgIdx];
        Value flatMemref = flatMemrefs[iterArgIdx];

        // Flatten vector type
        int64_t numElements = getVectorNumElements(info.vectorType);
        VectorType flatVectorType =
            VectorType::get({numElements}, info.vectorType.getElementType());

        // Transform the transfer operation to use the iter_arg pointer
        b.setInsertionPoint(info.op);

        AffineMap identityMap1D =
            AffineMap::get(1, 0, b.getAffineDimExpr(0), b.getContext());
        auto inBoundsAttr = b.getBoolArrayAttr({true});

        if (auto readOp = dyn_cast<vector::TransferReadOp>(info.op)) {
          Value flatRead = vector::TransferReadOp::create(
              b, loc, flatVectorType, flatMemref, ValueRange{ptrIterArg},
              AffineMapAttr::get(identityMap1D), readOp.getPadding(),
              /*mask=*/Value(), inBoundsAttr);
          Value shapedRead =
              vector::ShapeCastOp::create(b, loc, info.vectorType, flatRead);
          rewriter.replaceOp(readOp, shapedRead);
        } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(info.op)) {
          Value flatValue = vector::ShapeCastOp::create(b, loc, flatVectorType,
                                                        writeOp.getVector());
          rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
              writeOp, flatValue, flatMemref, ValueRange{ptrIterArg},
              AffineMapAttr::get(identityMap1D), /*mask=*/Value(),
              inBoundsAttr);
        }

        // Compute next pointer value: current_ptr + constant_stride
        Value strideConst =
            arith::ConstantIndexOp::create(b, yieldLoc, info.constantStride);
        Value nextPtr =
            arith::AddIOp::create(b, yieldLoc, ptrIterArg, strideConst);
        yieldValues.push_back(nextPtr);

        iterArgIdx++;
      }

      return yieldValues;
    };

    // Create new loop with additional iter_args for pointers
    FailureOr<LoopLikeOpInterface> newLoopResult =
        cast<LoopLikeOpInterface>(forOp.getOperation())
            .replaceWithAdditionalYields(
                rewriter, newInitArgs, // new init operands (base pointers)
                true,                  // replace uses in loop
                yieldValuesFn);

    if (failed(newLoopResult))
      return failure();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIEHoistVectorTransferPointersPass
//===----------------------------------------------------------------------===//

struct AIEHoistVectorTransferPointersPass
    : AIEHoistVectorTransferPointersBase<AIEHoistVectorTransferPointersPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<HoistVectorTransferPointersPattern>(context);

    // Apply patterns to the entire module - the pattern will only match scf.for
    // ops within aie.core regions
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
AIE::createAIEHoistVectorTransferPointersPass() {
  return std::make_unique<AIEHoistVectorTransferPointersPass>();
}
