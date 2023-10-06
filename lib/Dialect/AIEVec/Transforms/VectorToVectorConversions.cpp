//===-VectorToVectorConversions.cpp - Conversions within Vector -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "VectorToVectorConversions.h"

#define DEBUG_TYPE "aievec-canonicalization"

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

//============================================================================//
//================== Common AIE canonicalization analysis ====================//
//============================================================================//

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

  SplitUnalignedTransferReadPattern(MLIRContext *context, int64_t minVectorSize,
                                    int64_t maxVectorSize, int64_t alignment)
      : OpConversionPattern<vector::TransferReadOp>(context),
        minVectorSize(minVectorSize), maxVectorSize(maxVectorSize),
        vectorAlignment(alignment) {}

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
    Value newInnerMostIdx =
        rewriter
            .create<AffineApplyOp>(readOp.getLoc(), offsetCorrectionMap,
                                   SmallVector<Value, 1>({oldInnerMostIdx}))
            .getResult();
    SmallVector<Value, 8> alignedIdx;
    alignedIdx.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
    alignedIdx[alignedIdx.size() - 1] = newInnerMostIdx;

    // Create the aligned transfer read for a vector 2x as long that covers the
    // elements of the unaligned vector.
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        loc, longVecTy, adaptor.getSource(), alignedIdx, adaptor.getPadding());

    // Create a `vector.extract_strided_slice` to extract the unaligned vector.
    rewriter.replaceOpWithNewOp<vector::ExtractStridedSliceOp>(
        readOp, newReadOp.getResult(), offset, vLen, 1);

    return success();
  }

  int64_t minVectorSize;
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

    Value srcMemRef = adaptor.getSource();
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
      if (auto applyOp = newIdx.getDefiningOp<AffineApplyOp>())
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
      newIdx = rewriter
                   .create<AffineApplyOp>(readOp.getLoc(), newAddrMap,
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
    if (!defOp ||
        isa<vector::TransferReadOp, memref::LoadOp, AffineLoadOp, func::CallOp>(
            defOp))
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

    auto newOp =
        rewriter.create(extOp->getLoc(), defOp->getName().getIdentifier(),
                        inputs, {extOp.getOut().getType()}, defOp->getAttrs());
    rewriter.replaceOp(extOp, newOp->getResult(0));
    return success();
  }
};

//============================================================================//
//============ AIEML canonicalization conversion patterns ===============//
//============================================================================//

//============================================================================//
//================ Common AIE canonicalization configuration =================//
//============================================================================//
static void
configureCommonAIECanonicalizeLegalizations(ConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect, affine::AffineDialect,
                         memref::MemRefDialect, vector::VectorDialect>();
}

static void
populateCommonAIECanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertSplatTransferReadToBroadcastPattern>(
      patterns.getContext());
}

//============================================================================//
//============== AIEv1-specific canonicalization configuration ===============//
//============================================================================//

static void configureAIEv1CanonicalizeLegalizations(ConversionTarget &target) {
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant() &&
               getTransferReadAlignmentOffset(op, op.getVectorType(), 128)
                       .value_or(0) == 0;
      });
}

static void
populateAIEv1CanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<SplitUnalignedTransferReadPattern>(patterns.getContext(), 128,
                                                  512, 128);
}

//============================================================================//
//============== AIEML-specific canonicalization configuration ===============//
//============================================================================//

static void configureAIEMLCanonicalizeLegalizations(ConversionTarget &target) {
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant() &&
               getTransferReadAlignmentOffset(op, op.getVectorType(), 256)
                       .value_or(0) == 0;
      });
}

static void
populateAIEMLCanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<SplitUnalignedTransferReadPattern>(patterns.getContext(), 128,
                                                  1024, 256);
}

//============================================================================//
//=================== Common AIE Canonicalization Passes =====================//
//============================================================================//

// This pass converts standard vector ops into a subset of `Vector` ops more
// amenable to being converted to `AIEVec`. So far, this process consists of
// two steps:
//    1) Replace splat transfer reads with contiguous transfer reads followed
//       by `extract` + `broadcast` operations.
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
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    vector::VectorDialect, affine::AffineDialect>();
  }

  Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc("Select AIE version: \"aie\" or \"aieml\". This will "
                     "determine the vector size and available operations."),
      llvm::cl::init("aie")};

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    AIEArch aieVersion = AIEArch::AIE;
    if (!aieTarget.empty()) {
      std::string target = aieTarget;
      if (target == "aieml") {
        aieVersion = AIEArch::AIE_ML;
      } else if (target != "aie") {
        op->emitError() << "unknown AIE target '" << aieTarget << "'";
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
    auto op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<HoistCastOpToDataSourcePattern>(patterns.getContext());

    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

static std::unique_ptr<::mlir::Pass> createHoistCastOpToDataSourcePass() {
  return std::make_unique<HoistCastOpToDataSourcePass>();
}

//============================================================================//
//=============== Main Vector2Vector Pipeline Configuration ==================//
//============================================================================//

void xilinx::aievec::buildCanonicalizeVectorForAIEVec(
    OpPassManager &pm, const CanonicalizeVectorForAIEVecOptions &options) {
  // Add `Vector` code canonicalization passes
  // TODO: Add passes to unroll vector with unsupported types
  // TODO: Add passes to split vectors that won't fit in registers
  pm.addPass(createCopyRemovalPass());
  pm.addPass(createCanonicalizeVectorForAIEVecPass(options));
  pm.addPass(createHoistCastOpToDataSourcePass());
}
