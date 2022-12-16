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
#define GEN_PASS_DEF_CONVERTVECTORTOAIEVEC
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

// This pattern replaces `vector.transfer_read` with `aievec.upd`. Right now,
// it performs a naïve direct translation. This needs to be expanded to support
// more complex scenarios.
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
    // TODO: Verify that the access is contiguous and aligned.
    // TODO: Extend this function so it can address more than the trivial case.
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

// XXX: Notice that this template doesn't verify that the vector element type is
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
}

static void populateAIEVecV2ConversionPatterns(RewritePatternSet &patterns,
                                               AnalysisManager &am) {
  patterns.add<LowerVectorTransferReadToAIEUPD, SplitUPDOpOnAccPattern>(
      patterns.getContext(), am, 512);
}

//===----------------------------------------------------------------------===//
// Legalizations
//===----------------------------------------------------------------------===//

// TODO: Review the validity of these legalizations beyond basic cases.

static void configureAIEVecCommonLegalizations(ConversionTarget &target,
                                               AnalysisManager &am) {
  target.addLegalDialect<xilinx::aievec::AIEVecDialect>();
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
}

static void configureAIEVecV2Legalizations(ConversionTarget &target,
                                           AnalysisManager &am) {
  target.addDynamicallyLegalOp<aievec::UPDOp>([&am](aievec::UPDOp op) {
    return am.getChildAnalysis<UPDOpEffectiveAccessSizeAnalysis>(op)
               .effectiveSize <= 1024;
  });
}

//===----------------------------------------------------------------------===//
// Conversion pass
//===----------------------------------------------------------------------===//

// TODO: For more complex conversion from Vector to AIEVec, we may want to make
// TODO: this into a pipeline where:
// TODO:     1. If the operands of a vector op are too long, split it down to
// TODO:        right-sized vectors.
// TODO:     2. Unroll vector ops when the vector type is unsupported.
// TODO:     3. Perform the dialect conversion legalizations.
struct ConvertVectorToAIEVec
    : public aievec::impl::ConvertVectorToAIEVecBase<ConvertVectorToAIEVec> {
  using Base::Base;

  void runOnOperation() override;
};

std::unique_ptr<Pass> xilinx::aievec::createConvertVectorToAIEVecPass(
    const ConvertVectorToAIEVecOptions &options) {
  return std::make_unique<ConvertVectorToAIEVec>(options);
}

/// Convert incoming vector operations into their corresponding AIE vector
/// intrinsics.
void ConvertVectorToAIEVec::runOnOperation() {
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
