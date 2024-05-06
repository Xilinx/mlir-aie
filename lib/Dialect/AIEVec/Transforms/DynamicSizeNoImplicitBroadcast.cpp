//===- DynamicSizeNoImplicitBroadcast.cpp -=====================-*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// This file contains rewrites to the arith dialect to enable the support of
// dynamic sized tensor/memref for the auto-vectorization to CPP flow.
// MLIR-AIE auto-vectorization to CPP flow currently doesn't support to
// implicitly broadcast a dynamic dimension of size `1`. Hence, we assume that
// dynamic dimensions are not with size '1' that can be interpreted to various
// broadcasting scenarios. The effectiveness of this rewrite pattern is guarded
// by the attribute `tosa.no_implicit_broadcast_of_dynamic_sizes`.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "dynamic-size-no-implicit-broadcast"

using namespace llvm;
using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

//============================================================================//
//=========================== Rewrite Patterns ===============================//
//============================================================================//

// This pattern replaces a arith::CmpIOp with a arith::ConstantOp `false` only
// when the CmpIOp compares the equality of a dynamic dimension's runtime size
// to a constant 1, and is guarded by the attribute
// `tosa.no_implicit_broadcast_of_dynamic_sizes`.
struct DynamicSizeNoImplicitBroadcastPattern : RewritePattern {
  DynamicSizeNoImplicitBroadcastPattern(MLIRContext *context)
      : RewritePattern(arith::CmpIOp::getOperationName(), /*benefit=*/1,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isAssumingNoImplicitBroadcastOfDynamicSizes(rewriter))
      return failure();

    arith::CmpIOp cmpiOp = cast<arith::CmpIOp>(op);

    if (cmpiOp.getPredicate() != arith::CmpIPredicate::eq)
      return failure();

    auto lhsOp = cmpiOp.getLhs().getDefiningOp();
    auto rhsOp = cmpiOp.getRhs().getDefiningOp();
    if (!((isa<memref::DimOp>(lhsOp) || isa<tensor::DimOp>(lhsOp)) &&
          isa<arith::ConstantOp>(rhsOp)) &&
        !((isa<memref::DimOp>(rhsOp) || isa<tensor::DimOp>(rhsOp)) &&
          isa<arith::ConstantOp>(lhsOp)))
      return failure();

    // Make sure rhsOp is ConstantOp and lhsOp is DimOp
    if (isa<memref::DimOp>(rhsOp) || isa<tensor::DimOp>(rhsOp))
      std::swap(lhsOp, rhsOp);

    // If ConstantOp is 1 for Integer/Index, replace cmpiOp as constant 0
    auto constantOp = cast<arith::ConstantOp>(rhsOp);
    if (cast<IntegerAttr>(constantOp.getValue()).getValue().getZExtValue() != 1)
      return failure();

    // Check the DimOp's input is a dynamic dim from the given index
    auto constIndexOp = lhsOp->getOperand(1).getDefiningOp<arith::ConstantOp>();
    if (!constIndexOp)
      return failure();

    auto index =
        cast<IntegerAttr>(constIndexOp.getValue()).getValue().getZExtValue();
    auto inputDimType = dyn_cast<ShapedType>(lhsOp->getOperand(0).getType());
    if (!inputDimType || !inputDimType.isDynamicDim(index))
      return failure();

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        cmpiOp, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));

    return success();
  }
};

//============================================================================//
//======================== Canonicalization Passes ===========================//
//============================================================================//

struct DynamicSizeNoImplicitBroadcastPass
    : PassWrapper<DynamicSizeNoImplicitBroadcastPass, OperationPass<>> {

  StringRef getArgument() const final {
    return "test-dynamic-size-no-implicit-broadcast";
  }

  StringRef getDescription() const final {
    return "Test rewriting arith operations when assuming no implict "
           "broadcast of dynamic sizes";
  }

  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<DynamicSizeNoImplicitBroadcastPattern>(patterns.getContext());

    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

std::unique_ptr<::mlir::Pass>
xilinx::aievec::createDynamicSizeNoImplicitBroadcastPass() {
  return std::make_unique<DynamicSizeNoImplicitBroadcastPass>();
}

//============================================================================//
//====================== Main Pipeline Configuration =========================//
//============================================================================//

void xilinx::aievec::buildDynamicSizeNoImplicitBroadcastPass(
    OpPassManager &pm) {
  pm.addPass(createDynamicSizeNoImplicitBroadcastPass());
}
