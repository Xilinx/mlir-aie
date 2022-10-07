//===- StreamOp.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Transform/AIE/Physical/StreamOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::phy::physical;
using namespace xilinx::phy::transform::aie;

class StreamOpToAieLowering : public OpConversionPattern<StreamOp> {
  using OpAdaptor = typename StreamOp::Adaptor;

public:
  StreamOpToAieLowering(mlir::MLIRContext *context,
                        AIELoweringPatternSets *lowering)
      : OpConversionPattern<StreamOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(StreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

void StreamOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<StreamOpToAieLowering>(patterns.getContext(), lowering);
}
