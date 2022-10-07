//===- BufferOp.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Transform/AIE/Physical/BufferOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Support/LexicalCast.h"

#include "aie/AIEDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace xilinx;
using namespace xilinx::phy::physical;
using namespace xilinx::phy::transform::aie;

class BufferOpToAieLowering : public OpConversionPattern<BufferOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename BufferOp::Adaptor;

public:
  BufferOpToAieLowering(mlir::MLIRContext *context,
                        AIELoweringPatternSets *lowering)
      : OpConversionPattern<BufferOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(BufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (auto address = op.getOperation()->getAttrOfType<StringAttr>(
            "aie.external_address")) {
      rewriter.replaceOpWithNewOp<xilinx::AIE::ExternalBufferOp>(
          op, op.getBuffer().getType(), lexicalCast<long long>(address.str()));

    } else {
      auto tile = lowering->getTile(op);
      rewriter.replaceOpWithNewOp<xilinx::AIE::BufferOp>(
          op, op.getBuffer().getType(), tile);
    }

    return success();
  }
};

void BufferOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<BufferOpToAieLowering>(patterns.getContext(), lowering);
}
