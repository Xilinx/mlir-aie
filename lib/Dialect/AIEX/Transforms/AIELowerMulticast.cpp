//===- AIELowerMulticast.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/AIENetlistAnalysis.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "aie-lower-multicast"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

template <typename MyOp>
struct AIEOpRemoval : public OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;

  AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(MyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIELowerMulticastPass : public AIEMulticastBase<AIELowerMulticastPass> {
  void runOnOperation() override {

    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    for (auto multicast : device.getOps<MulticastOp>()) {
      Region &r = multicast.getPorts();
      Block &b = r.front();
      Port sourcePort = multicast.port();
      TileOp srcTile = dyn_cast<TileOp>(multicast.getTile().getDefiningOp());
      for (Operation &Op : b.getOperations()) {
        if (MultiDestOp multiDest = dyn_cast<MultiDestOp>(Op)) {
          TileOp destTile =
              dyn_cast<TileOp>(multiDest.getTile().getDefiningOp());
          Port destPort = multiDest.port();
          builder.create<FlowOp>(builder.getUnknownLoc(), srcTile,
                                 sourcePort.first, sourcePort.second, destTile,
                                 destPort.first, destPort.second);
        }
      }
    }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<AIEOpRemoval<MulticastOp>>(device.getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIELowerMulticastPass() {
  return std::make_unique<AIELowerMulticastPass>();
}