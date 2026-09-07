//===- AIENormalizeDmaBdDims.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIENORMALIZEDMABDDIMS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-normalize-dma-bd-dims"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIENormalizeDmaBdDimsPass
    : xilinx::AIE::impl::AIENormalizeDmaBdDimsBase<AIENormalizeDmaBdDimsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();

    device.walk([&](DMABDOp op) {
      if (op.getMixedSizes().empty())
        return;

      for (OpFoldResult s : op.getMixedSizes())
        if (!getConstantIntValue(s))
          return;
      for (OpFoldResult s : op.getMixedStrides())
        if (!getConstantIntValue(s))
          return;

      if (op.getPadDimensions() && !op.getPadDimensions()->empty())
        return;

      std::optional<SmallVector<BDDimLayoutAttr>> maybeDims =
          op.getConstantDimensions();
      if (!maybeDims || maybeDims->empty())
        return;
      SmallVector<BDDimLayoutAttr> &origDims = *maybeDims;

      SmallVector<BDDimLayoutAttr> newDims;
      for (BDDimLayoutAttr dim : origDims) {
        if (dim.getSize() != 1)
          newDims.push_back(dim);
      }

      if (newDims.size() == origDims.size())
        return;

      if (newDims.empty() || isContiguousBDTransfer(newDims)) {
        int64_t product = 1;
        for (BDDimLayoutAttr dim : origDims)
          product *= dim.getSize();

        if (!op.hasLen() && !op.getBuffer().getType().hasStaticShape())
          return;
        std::optional<int32_t> lenVal = op.getConstantLen();
        if (lenVal.has_value() &&
            static_cast<int64_t>(*lenVal) != product)
          return;
        int32_t len = static_cast<int32_t>(product);

        op.getLenMutable().clear();
        op.setStaticLen(len);
        op.getSizesMutable().clear();
        op.getStridesMutable().clear();
        op.setStaticSizes(std::nullopt);
        op.setStaticStrides(std::nullopt);
        return;
      }

      SmallVector<int64_t> sizes, strides;
      sizes.reserve(newDims.size());
      strides.reserve(newDims.size());
      for (BDDimLayoutAttr dim : newDims) {
        sizes.push_back(static_cast<int64_t>(dim.getSize()));
        strides.push_back(static_cast<int64_t>(dim.getStride()));
      }

      op.getSizesMutable().clear();
      op.getStridesMutable().clear();
      op.setStaticSizes(sizes);
      op.setStaticStrides(strides);
    });
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIENormalizeDmaBdDimsPass() {
  return std::make_unique<AIENormalizeDmaBdDimsPass>();
}
