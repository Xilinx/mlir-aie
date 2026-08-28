//===- AIEResolveAddressPatchBuffers.cpp ------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIERESOLVEADDRESSPATCHBUFFERS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIEResolveAddressPatchBuffersPass
    : xilinx::AIEX::impl::AIEResolveAddressPatchBuffersBase<
          AIEResolveAddressPatchBuffersPass> {

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    SmallVector<NpuAddressPatchOp> patches;
    device.walk([&](NpuAddressPatchOp op) {
      if (op.getBuffer())
        patches.push_back(op);
    });

    for (NpuAddressPatchOp op : patches) {
      auto seq = op->getParentOfType<AIE::RuntimeSequenceOp>();
      if (!seq) {
        op.emitOpError("must be inside a runtime sequence to resolve its host "
                       "buffer");
        return signalPassFailure();
      }
      auto traced = traceSubviewToBlockArgument(op.getBuffer());
      if (!traced || traced->rootArg.getOwner() != &seq.getBody().front()) {
        op.emitOpError("buffer must be an argument of the enclosing runtime "
                       "sequence, or a static contiguous view of one");
        return signalPassFailure();
      }
      std::optional<unsigned> argIdx = getHostBufferArgIndex(traced->rootArg);
      if (!argIdx) {
        op.emitOpError("buffer must resolve to a memref argument");
        return signalPassFailure();
      }

      OpBuilder builder(op);
      Value argPlus = op.getArgPlus();
      if (traced->offsetInBytes != 0) {
        if (auto base = getConstantIntOperand(argPlus)) {
          argPlus = createConstantI32(builder, op.getLoc(),
                                      *base + traced->offsetInBytes);
        } else {
          Value shift =
              createConstantI32(builder, op.getLoc(), traced->offsetInBytes);
          argPlus = arith::AddIOp::create(builder, op.getLoc(), argPlus, shift);
        }
      }
      NpuAddressPatchOp::create(builder, op.getLoc(), op.getAddr(),
                                op.getAddrVal(), static_cast<int32_t>(*argIdx),
                                argPlus);
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEResolveAddressPatchBuffersPass() {
  return std::make_unique<AIEResolveAddressPatchBuffersPass>();
}
