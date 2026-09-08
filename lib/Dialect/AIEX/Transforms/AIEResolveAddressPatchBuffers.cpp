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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIERESOLVEADDRESSPATCHBUFFERS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct ResolveBufferOperand : OpRewritePattern<NpuAddressPatchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NpuAddressPatchOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getBuffer()) {
      return failure();
    }

    auto seq = op->getParentOfType<AIE::RuntimeSequenceOp>();
    if (!seq) {
      return rewriter.notifyMatchFailure(
          op, "must be inside a runtime sequence to resolve its host buffer");
    }
    std::optional<SubviewTraceResult> traced =
        traceSubviewToBlockArgument(op.getBuffer());
    if (!traced || traced->rootArg.getOwner() != &seq.getBody().front()) {
      return rewriter.notifyMatchFailure(
          op, "buffer must be an argument of the enclosing runtime sequence, "
              "or a static contiguous view of one");
    }
    std::optional<unsigned> argIdx = getHostBufferArgIndex(traced->rootArg);
    if (!argIdx) {
      return rewriter.notifyMatchFailure(
          op, "buffer must resolve to a memref argument");
    }

    Value argPlus = op.getArgPlus();
    if (traced->offsetInBytes != 0) {
      std::optional<uint32_t> base = getConstantIntOperand(argPlus);
      if (base) {
        argPlus = createConstantI32(rewriter, op.getLoc(),
                                    *base + traced->offsetInBytes);
      } else {
        Value shift =
            createConstantI32(rewriter, op.getLoc(), traced->offsetInBytes);
        argPlus = arith::AddIOp::create(rewriter, op.getLoc(), argPlus, shift);
      }
    }

    rewriter.replaceOpWithNewOp<NpuAddressPatchOp>(
        op, op.getAddr(), op.getAddrVal(), static_cast<int32_t>(*argIdx),
        argPlus);
    return success();
  }
};

struct AIEResolveAddressPatchBuffersPass
    : xilinx::AIEX::impl::AIEResolveAddressPatchBuffersBase<
          AIEResolveAddressPatchBuffersPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ResolveBufferOperand>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    // A pattern that declines to match leaves the operand in place. The targets
    // cannot emit that, so report it here instead.
    WalkResult unresolved = getOperation().walk([&](NpuAddressPatchOp op) {
      if (op.getBuffer()) {
        op.emitOpError("could not resolve its buffer operand to a host buffer "
                       "index of the enclosing runtime sequence");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (unresolved.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIEResolveAddressPatchBuffersPass() {
  return std::make_unique<AIEResolveAddressPatchBuffersPass>();
}
