//===- AIEFuseTraceBuffers.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEFUSETRACEBUFFERS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIEFuseTraceBuffersPass
    : xilinx::AIEX::impl::AIEFuseTraceBuffersBase<AIEFuseTraceBuffersPass> {

  // Bytes each sequence claims for itself and everything it calls.
  llvm::DenseMap<Operation *, int64_t> claim;
  // Flattened slice list per sequence, in fused-buffer order.
  llvm::DenseMap<Operation *, SmallVector<AIE::TraceSliceAttr>> slices;
  llvm::DenseSet<Operation *> done;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    for (auto device : mod.getOps<AIE::DeviceOp>()) {
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>()) {
        if (failed(fuse(seq))) {
          return signalPassFailure();
        }
      }
    }
  }

  /// Give `seq` one trace buffer covering its own traces and those of every
  /// sequence it calls. Pass each call site its slice.
  LogicalResult fuse(AIE::RuntimeSequenceOp seq) {
    if (!done.insert(seq.getOperation()).second) {
      return success();
    }

    SmallVector<RunOp> runs;
    seq.walk([&](RunOp run) { runs.push_back(run); });

    // A callee's claim covers its own callees. Recurse before summing.
    for (RunOp run : runs) {
      AIE::RuntimeSequenceOp callee = run.getCalleeRuntimeSequenceOp();
      if (!callee) {
        return run.emitOpError("cannot resolve callee runtime sequence");
      }
      if (failed(fuse(callee))) {
        return failure();
      }
    }

    AIE::TraceBufferAttr ownBuffer = seq.getTraceBufferAttr();
    int64_t ownClaim = ownBuffer ? ownBuffer.getSize() : 0;
    int64_t total = ownClaim;
    for (RunOp run : runs) {
      total += claim.lookup(run.getCalleeRuntimeSequenceOp().getOperation());
    }

    claim[seq.getOperation()] = total;
    if (total == 0) {
      return success();
    }

    // No callee claims to place: the argument from trace lowering already
    // covers this sequence.
    bool needsFusing = total > ownClaim;
    if (!needsFusing) {
      slices[seq.getOperation()] = ownSlices(seq, ownBuffer, /*base=*/0);
      return success();
    }

    if (ownBuffer && !ownBuffer.getDedicated()) {
      return seq.emitOpError(
          "runs traced designs and sets trace.host_config "
          "reuse_output_buffer=true. Growing the output buffer to hold their "
          "slices is not supported. Use a separate trace buffer instead");
    }

    OpBuilder builder(seq);
    Block &entry = seq.getBody().front();
    auto i8 = IntegerType::get(seq.getContext(), 8);
    auto fusedType = MemRefType::get({total}, i8);

    BlockArgument fusedArg;
    if (ownBuffer) {
      fusedArg = entry.getArgument(ownBuffer.getArgIndex());
      fusedArg.setType(fusedType);
    } else {
      fusedArg = entry.addArgument(fusedType, seq.getLoc());
    }

    // This sequence's own patches use offsets from the argument base. Its
    // traces must occupy the front.
    SmallVector<AIE::TraceSliceAttr> flattened =
        ownSlices(seq, ownBuffer, /*base=*/0);
    int64_t cursor = ownClaim;

    for (RunOp run : runs) {
      AIE::RuntimeSequenceOp callee = run.getCalleeRuntimeSequenceOp();
      int64_t calleeClaim = claim.lookup(callee.getOperation());
      if (calleeClaim == 0) {
        continue;
      }

      AIE::TraceBufferAttr calleeBuffer = callee.getTraceBufferAttr();
      if (calleeBuffer && !calleeBuffer.getDedicated()) {
        return run.emitOpError(
            "calls a design that sets trace.host_config "
            "reuse_output_buffer=true. That design writes its trace data into "
            "its own output buffer, so this call site has no argument to "
            "slice. Slicing an output buffer is not supported. Use a separate "
            "trace buffer instead");
      }
      unsigned calleeArgIdx = calleeBuffer
                                  ? calleeBuffer.getArgIndex()
                                  : callee.getBody().getNumArguments() - 1;
      auto calleeType = cast<MemRefType>(
          callee.getBody().getArgument(calleeArgIdx).getType());

      builder.setInsertionPoint(run);
      Value slice =
          makeSlice(builder, run.getLoc(), fusedArg, cursor, calleeType);
      run.getOperation()->insertOperands(run.getOperation()->getNumOperands(),
                                         {slice});

      for (AIE::TraceSliceAttr s : slices.lookup(callee.getOperation())) {
        flattened.push_back(AIE::TraceSliceAttr::get(
            seq.getContext(), s.getDevice(), s.getSequence(),
            s.getOffset() + cursor, s.getSize()));
      }
      cursor += calleeClaim;
    }

    slices[seq.getOperation()] = flattened;
    seq.setTraceSlicesAttr(builder.getArrayAttr(
        SmallVector<Attribute>(flattened.begin(), flattened.end())));
    seq.setTraceBufferAttr(AIE::TraceBufferAttr::get(
        seq.getContext(), fusedArg.getArgNumber(), /*offset=*/0, total,
        /*dedicated=*/true));
    return success();
  }

  SmallVector<AIE::TraceSliceAttr> ownSlices(AIE::RuntimeSequenceOp seq,
                                             AIE::TraceBufferAttr ownBuffer,
                                             int64_t base) {
    SmallVector<AIE::TraceSliceAttr> result;
    if (ArrayAttr existing = seq.getTraceSlicesAttr()) {
      for (Attribute a : existing) {
        auto s = cast<AIE::TraceSliceAttr>(a);
        result.push_back(AIE::TraceSliceAttr::get(
            seq.getContext(), s.getDevice(), s.getSequence(),
            base + s.getOffset(), s.getSize()));
      }
      return result;
    }
    if (!ownBuffer) {
      return result;
    }
    auto device = seq->getParentOfType<AIE::DeviceOp>();
    StringAttr deviceName = device.getSymNameAttr();
    result.push_back(AIE::TraceSliceAttr::get(
        seq.getContext(), deviceName ? deviceName.getValue() : StringRef(),
        seq.getSymName(), base, ownBuffer.getSize()));
    return result;
  }

  /// A `size`-byte window of `buffer` at `offset`, typed as the callee's
  /// argument. `aiex.run` requires exact type equality. The reinterpret_cast
  /// therefore erases the subview's strided layout.
  /// `traceSubviewToBlockArgument` reads the byte offset from the subview.
  Value makeSlice(OpBuilder &builder, Location loc, Value buffer,
                  int64_t offset, MemRefType resultType) {
    int64_t size = resultType.getNumElements();
    Value sub = memref::SubViewOp::create(
        builder, loc, buffer,
        ArrayRef<OpFoldResult>{builder.getIndexAttr(offset)},
        ArrayRef<OpFoldResult>{builder.getIndexAttr(size)},
        ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
    return memref::ReinterpretCastOp::create(
        builder, loc, resultType, sub, builder.getIndexAttr(0),
        ArrayRef<OpFoldResult>{builder.getIndexAttr(size)},
        ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEFuseTraceBuffersPass() {
  return std::make_unique<AIEFuseTraceBuffersPass>();
}
