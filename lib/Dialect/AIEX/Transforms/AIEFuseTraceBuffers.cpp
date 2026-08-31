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

// One traced sequence's region of the fused buffer, flattened across nesting.
struct TraceSlice {
  StringAttr device;
  StringAttr sequence;
  int64_t offset;
  int64_t size;
};

constexpr StringLiteral kTraceBufferAttr = "aie.trace_buffer";
constexpr StringLiteral kTraceSlicesAttr = "aie.trace_slices";

struct AIEFuseTraceBuffersPass
    : xilinx::AIEX::impl::AIEFuseTraceBuffersBase<AIEFuseTraceBuffersPass> {

  // Bytes each sequence claims for itself and everything it calls.
  llvm::DenseMap<Operation *, int64_t> claim;
  // Flattened slice list per sequence, in fused-buffer order.
  llvm::DenseMap<Operation *, SmallVector<TraceSlice>> slices;
  llvm::DenseSet<Operation *> done;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    for (auto device : mod.getOps<AIE::DeviceOp>())
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>())
        if (failed(fuse(seq)))
          return signalPassFailure();
  }

  static DictionaryAttr getTraceBufferAttr(AIE::RuntimeSequenceOp seq) {
    return seq->getAttrOfType<DictionaryAttr>(kTraceBufferAttr);
  }

  static int64_t getIntField(DictionaryAttr dict, StringRef name) {
    return cast<IntegerAttr>(dict.get(name)).getInt();
  }

  /// Give `seq` one trace buffer covering its own traces and those of every
  /// sequence it calls, and pass each call site its slice.
  LogicalResult fuse(AIE::RuntimeSequenceOp seq) {
    if (!done.insert(seq.getOperation()).second)
      return success();

    SmallVector<RunOp> runs;
    seq.walk([&](RunOp run) { runs.push_back(run); });

    // A callee's claim covers its own callees, so recurse before summing.
    for (RunOp run : runs) {
      AIE::RuntimeSequenceOp callee = run.getCalleeRuntimeSequenceOp();
      if (!callee)
        return run.emitOpError("cannot resolve callee runtime sequence");
      if (failed(fuse(callee)))
        return failure();
    }

    DictionaryAttr ownBuffer = getTraceBufferAttr(seq);
    int64_t ownClaim = ownBuffer ? getIntField(ownBuffer, "size") : 0;
    int64_t total = ownClaim;
    for (RunOp run : runs)
      total += claim.lookup(run.getCalleeRuntimeSequenceOp().getOperation());

    claim[seq.getOperation()] = total;
    if (total == 0)
      return success();

    // No callee claims to place: the argument from trace lowering already
    // covers this sequence.
    bool needsFusing = total > ownClaim;
    if (!needsFusing) {
      slices[seq.getOperation()] = ownSlices(seq, ownBuffer, /*base=*/0);
      return success();
    }

    if (ownBuffer && !cast<BoolAttr>(ownBuffer.get("dedicated")).getValue())
      return seq.emitOpError(
          "trace.host_config reuse_output_buffer=true cannot be combined with "
          "aiex.run calls into traced designs: their trace slices would have "
          "to grow the output buffer. Use a separate trace buffer instead");

    OpBuilder builder(seq);
    Block &entry = seq.getBody().front();
    auto i8 = IntegerType::get(seq.getContext(), 8);
    auto fusedType = MemRefType::get({total}, i8);

    BlockArgument fusedArg;
    if (ownBuffer) {
      fusedArg = entry.getArgument(getIntField(ownBuffer, "arg_index"));
      fusedArg.setType(fusedType);
    } else {
      fusedArg = entry.addArgument(fusedType, seq.getLoc());
    }

    // This sequence's own patches use offsets from the argument base, so its
    // traces must occupy the front.
    SmallVector<TraceSlice> flattened = ownSlices(seq, ownBuffer, /*base=*/0);
    int64_t cursor = ownClaim;

    for (RunOp run : runs) {
      AIE::RuntimeSequenceOp callee = run.getCalleeRuntimeSequenceOp();
      int64_t calleeClaim = claim.lookup(callee.getOperation());
      if (calleeClaim == 0)
        continue;

      DictionaryAttr calleeBuffer = getTraceBufferAttr(callee);
      if (calleeBuffer &&
          !cast<BoolAttr>(calleeBuffer.get("dedicated")).getValue())
        return run.emitOpError(
            "calls a design whose trace.host_config sets "
            "reuse_output_buffer=true. That design writes its trace data into "
            "its own output buffer, which this call site cannot slice. Use a "
            "separate trace buffer instead");
      unsigned calleeArgIdx =
          calleeBuffer
              ? static_cast<unsigned>(getIntField(calleeBuffer, "arg_index"))
              : callee.getBody().getNumArguments() - 1;
      auto calleeType = cast<MemRefType>(
          callee.getBody().getArgument(calleeArgIdx).getType());

      builder.setInsertionPoint(run);
      Value slice =
          makeSlice(builder, run.getLoc(), fusedArg, cursor, calleeType);
      run.getOperation()->insertOperands(run.getOperation()->getNumOperands(),
                                         {slice});

      for (TraceSlice s : slices.lookup(callee.getOperation())) {
        s.offset += cursor;
        flattened.push_back(s);
      }
      cursor += calleeClaim;
    }

    slices[seq.getOperation()] = flattened;
    seq->setAttr(kTraceSlicesAttr, buildSlicesAttr(builder, flattened));
    seq->setAttr(
        kTraceBufferAttr,
        builder.getDictionaryAttr(
            {builder.getNamedAttr("arg_index", builder.getI64IntegerAttr(
                                                   fusedArg.getArgNumber())),
             builder.getNamedAttr("offset", builder.getI64IntegerAttr(0)),
             builder.getNamedAttr("size", builder.getI64IntegerAttr(total)),
             builder.getNamedAttr("dedicated", builder.getBoolAttr(true))}));
    return success();
  }

  SmallVector<TraceSlice> ownSlices(AIE::RuntimeSequenceOp seq,
                                    DictionaryAttr ownBuffer, int64_t base) {
    SmallVector<TraceSlice> result;
    if (auto existing = seq->getAttrOfType<ArrayAttr>(kTraceSlicesAttr)) {
      for (Attribute a : existing) {
        auto d = cast<DictionaryAttr>(a);
        result.push_back(
            {d.getAs<StringAttr>("device"), d.getAs<StringAttr>("sequence"),
             base + getIntField(d, "offset"), getIntField(d, "size")});
      }
      return result;
    }
    if (!ownBuffer)
      return result;
    auto device = seq->getParentOfType<AIE::DeviceOp>();
    result.push_back({device.getSymNameAttr(), seq.getSymNameAttr(), base,
                      getIntField(ownBuffer, "size")});
    return result;
  }

  ArrayAttr buildSlicesAttr(OpBuilder &builder,
                            ArrayRef<TraceSlice> traceSlices) {
    SmallVector<Attribute> entries;
    for (const TraceSlice &s : traceSlices) {
      SmallVector<NamedAttribute> fields;
      if (s.device)
        fields.push_back(builder.getNamedAttr("device", s.device));
      if (s.sequence)
        fields.push_back(builder.getNamedAttr("sequence", s.sequence));
      fields.push_back(
          builder.getNamedAttr("offset", builder.getI64IntegerAttr(s.offset)));
      fields.push_back(
          builder.getNamedAttr("size", builder.getI64IntegerAttr(s.size)));
      entries.push_back(builder.getDictionaryAttr(fields));
    }
    return builder.getArrayAttr(entries);
  }

  /// A `size`-byte window of `buffer` at `offset`, typed as the callee's
  /// argument. `aiex.run` requires exact type equality, so the reinterpret_cast
  /// erases the subview's strided layout. `traceSubviewToBlockArgument` reads
  /// the byte offset from the subview.
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
