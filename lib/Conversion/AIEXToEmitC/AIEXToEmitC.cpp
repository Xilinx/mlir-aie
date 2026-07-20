//===- AIEXToEmitC.cpp - AIEX runtime sequence to EmitC --------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts the straight-line npu transaction ops in an aie.runtime_sequence
// into EmitC dialect calls naming the functions in aie/Runtime/TxnEncoding.h.
// translateToCpp() on the result produces a standalone C++ function that
// assembles the same TXN words the compile-time binary emitter
// (AIETargetNPU.cpp) produces -- this is the runtime-parameterizable mirror of
// that path. This pass handles straight-line sequences only; control flow is
// rejected with a diagnostic and added by later dynamic-sequences passes.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEXToEmitC/AIEXToEmitC.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

namespace xilinx {
#define GEN_PASS_DEF_CONVERTAIEXTOEMITC
#include "aie/Conversion/Passes.h.inc"
} // namespace xilinx

using namespace mlir;
using namespace xilinx;

namespace {

// The C++ type of the TXN accumulator the generated function builds.
constexpr llvm::StringLiteral kTxnVecType = "std::vector<uint32_t>";

emitc::OpaqueType getU32Type(MLIRContext *ctx) {
  return emitc::OpaqueType::get(ctx, "uint32_t");
}

// A uint32_t literal operand (e.g. `42u`) for a txn_append_* argument. Used for
// the compile-time-resolved address fields (buffer/col/row are folded in).
Value u32Literal(OpBuilder &b, Location loc, uint32_t val) {
  return emitc::ConstantOp::create(
      b, loc, getU32Type(b.getContext()),
      emitc::OpaqueAttr::get(b.getContext(), std::to_string(val) + "u"));
}

// Emit a call to one of TxnEncoding.h's free functions on the txn vector.
// Scalar operands are passed through as-is: an arith.constant or a runtime
// runtime-sequence value lowers to the matching C++ literal or function
// parameter by the later convert-arith-to-emitc step. TxnEncoding.h's uint32_t
// parameters take the implicit conversion, byte-identical to the static path.
void emitTxnCall(OpBuilder &b, Location loc, StringRef fn, Value txnVec,
                 ValueRange args) {
  SmallVector<Value> callArgs;
  callArgs.push_back(txnVec);
  callArgs.append(args.begin(), args.end());
  emitc::CallOpaqueOp::create(b, loc, TypeRange{}, ("aie_runtime::" + fn).str(),
                              callArgs);
}

// Device-relative values resolved on the original in-device ops, keyed by their
// clone in the emitc.func. getAbsoluteAddress()/getDataWords() resolve
// buffer/column/row and memref.global symbols against the parent aie.device, so
// they must be computed before the ops are cloned out of it.
struct DeviceResolved {
  llvm::DenseMap<Operation *, uint32_t> absoluteAddr;
  llvm::DenseMap<Operation *, DenseIntElementsAttr> blockWriteData;
};

class AIEXToEmitCConverter {
public:
  AIEXToEmitCConverter(emitc::FuncOp funcOp, Value txnVec,
                       const DeviceResolved &resolved)
      : funcOp(funcOp), txnVec(txnVec), resolved(resolved) {}

  // Convert every straight-line op cloned into the function body. Returns the
  // number of txn ops appended (the op_count for the header), or nullopt on a
  // conversion error already diagnosed.
  std::optional<uint32_t> run() {
    Block &body = funcOp.getBlocks().front();
    SmallVector<Operation *> work;
    for (Operation &op : body)
      work.push_back(&op);

    uint32_t count = 0;
    SmallVector<Operation *> consumed;
    for (Operation *op : work) {
      // Structural emitc ops (txn vector decl, txn_init) are already done.
      // arith ops (constant scalar fields and runtime-value arithmetic) are
      // left in place for the later convert-arith-to-emitc step -- passing
      // operands through uniformly is what lets runtime values flow into the
      // C++.
      if (isa<emitc::EmitCDialect, arith::ArithDialect>(op->getDialect()))
        continue;
      convertOne(op, count);
      if (!ok)
        return std::nullopt;
      // The npu op is replaced by its emitc call; mark it for erase. A
      // memref.get_global feeding a blockwrite is now dead, but a later
      // blockwrite may still reference it, so defer its erase to the dead-op
      // sweep below.
      if (!isa<memref::GetGlobalOp>(op))
        consumed.push_back(op);
    }

    // Erase the converted npu ops, then sweep now-dead clones back to front so
    // a def outlives its uses (e.g. get_global feeding a blockwrite, or the
    // arith.constant that defined an address we replaced with a folded
    // literal). Live arith feeding runtime value/mask/sync fields has uses and
    // is kept for the arith-to-emitc step.
    for (Operation *op : llvm::reverse(consumed))
      op->erase();
    for (Operation *op : llvm::reverse(work))
      if (op->getBlock() &&
          isa<memref::MemRefDialect, arith::ArithDialect>(op->getDialect()) &&
          op->use_empty())
        op->erase();
    return count;
  }

private:
  // Convert one op, appending to the txn vector. Increments `count` for each
  // txn instruction emitted. Scalar operands of the (already-cloned) op are
  // passed through directly -- a constant or a runtime value both flow as SSA,
  // lowered later by convert-arith-to-emitc. Only the address field is replaced
  // by its device-resolved literal (which folds in buffer/col/row). Unsupported
  // ops are diagnosed (sets `ok` false).
  void convertOne(Operation *op, uint32_t &count) {
    OpBuilder b(op);
    Location loc = op->getLoc();

    llvm::TypeSwitch<Operation *, void>(op)
        .Case<AIEX::NpuWrite32Op>([&](auto w) {
          Value addrV = resolvedAddr(b, loc, w);
          if (!addrV)
            return fail(w, "cannot convert a symbolic/unresolved write32 "
                           "address to the C++ TXN target");
          emitTxnCall(b, loc, "txn_append_write32", txnVec,
                      {addrV, w.getValue()});
          ++count;
        })
        .Case<AIEX::NpuMaskWrite32Op>([&](auto mw) {
          Value addrV = resolvedAddr(b, loc, mw);
          if (!addrV)
            return fail(mw, "cannot convert a symbolic/unresolved maskwrite32 "
                            "address to the C++ TXN target");
          emitTxnCall(b, loc, "txn_append_maskwrite32", txnVec,
                      {addrV, mw.getValue(), mw.getMask()});
          ++count;
        })
        .Case<AIEX::NpuSyncOp>([&](auto s) {
          emitTxnCall(b, loc, "txn_append_sync", txnVec,
                      {s.getColumn(), s.getRow(), s.getDirection(),
                       s.getChannel(), s.getColumnNum(), s.getRowNum()});
          ++count;
        })
        .Case<AIEX::NpuAddressPatchOp>([&](auto ap) {
          Value addrV = u32Literal(b, loc, ap.getAddr());
          Value idxV = emitc::ConstantOp::create(
              b, loc, emitc::OpaqueType::get(b.getContext(), "int32_t"),
              emitc::OpaqueAttr::get(b.getContext(),
                                     std::to_string(ap.getArgIdx())));
          emitTxnCall(b, loc, "txn_append_address_patch", txnVec,
                      {addrV, idxV, ap.getArgPlus()});
          ++count;
        })
        .Case<AIEX::NpuBlockWriteOp>([&](auto bw) {
          convertBlockWrite(b, loc, bw);
          ++count;
        })
        .Case<AIEX::NpuAssertBdFieldOp>([&](auto g) {
          // Host-side bounds guard: if the runtime value overflows its narrow
          // BD field, the builder yields no stream (std::nullopt) rather than a
          // truncated one. Appends nothing, so not counted.
          emitc::VerbatimOp::create(b, loc,
                                    "if ({} > " + std::to_string(g.getMax()) +
                                        ") return std::nullopt;",
                                    ValueRange{g.getValue()});
        })
        .Case<AIEX::NpuAssertBdDivisibleOp>([&](auto g) {
          // Host-side realizability guard: a runtime size/stride whose byte
          // extent isn't a whole number of granules can't be encoded, so the
          // builder yields no stream. allow_unit exempts a unit stride (the
          // contiguous sub-granule case). Appends nothing, so not counted.
          std::string d = std::to_string(g.getDivisor());
          if (g.getAllowUnit())
            emitc::VerbatimOp::create(b, loc,
                                      "if ({} != 1 && {} % " + d +
                                          " != 0) return "
                                          "std::nullopt;",
                                      ValueRange{g.getValue(), g.getValue()});
          else
            emitc::VerbatimOp::create(
                b, loc, "if ({} % " + d + " != 0) return std::nullopt;",
                ValueRange{g.getValue()});
        })
        // memref.get_global feeding a blockwrite is consumed by
        // convertBlockWrite (data inlined); the now-dead op is erased later.
        .Case<memref::GetGlobalOp>([&](auto) {})
        .Default([&](Operation *unsupported) {
          if (isa<scf::SCFDialect>(unsupported->getDialect()))
            fail(unsupported,
                 "control flow in runtime sequences is not yet supported by "
                 "the C++ TXN target");
          else
            fail(unsupported, "is not supported by the C++ TXN target");
        });
  }

  // The compile-time-resolved absolute address (folding buffer/col/row) as a
  // uint32_t literal, or null if it could not be resolved statically. Address
  // resolution depends on the parent device, so it is precomputed before the op
  // is cloned out (see recordDeviceResolved).
  Value resolvedAddr(OpBuilder &b, Location loc, Operation *clone) {
    auto it = resolved.absoluteAddr.find(clone);
    if (it == resolved.absoluteAddr.end())
      return {};
    return u32Literal(b, loc, it->second);
  }

  // Emit a diagnostic on `op` and mark the conversion failed.
  void fail(Operation *op, const llvm::Twine &msg) {
    op->emitOpError(msg);
    ok = false;
  }

  void convertBlockWrite(OpBuilder &b, Location loc, AIEX::NpuBlockWriteOp bw) {
    auto addrIt = resolved.absoluteAddr.find(bw);
    auto dataIt = resolved.blockWriteData.find(bw);
    if (addrIt == resolved.absoluteAddr.end() ||
        dataIt == resolved.blockWriteData.end() || !dataIt->second) {
      bw.emitOpError("cannot convert blockwrite without a constant address and "
                     "data to the C++ TXN target");
      ok = false;
      return;
    }
    uint32_t addr = addrIt->second;
    DenseIntElementsAttr data = dataIt->second;

    // Emit the payload as a typed EmitC array variable rather than a verbatim
    // C++ string, so the generated code carries real types.
    MLIRContext *ctx = b.getContext();
    int64_t n = data.size();
    auto arrTy =
        emitc::ArrayType::get(ctx, SmallVector<int64_t>{n}, getU32Type(ctx));
    std::string init = "{";
    llvm::raw_string_ostream ss(init);
    bool first = true;
    for (APInt word : data.getValues<APInt>()) {
      if (!first)
        ss << ", ";
      ss << llvm::format("0x%08xu", static_cast<unsigned>(word.getZExtValue()));
      first = false;
    }
    ss << "}";
    auto arrVar = emitc::VariableOp::create(b, loc, arrTy,
                                            emitc::OpaqueAttr::get(ctx, init));

    uint32_t colVal = 0, rowVal = 0;
    if (bw.getColumn() && bw.getRow()) {
      colVal = *bw.getColumn();
      rowVal = *bw.getRow();
    }
    emitTxnCall(b, loc, "txn_append_blockwrite", txnVec,
                {u32Literal(b, loc, addr), arrVar.getResult(),
                 u32Literal(b, loc, static_cast<uint32_t>(n)),
                 u32Literal(b, loc, colVal), u32Literal(b, loc, rowVal)});
  }

  emitc::FuncOp funcOp;
  Value txnVec;
  const DeviceResolved &resolved;
  bool ok = true;
};

struct ConvertAIEXToEmitCPass
    : xilinx::impl::ConvertAIEXToEmitCBase<ConvertAIEXToEmitCPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    struct SeqInfo {
      AIE::RuntimeSequenceOp seq;
      AIE::DeviceOp device;
    };
    SmallVector<SeqInfo> sequences;
    moduleOp.walk([&](AIE::RuntimeSequenceOp seq) {
      sequences.push_back({seq, seq->getParentOfType<AIE::DeviceOp>()});
    });
    if (sequences.empty())
      return;

    OpBuilder builder(ctx);
    for (auto &[seqOp, deviceOp] : sequences) {
      if (!deviceOp) {
        seqOp.emitOpError("must be nested inside an aie.device");
        return signalPassFailure();
      }
      if (failed(emitFunction(builder, moduleOp, seqOp, deviceOp)))
        return signalPassFailure();
    }

    // Replace the module body with just the generated emitc funcs + includes.
    SmallVector<Operation *> toErase;
    for (Operation &op : *moduleOp.getBody())
      if (!isa<emitc::FuncOp, emitc::IncludeOp>(op))
        toErase.push_back(&op);
    for (Operation *op : llvm::reverse(toErase))
      op->erase();

    // Lower the arith ops left behind in the function bodies (constant scalar
    // fields and any runtime-value arithmetic feeding npu ops) to emitc. scf
    // was already rejected, so only arith needs lowering here.
    if (failed(lowerArithToEmitC(moduleOp)))
      return signalPassFailure();

    builder.setInsertionPointToStart(moduleOp.getBody());
    emitc::IncludeOp::create(builder, moduleOp.getLoc(),
                             "aie/Runtime/TxnEncoding.h",
                             /*is_standard=*/false);
    emitc::IncludeOp::create(builder, moduleOp.getLoc(), "cstdint",
                             /*is_standard=*/true);
    emitc::IncludeOp::create(builder, moduleOp.getLoc(), "vector",
                             /*is_standard=*/true);
    // The builder returns std::optional: a runtime scalar that would overflow a
    // narrow BD field yields std::nullopt instead of a truncated stream.
    emitc::IncludeOp::create(builder, moduleOp.getLoc(), "optional",
                             /*is_standard=*/true);
  }

  // Run the upstream arith-to-emitc conversion over the generated functions.
  LogicalResult lowerArithToEmitC(ModuleOp moduleOp) {
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    populateEmitCSizeTTypeConversions(typeConverter);

    RewritePatternSet patterns(moduleOp.getContext());
    populateArithToEmitCPatterns(typeConverter, patterns);

    ConversionTarget target(*moduleOp.getContext());
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    return applyPartialConversion(moduleOp, target, std::move(patterns));
  }

private:
  // Resolve `orig`'s device-dependent values (still under the device) keyed by
  // its detached `clone`.
  static void recordDeviceResolved(Operation *orig, Operation *clone,
                                   DeviceResolved &resolved) {
    if (auto w = dyn_cast<AIEX::NpuWrite32Op>(orig)) {
      if (auto a = w.getAbsoluteAddress())
        resolved.absoluteAddr[clone] = *a;
    } else if (auto mw = dyn_cast<AIEX::NpuMaskWrite32Op>(orig)) {
      if (auto a = mw.getAbsoluteAddress())
        resolved.absoluteAddr[clone] = *a;
    } else if (auto bw = dyn_cast<AIEX::NpuBlockWriteOp>(orig)) {
      if (auto a = bw.getAbsoluteAddress())
        resolved.absoluteAddr[clone] = *a;
      if (auto d = bw.getDataWords())
        resolved.blockWriteData[clone] = d;
    }
  }

  LogicalResult emitFunction(OpBuilder &builder, ModuleOp moduleOp,
                             AIE::RuntimeSequenceOp seqOp,
                             AIE::DeviceOp deviceOp) {
    Location loc = seqOp.getLoc();
    Block &entry = seqOp.getBody().front();

    // The generated function takes the sequence's non-memref args (the runtime
    // scalars) and returns the assembled txn vector. Memref args address host
    // buffers resolved by address_patch; they are not function parameters here.
    SmallVector<Type> paramTypes;
    auto txnVecType =
        emitc::OpaqueType::get(moduleOp.getContext(), kTxnVecType);
    // The builder can fail at TXN-build time (a runtime scalar overflowing a
    // narrow BD field), so it returns std::optional<std::vector<uint32_t>>:
    // std::nullopt on a failed guard, the assembled vector otherwise.
    auto txnRetType = emitc::OpaqueType::get(
        moduleOp.getContext(),
        (llvm::Twine("std::optional<") + kTxnVecType + ">").str());
    for (BlockArgument arg : entry.getArguments())
      if (!isa<BaseMemRefType>(arg.getType()))
        paramTypes.push_back(arg.getType());

    builder.setInsertionPointToEnd(moduleOp.getBody());
    std::string funcName = "generate_txn_" + deviceOp.getSymName().str() + "_" +
                           seqOp.getSymName().str();
    auto funcOp = emitc::FuncOp::create(
        builder, loc, funcName,
        FunctionType::get(moduleOp.getContext(), paramTypes,
                          TypeRange{txnRetType}));
    funcOp.setSpecifiersAttr(builder.getStrArrayAttr({"inline"}));
    Block *funcBlock = funcOp.addEntryBlock();
    OpBuilder fb = OpBuilder::atBlockBegin(funcBlock);

    // EmitC has no native std::vector construction, so declare the accumulator
    // with a verbatim statement and reference it by name via an emitc.literal
    // (which translates inline, no extra copy). All txn_append_* calls take
    // this value, so the generated C++ operates on the single `txn` vector.
    emitc::VerbatimOp::create(fb, loc, "std::vector<uint32_t> txn;");
    Value txnVec = emitc::LiteralOp::create(fb, loc, txnVecType, "txn");
    emitTxnCall(fb, loc, "txn_init", txnVec, {});

    // Clone the straight-line sequence body into the function, mapping the
    // non-memref block args to the function parameters. While the originals are
    // still under the device, resolve their device-relative values (absolute
    // address, blockwrite data) keyed by the clone -- the converter reads these
    // instead of calling device-dependent accessors on the detached clones.
    IRMapping mapping;
    unsigned p = 0;
    for (BlockArgument arg : entry.getArguments())
      if (!isa<BaseMemRefType>(arg.getType()))
        mapping.map(arg, funcBlock->getArgument(p++));
    DeviceResolved resolved;
    for (Operation &op : entry.without_terminator()) {
      Operation *clone = fb.clone(op, mapping);
      recordDeviceResolved(&op, clone, resolved);
    }

    AIEXToEmitCConverter conv(funcOp, txnVec, resolved);
    std::optional<uint32_t> count = conv.run();
    if (!count)
      return failure();

    // Finalize the header with the statically-known op count + device info.
    const AIE::AIETargetModel &tm = deviceOp.getTargetModel();
    uint8_t devGen = isa<AIE::BaseNPU2TargetModel>(tm) ? 4 : 3;
    OpBuilder eb(funcBlock, funcBlock->end());
    std::string header =
        "aie_runtime::txn_prepend_header(txn, " + std::to_string(*count) +
        "u, {0, 1, " + std::to_string(devGen) + ", " +
        std::to_string(tm.rows()) + ", " + std::to_string(tm.columns()) + ", " +
        std::to_string(tm.getNumMemTileRows()) + "});";
    emitc::VerbatimOp::create(eb, loc, header);
    // Return the vector as the optional result. The literal text differs from
    // the "txn" accumulator literal (so the two are not deduplicated) and is
    // typed as the optional return so emitc's return-type check passes; it
    // relies on the implicit std::vector -> std::optional conversion in C++.
    Value ret = emitc::LiteralOp::create(eb, loc, txnRetType, "std::move(txn)");
    emitc::ReturnOp::create(eb, loc, ret);
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::createConvertAIEXToEmitCPass() {
  return std::make_unique<ConvertAIEXToEmitCPass>();
}
