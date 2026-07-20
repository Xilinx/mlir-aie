//===- AIEXToEmitC.cpp - AIEX runtime sequence to EmitC --------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts npu transaction ops in an aie.runtime_sequence into EmitC calls to
// aie/Runtime/TxnEncoding.h; translateToCpp() then yields a standalone C++
// function assembling the same TXN words as the binary emitter
// (AIETargetNPU.cpp), but runtime-parameterizable. A runtime-bound scf.for
// (dynamic BD pool path) is preserved and lowered to emitc.for, keeping the
// loop rolled at runtime.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEXToEmitC/AIEXToEmitC.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallSet.h"
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

// The C++ variable name of the runtime BD pool for a tile. One pool per tile
// that draws BD ids at runtime, declared in the generated function's prologue.
std::string bdPoolName(uint32_t col, uint32_t row) {
  return "bd_pool_" + std::to_string(col) + "_" + std::to_string(row);
}

// A uint32_t literal operand (e.g. `42u`) for a txn_append_* argument. Used for
// the compile-time-resolved address fields (buffer/col/row are folded in).
Value u32Literal(OpBuilder &b, Location loc, uint32_t val) {
  return emitc::ConstantOp::create(
      b, loc, getU32Type(b.getContext()),
      emitc::OpaqueAttr::get(b.getContext(), std::to_string(val) + "u"));
}

// Emit a call to a TxnEncoding.h free function on the txn vector. Scalar
// operands pass through as-is; convert-arith-to-emitc later lowers them to C++
// literals or function parameters, byte-identical to the static path.
void emitTxnCall(OpBuilder &b, Location loc, StringRef fn, Value txnVec,
                 ValueRange args) {
  SmallVector<Value> callArgs;
  callArgs.push_back(txnVec);
  callArgs.append(args.begin(), args.end());
  emitc::CallOpaqueOp::create(b, loc, TypeRange{}, ("aie_runtime::" + fn).str(),
                              callArgs);
}

// Device-relative values (absolute addr, blockwrite data) resolved before the
// ops are cloned out of the aie.device, keyed by their clone in the emitc.func.
struct DeviceResolved {
  llvm::DenseMap<Operation *, uint32_t> absoluteAddr;
  llvm::DenseMap<Operation *, DenseIntElementsAttr> blockWriteData;
};

class AIEXToEmitCConverter {
public:
  AIEXToEmitCConverter(emitc::FuncOp funcOp, Value txnVec,
                       const DeviceResolved &resolved, Value opCountVar)
      : funcOp(funcOp), txnVec(txnVec), resolved(resolved),
        opCountVar(opCountVar) {}

  // Convert every npu/pool op in the body, recursing into scf regions. Returns
  // the compile-time op count for a straight-line body (the literal op_count),
  // or nullopt on error. With a runtime op-count var (a loop is present) each
  // op increments it instead and the returned count is unused.
  std::optional<uint32_t> run() {
    uint32_t count = 0;
    SmallVector<Operation *> consumed;
    convertBlockRecursive(funcOp.getBlocks().front(), count, consumed);
    if (!ok)
      return std::nullopt;

    // Erase the converted npu ops, then sweep now-dead memref/arith clones
    // (e.g. a get_global or folded-away address constant). Live arith feeding
    // runtime fields is kept for arith-to-emitc. Collect first: erasing
    // mid-walk is unsafe.
    for (Operation *op : llvm::reverse(consumed))
      op->erase();
    SmallVector<Operation *> deadSweep;
    funcOp.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (isa<memref::MemRefDialect, arith::ArithDialect>(op->getDialect()) &&
          op->use_empty())
        deadSweep.push_back(op);
    });
    for (Operation *op : deadSweep)
      op->erase();
    return count;
  }

private:
  // Convert the npu/pool ops in `block`, recursing through scf op regions so a
  // rolled loop's body is lowered in place (the scf.for itself is left for the
  // later convert-scf-to-emitc step). Collects converted ops into `consumed`.
  void convertBlockRecursive(Block &block, uint32_t &count,
                             SmallVector<Operation *> &consumed) {
    SmallVector<Operation *> work;
    for (Operation &op : block)
      work.push_back(&op);
    for (Operation *op : work) {
      if (!ok)
        return;
      // Structural emitc ops and arith (constant / runtime-value math) are left
      // for the later arith-to-emitc step. Recurse into scf ops (and any other
      // region-carrying control flow) to convert their bodies; the control-flow
      // op itself is handled by convert-scf-to-emitc afterwards.
      if (isa<emitc::EmitCDialect, arith::ArithDialect>(op->getDialect()))
        continue;
      if (isa<scf::SCFDialect>(op->getDialect())) {
        for (Region &r : op->getRegions())
          for (Block &b : r)
            convertBlockRecursive(b, count, consumed);
        continue;
      }
      convertOne(op, count);
      if (!isa<memref::GetGlobalOp>(op))
        consumed.push_back(op);
    }
  }

private:
  // Record one emitted txn op: bump the compile-time count and, when a runtime
  // op-count variable exists (the sequence has a loop, so the count is not
  // known at compile time), emit a `++__opcount;` so the header gets the true
  // runtime total.
  void countOp(OpBuilder &b, Location loc, uint32_t &count) {
    ++count;
    if (opCountVar)
      emitc::VerbatimOp::create(b, loc, "++{};", ValueRange{opCountVar});
  }

  // Convert one op, appending to the txn vector and incrementing `count`.
  // Scalar operands flow through as SSA (lowered later by
  // convert-arith-to-emitc); only the address field is replaced by its
  // device-resolved literal. Unsupported ops set `ok` false.
  void convertOne(Operation *op, uint32_t &count) {
    OpBuilder b(op);
    Location loc = op->getLoc();

    llvm::TypeSwitch<Operation *, void>(op)
        .Case<AIEX::NpuWrite32Op>([&](auto w) {
          // A compile-time address folds to a literal; a runtime address (the
          // dynamic BD pool's bdBase + wordIdx, arith over a popped id) is
          // passed through as SSA and lowered by convert-arith-to-emitc.
          Value addrV = runtimeOrResolvedAddr(b, loc, w, w.getAddress());
          if (!addrV)
            return fail(w, "cannot convert a symbolic/unresolved write32 "
                           "address to the C++ TXN target");
          emitTxnCall(b, loc, "txn_append_write32", txnVec,
                      {addrV, w.getValue()});
          countOp(b, loc, count);
        })
        .Case<AIEX::NpuMaskWrite32Op>([&](auto mw) {
          Value addrV = runtimeOrResolvedAddr(b, loc, mw, mw.getAddress());
          if (!addrV)
            return fail(mw, "cannot convert a symbolic/unresolved maskwrite32 "
                            "address to the C++ TXN target");
          emitTxnCall(b, loc, "txn_append_maskwrite32", txnVec,
                      {addrV, mw.getValue(), mw.getMask()});
          countOp(b, loc, count);
        })
        .Case<AIEX::NpuSyncOp>([&](auto s) {
          emitTxnCall(b, loc, "txn_append_sync", txnVec,
                      {s.getColumn(), s.getRow(), s.getDirection(),
                       s.getChannel(), s.getColumnNum(), s.getRowNum()});
          countOp(b, loc, count);
        })
        .Case<AIEX::NpuAddressPatchOp>([&](auto ap) {
          // A runtime bd_id makes the patched register address runtime: prefer
          // the SSA addr_val (flows through arith-to-emitc) over the constant.
          Value addrV = ap.getAddrVal() ? ap.getAddrVal()
                                        : u32Literal(b, loc, ap.getAddr());
          Value idxV = emitc::ConstantOp::create(
              b, loc, emitc::OpaqueType::get(b.getContext(), "int32_t"),
              emitc::OpaqueAttr::get(b.getContext(),
                                     std::to_string(ap.getArgIdx())));
          emitTxnCall(b, loc, "txn_append_address_patch", txnVec,
                      {addrV, idxV, ap.getArgPlus()});
          countOp(b, loc, count);
        })
        .Case<AIEX::NpuBlockWriteOp>([&](auto bw) {
          convertBlockWrite(b, loc, bw);
          countOp(b, loc, count);
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
        .Case<AIEX::DMABdPoolPopOp>([&](AIEX::DMABdPoolPopOp pop) {
          // Draw a BD id from the tile's runtime pool (declared in the
          // prologue) into a fresh C++ variable; nullopt if the pool is empty.
          // Downstream ops consume the id, so replace the SSA result with a
          // literal naming the variable.
          std::string var = "bd_" + std::to_string(nextPoolVar++);
          emitc::VerbatimOp::create(
              b, loc,
              "uint32_t " + var + "; if (!aie_runtime::bd_pool_pop(" +
                  bdPoolName(pop.getColumn(), pop.getRow()) + ", " + var +
                  ")) return std::nullopt;");
          Value ref =
              emitc::LiteralOp::create(b, loc, pop.getBdId().getType(), var);
          pop.getBdId().replaceAllUsesWith(ref);
        })
        .Case<AIEX::DMABdPoolPushOp>([&](AIEX::DMABdPoolPushOp push) {
          // Return a BD id to the tile's runtime pool.
          emitc::CallOpaqueOp::create(
              b, loc, TypeRange{}, "aie_runtime::bd_pool_push",
              ValueRange{poolRef(b, loc, push.getColumn(), push.getRow()),
                         push.getBdId()});
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

  // The address for a write32/maskwrite32: the device-resolved literal, else a
  // genuine runtime address (arith-derived, e.g. the pool's bdBase + wordIdx).
  // A bare block argument selects no register and is rejected; null for a
  // symbolic/unresolved address.
  Value runtimeOrResolvedAddr(OpBuilder &b, Location loc, Operation *clone,
                              Value addrOperand) {
    if (Value lit = resolvedAddr(b, loc, clone))
      return lit;
    Operation *def = addrOperand ? addrOperand.getDefiningOp() : nullptr;
    if (def && !isa<arith::ConstantOp>(def))
      return addrOperand;
    return {};
  }

  // Emit a diagnostic on `op` and mark the conversion failed.
  void fail(Operation *op, const llvm::Twine &msg) {
    op->emitOpError(msg);
    ok = false;
  }

  // An emitc.literal referencing a tile's pool variable (for pass-by-reference
  // into bd_pool_push).
  Value poolRef(OpBuilder &b, Location loc, uint32_t col, uint32_t row) {
    return emitc::LiteralOp::create(
        b, loc, emitc::OpaqueType::get(b.getContext(), "aie_runtime::BdPool"),
        bdPoolName(col, row));
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
  // Runtime op-count variable (the C++ `__opcount`), or null when the sequence
  // is straight-line and the count is a compile-time literal.
  Value opCountVar;
  bool ok = true;
  // Counter for unique popped-BD-id C++ variable names.
  unsigned nextPoolVar = 0;
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

    // Lower the arith ops (constant scalar fields, runtime-value arithmetic
    // feeding npu ops) and any scf control flow (a rolled dynamic loop) left in
    // the function bodies to emitc.
    if (failed(lowerArithAndScfToEmitC(moduleOp)))
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

  // Run the upstream arith-to-emitc and scf-to-emitc conversions over the
  // generated functions in one pass: a rolled dynamic loop is scf.for, and its
  // bounds / iter_args are arith, so both must lower together.
  LogicalResult lowerArithAndScfToEmitC(ModuleOp moduleOp) {
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    populateEmitCSizeTTypeConversions(typeConverter);

    RewritePatternSet patterns(moduleOp.getContext());
    populateArithToEmitCPatterns(typeConverter, patterns);
    populateSCFToEmitCConversionPatterns(patterns, typeConverter);

    ConversionTarget target(*moduleOp.getContext());
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
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
    // A runtime scalar of type `index` (e.g. a loop trip count) has no C++
    // spelling on its own; emit it as emitc.size_t so it flows straight into a
    // converted scf.for bound with no leftover cast. Other scalar types pass
    // through unchanged.
    auto sizeT = emitc::SizeTType::get(moduleOp.getContext());
    auto paramTypeFor = [&](Type t) -> Type {
      return isa<IndexType>(t) ? Type(sizeT) : t;
    };
    for (BlockArgument arg : entry.getArguments())
      if (!isa<BaseMemRefType>(arg.getType()))
        paramTypes.push_back(paramTypeFor(arg.getType()));

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

    // Declare a runtime BD free-list pool for each tile that draws ids at
    // runtime (dynamic free-list path). The pool size is the tile's BD count
    // from the target model -- never hardcoded here. One decl per distinct
    // tile.
    const AIE::AIETargetModel &targetModel = deviceOp.getTargetModel();
    llvm::SmallSet<std::pair<uint32_t, uint32_t>, 4> pooledTiles;
    seqOp.walk([&](Operation *op) {
      uint32_t col, row;
      if (auto pop = dyn_cast<AIEX::DMABdPoolPopOp>(op)) {
        col = pop.getColumn();
        row = pop.getRow();
      } else if (auto push = dyn_cast<AIEX::DMABdPoolPushOp>(op)) {
        col = push.getColumn();
        row = push.getRow();
      } else {
        return;
      }
      if (!pooledTiles.insert({col, row}).second)
        return;
      uint32_t numBDs = targetModel.getNumBDs(col, row);
      emitc::VerbatimOp::create(fb, loc,
                                "aie_runtime::BdPool " + bdPoolName(col, row) +
                                    " = aie_runtime::bd_pool_init(" +
                                    std::to_string(numBDs) + ");");
    });

    // A rolled loop makes the op count a runtime quantity: declare a __opcount
    // accumulator the converter bumps per op. A straight-line sequence keeps
    // the compile-time literal count (byte-identical golden).
    bool hasControlFlow = false;
    seqOp.walk([&](Operation *op) {
      if (isa<scf::SCFDialect>(op->getDialect()))
        hasControlFlow = true;
    });
    Value opCountVar;
    if (hasControlFlow) {
      emitc::VerbatimOp::create(fb, loc, "uint32_t __opcount = 0;");
      opCountVar = emitc::LiteralOp::create(
          fb, loc, getU32Type(moduleOp.getContext()), "__opcount");
    }

    // Clone the sequence body (whole regions, so a rolled scf.for comes along),
    // mapping non-memref block args to function params. Resolve device-relative
    // values (address, blockwrite data) while the originals are still under the
    // device, keyed by clone, since the accessors don't work once detached.
    IRMapping mapping;
    unsigned p = 0;
    for (BlockArgument arg : entry.getArguments())
      if (!isa<BaseMemRefType>(arg.getType()))
        mapping.map(arg, funcBlock->getArgument(p++));
    DeviceResolved resolved;
    for (Operation &op : entry.without_terminator()) {
      fb.clone(op, mapping);
      // Record device-resolved values for the clone and any nested op (a BD
      // inside a rolled loop resolves the same way as a top-level one).
      op.walk([&](Operation *orig) {
        Operation *c = mapping.lookupOrNull(orig);
        if (c)
          recordDeviceResolved(orig, c, resolved);
      });
    }

    AIEXToEmitCConverter conv(funcOp, txnVec, resolved, opCountVar);
    std::optional<uint32_t> count = conv.run();
    if (!count)
      return failure();

    // Finalize the header with the op count + device info. The count is the
    // runtime `__opcount` when the sequence has a loop, else the compile-time
    // literal.
    const AIE::AIETargetModel &tm = deviceOp.getTargetModel();
    uint8_t devGen = isa<AIE::BaseNPU2TargetModel>(tm) ? 4 : 3;
    OpBuilder eb(funcBlock, funcBlock->end());
    std::string countStr =
        hasControlFlow ? "__opcount" : (std::to_string(*count) + "u");
    std::string header = "aie_runtime::txn_prepend_header(txn, " + countStr +
                         ", {0, 1, " + std::to_string(devGen) + ", " +
                         std::to_string(tm.rows()) + ", " +
                         std::to_string(tm.columns()) + ", " +
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
