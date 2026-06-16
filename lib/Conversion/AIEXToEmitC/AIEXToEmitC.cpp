//===- AIEXToEmitC.cpp - AIEX to EmitC conversion ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass converts AIEX dynamic runtime sequence operations and static NPU
// ops into EmitC dialect operations. The generated EmitC IR calls functions
// from TxnEncoding.h, and MLIR's translateToCpp() produces compilable C++ code.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEXToEmitC/AIEXToEmitC.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"

namespace xilinx {
#define GEN_PASS_DEF_CONVERTAIEXTOEMITC
#include "aie/Conversion/Passes.h.inc"
} // namespace xilinx

using namespace mlir;
using namespace xilinx;

namespace {

// Shared helper: get the emitc opaque type for uint32_t
emitc::OpaqueType getU32Type(MLIRContext *ctx) {
  return emitc::OpaqueType::get(ctx, "uint32_t");
}

// Shared helper: create a uint32_t constant
Value createU32Constant(OpBuilder &builder, Location loc, uint32_t val) {
  auto u32Type = getU32Type(builder.getContext());
  return emitc::ConstantOp::create(
      builder, loc, u32Type,
      emitc::OpaqueAttr::get(builder.getContext(), std::to_string(val) + "u"));
}

// Shared helper: cast an SSA value to uint32_t using static_cast
Value castToU32(OpBuilder &builder, Location loc, Value val) {
  auto u32Type = getU32Type(builder.getContext());
  if (val.getType() == u32Type)
    return val;
  return emitc::CastOp::create(builder, loc, u32Type, val);
}

// Emit: op_count++
void emitIncrementOpCount(OpBuilder &builder, Location loc) {
  emitc::VerbatimOp::create(builder, loc, "op_count++;");
}

// Emit: aie_runtime::txn_append_write32(txn, addr, val)
void emitTxnWrite32(OpBuilder &builder, Location loc, Value txnVec, Value addr,
                    Value val) {
  auto u32 = castToU32(builder, loc, addr);
  auto u32v = castToU32(builder, loc, val);
  emitc::CallOpaqueOp::create(builder, loc, TypeRange{},
                              "aie_runtime::txn_append_write32",
                              ValueRange{txnVec, u32, u32v});
  emitIncrementOpCount(builder, loc);
}

// Emit: aie_runtime::txn_append_maskwrite32(txn, addr, val, mask)
void emitTxnMaskWrite32(OpBuilder &builder, Location loc, Value txnVec,
                        Value addr, Value val, Value mask) {
  auto u32a = castToU32(builder, loc, addr);
  auto u32v = castToU32(builder, loc, val);
  auto u32m = castToU32(builder, loc, mask);
  emitc::CallOpaqueOp::create(builder, loc, TypeRange{},
                              "aie_runtime::txn_append_maskwrite32",
                              ValueRange{txnVec, u32a, u32v, u32m});
  emitIncrementOpCount(builder, loc);
}

// Emit: aie_runtime::txn_append_sync(txn, col, row, dir, chan, ncol, nrow)
void emitTxnSync(OpBuilder &builder, Location loc, Value txnVec, Value col,
                 Value row, Value dir, Value chan, Value ncol, Value nrow) {
  auto u32col = castToU32(builder, loc, col);
  auto u32row = castToU32(builder, loc, row);
  auto u32dir = castToU32(builder, loc, dir);
  auto u32chan = castToU32(builder, loc, chan);
  auto u32ncol = castToU32(builder, loc, ncol);
  auto u32nrow = castToU32(builder, loc, nrow);
  emitc::CallOpaqueOp::create(
      builder, loc, TypeRange{}, "aie_runtime::txn_append_sync",
      ValueRange{txnVec, u32col, u32row, u32dir, u32chan, u32ncol, u32nrow});
  emitIncrementOpCount(builder, loc);
}

// Emit: aie_runtime::txn_append_address_patch(txn, addr, arg_idx, arg_plus)
void emitTxnAddressPatch(OpBuilder &builder, Location loc, Value txnVec,
                         uint32_t addr, int32_t argIdx, Value argPlus) {
  auto addrVal = createU32Constant(builder, loc, addr);
  auto idxVal = createU32Constant(builder, loc, static_cast<uint32_t>(argIdx));
  auto plusVal = castToU32(builder, loc, argPlus);
  emitc::CallOpaqueOp::create(builder, loc, TypeRange{},
                              "aie_runtime::txn_append_address_patch",
                              ValueRange{txnVec, addrVal, idxVal, plusVal});
  emitIncrementOpCount(builder, loc);
}

// Emit: aie_runtime::txn_append_blockwrite(txn, addr, data, count)
// For blockwrite, we emit the data as an inline array literal.
void emitTxnBlockWrite(OpBuilder &builder, Location loc, Value txnVec,
                       uint32_t addr, DenseIntElementsAttr data) {
  // Build inline array data string: "uint32_t data_N[] = {0x..., 0x..., ...};"
  std::string arrayStr = "{";
  llvm::raw_string_ostream ss(arrayStr);
  bool first = true;
  for (auto d : data) {
    if (!first)
      ss << ", ";
    uint32_t word = d.getZExtValue();
    ss << llvm::format("0x%08Xu", word);
    first = false;
  }
  ss << "}";

  // Emit blockwrite via VerbatimOp since arrays don't map cleanly to emitc.
  std::string stmt = "{\n  static const uint32_t _bd_data[] = " + arrayStr +
                     ";\n  aie_runtime::txn_append_blockwrite(txn, " +
                     std::to_string(addr) + "u, _bd_data, " +
                     std::to_string(data.size()) + ");\n}";
  emitc::VerbatimOp::create(builder, loc, stmt);
  emitIncrementOpCount(builder, loc);
}

// Emit: blockwrite with runtime-provided dynamic word overrides.
// Creates the static BD array, overrides specific word indices with dynamic
// SSA values, then calls txn_append_blockwrite. This generalizes the previous
// single-word override to support multiple dynamic BD words.
void emitTxnBlockWriteDynamicWords(
    OpBuilder &builder, Location loc, Value txnVec, uint32_t addr,
    DenseIntElementsAttr data,
    ArrayRef<std::pair<uint32_t, Value>> dynamicWords) {
  auto *ctx = builder.getContext();
  auto u32Type = getU32Type(ctx);
  auto arrayType = emitc::ArrayType::get(
      ctx, SmallVector<int64_t>{static_cast<int64_t>(data.size())}, u32Type);

  std::string arrayInit = "{";
  llvm::raw_string_ostream ss(arrayInit);
  bool first = true;
  for (auto d : data) {
    if (!first)
      ss << ", ";
    uint32_t word = d.getZExtValue();
    ss << llvm::format("0x%08Xu", word);
    first = false;
  }
  ss << "}";

  auto arrayVar = emitc::VariableOp::create(
      builder, loc, arrayType, emitc::OpaqueAttr::get(ctx, arrayInit));

  for (auto &[wordIdx, dynVal] : dynamicWords) {
    auto indexConst = createU32Constant(builder, loc, wordIdx);
    auto elem = emitc::SubscriptOp::create(
        builder, loc, cast<TypedValue<emitc::ArrayType>>(arrayVar.getResult()),
        ValueRange{indexConst});
    emitc::AssignOp::create(builder, loc, elem.getResult(),
                            castToU32(builder, loc, dynVal));
  }

  auto addrVal = createU32Constant(builder, loc, addr);
  auto countVal =
      createU32Constant(builder, loc, static_cast<uint32_t>(data.size()));
  emitc::CallOpaqueOp::create(
      builder, loc, TypeRange{}, "aie_runtime::txn_append_blockwrite",
      ValueRange{txnVec, addrVal, arrayVar.getResult(), countVal});
  emitIncrementOpCount(builder, loc);
}

/// The main pass that converts AIEX runtime sequence ops to C++-emittable IR.
/// AIEX TXN ops are lowered directly to EmitC calls, while the surrounding
/// arith/scf structure is cloned as regular MLIR and then lowered via upstream
/// convert-arith-to-emitc / convert-scf-to-emitc patterns.
struct ConvertAIEXToEmitCPass
    : xilinx::impl::ConvertAIEXToEmitCBase<ConvertAIEXToEmitCPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    // Collect all RuntimeSequenceOps and their parent DeviceOps.
    struct SeqInfo {
      AIE::RuntimeSequenceOp seq;
      AIE::DeviceOp device;
    };
    SmallVector<SeqInfo> sequences;
    moduleOp.walk([&](AIE::RuntimeSequenceOp seq) {
      auto device = seq->getParentOfType<AIE::DeviceOp>();
      sequences.push_back({seq, device});
    });

    if (sequences.empty())
      return;

    // We'll build a new module body with emitc ops.
    OpBuilder builder(ctx);

    for (auto &[seqOp, deviceOp] : sequences) {
      if (!deviceOp) {
        seqOp.emitOpError("must be nested inside an aie.device");
        signalPassFailure();
        return;
      }

      std::string seqName = seqOp.getSymName().str();

      if (failed(createGeneratedFunction(builder, moduleOp, seqOp, deviceOp,
                                         seqName))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<Operation *> toErase;
    for (auto &op : moduleOp.getBody()->getOperations()) {
      if (!isa<emitc::FuncOp>(op) && !isa<emitc::IncludeOp>(op))
        toErase.push_back(&op);
    }
    for (auto *op : llvm::reverse(toErase))
      op->erase();

    // Add the #include at the top of the module.
    builder.setInsertionPointToStart(moduleOp.getBody());
    auto includeOp = emitc::IncludeOp::create(builder, moduleOp.getLoc(),
                                              "aie/Runtime/TxnEncoding.h",
                                              /*is_standard=*/false);
    // Also include standard headers.
    builder.setInsertionPointAfter(includeOp);
    emitc::IncludeOp::create(builder, moduleOp.getLoc(), "cstdint",
                             /*is_standard=*/true);
    emitc::IncludeOp::create(builder, moduleOp.getLoc(), "cstddef",
                             /*is_standard=*/true);
    emitc::IncludeOp::create(builder, moduleOp.getLoc(), "vector",
                             /*is_standard=*/true);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    populateEmitCSizeTTypeConversions(typeConverter);

    RewritePatternSet patterns(ctx);
    // Expand arith.ceildivsi / arith.floordivsi into basic arith ops
    // (cmpi/select/divsi) that the upstream ArithToEmitC patterns know how
    // to lower. The Python front-end naturally produces these ops from
    // expressions like `M // m` on SSA i32 runtime-sequence arguments.
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    // Expand arith.minsi / maxsi / minui / maxui into cmpi + select, which the
    // upstream ArithToEmitC patterns can lower (EmitC has no min/max op). The
    // Python front-end produces these from runtime-sequence tail-tile clamps
    // like `arith.minsi(rows_per_block // 2, M_div_m - row_base)`.
    arith::populateArithExpandOpsPatterns(patterns);
    populateArithToEmitCPatterns(typeConverter, patterns);
    populateSCFToEmitCConversionPatterns(patterns, typeConverter);

    ConversionTarget target(*ctx);
    target.addLegalDialect<emitc::EmitCDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();
    target.addIllegalDialect<arith::ArithDialect, scf::SCFDialect>();

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    if (failed(lowerRemainingUnrealizedCasts(moduleOp))) {
      signalPassFailure();
      return;
    }
  }

private:
  // Device-relative values resolved on the original in-device ops, keyed by
  // their clone in the emitc.func. getAbsoluteAddress() and getDataWords()
  // resolve buffer/column/row and memref.global symbols against the parent
  // aie.device, so they must be computed before the ops are cloned out of it.
  struct DeviceResolved {
    // Absolute address for write32/maskwrite32/blockwrite clones whose address
    // resolves through buffer/column/row (absent when the address is a plain
    // SSA value to pass through).
    llvm::DenseMap<Operation *, uint32_t> absoluteAddr;
    // BD data words for blockwrite clones.
    llvm::DenseMap<Operation *, DenseIntElementsAttr> blockWriteData;
  };

  // Resolve the device-dependent values of `orig` (still under the device) and
  // store them keyed by `clone`.
  void recordDeviceResolved(Operation *orig, Operation *clone,
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

  // Walk `orig` and its structurally-identical `clone` subtree in lockstep,
  // resolving every AIEX op (including those nested in scf regions). builder
  // .clone produces an isomorphic op/region/block structure, so a parallel
  // pre-order walk pairs each original op with its clone.
  void recordDeviceResolvedRecursive(Operation *orig, Operation *clone,
                                     DeviceResolved &resolved) {
    recordDeviceResolved(orig, clone, resolved);
    for (auto [origRegion, cloneRegion] :
         llvm::zip(orig->getRegions(), clone->getRegions()))
      for (auto [origBlock, cloneBlock] :
           llvm::zip(origRegion.getBlocks(), cloneRegion.getBlocks()))
        for (auto [origOp, cloneOp] :
             llvm::zip(origBlock.getOperations(), cloneBlock.getOperations()))
          recordDeviceResolvedRecursive(&origOp, &cloneOp, resolved);
  }

  LogicalResult createGeneratedFunction(OpBuilder &builder, ModuleOp moduleOp,
                                        AIE::RuntimeSequenceOp seqOp,
                                        AIE::DeviceOp deviceOp,
                                        StringRef seqName) {
    Location loc = seqOp.getLoc();
    Block &entryBlock = seqOp.getBody().front();

    SmallVector<Type> paramTypes;
    for (auto arg : entryBlock.getArguments()) {
      if (!isa<MemRefType, UnrankedMemRefType>(arg.getType()))
        paramTypes.push_back(arg.getType());
    }

    auto txnVecType =
        emitc::OpaqueType::get(moduleOp.getContext(), "std::vector<uint32_t>");
    auto funcType = FunctionType::get(moduleOp.getContext(), paramTypes,
                                      TypeRange{txnVecType});

    builder.setInsertionPointToEnd(moduleOp.getBody());
    std::string funcName = "generate_txn_" + seqName.str();
    auto funcOp = emitc::FuncOp::create(builder, loc, funcName, funcType);
    funcOp.setSpecifiersAttr(builder.getStrArrayAttr({"inline"}));
    Block *funcBlock = funcOp.addEntryBlock();
    OpBuilder funcBuilder = OpBuilder::atBlockBegin(funcBlock);

    emitc::VerbatimOp::create(funcBuilder, loc, "std::vector<uint32_t> txn;");
    emitc::VerbatimOp::create(funcBuilder, loc, "aie_runtime::txn_init(txn);");
    emitc::VerbatimOp::create(funcBuilder, loc, "uint32_t op_count = 0;");
    Value txnVec =
        emitc::LiteralOp::create(funcBuilder, loc, txnVecType, "txn");

    // Clone the runtime-sequence body verbatim into the function. builder.clone
    // copies nested regions (scf.for/scf.if bodies, their yields and iter-args)
    // and remaps all SSA values through `mapping`, so we never reconstruct
    // control flow by hand. Memref block arguments are dropped (the NPU
    // lowering already replaced DMA ops referencing them); they have no
    // remaining uses at this point.
    IRMapping mapping;
    unsigned paramIdx = 0;
    for (auto arg : entryBlock.getArguments()) {
      if (isa<MemRefType, UnrankedMemRefType>(arg.getType()))
        continue;
      mapping.map(arg, funcBlock->getArgument(paramIdx++));
    }

    // Ops in the sequence may reference arith.constants defined outside it
    // (e.g. hoisted to the device). Clone those into the function first so the
    // verbatim body clone below has every operand available in `mapping`.
    if (failed(cloneExternalConstants(seqOp, funcBuilder, mapping)))
      return failure();

    // Device-relative resolution (absolute addresses, blockwrite data) must
    // happen while the ops still live under the aie.device. Clone each
    // top-level op (which deep-clones its regions), then walk the original and
    // cloned subtrees in lockstep to resolve every AIEX op -- including those
    // nested in scf.for/scf.if -- keyed by its clone. convertTxnOps then reads
    // from these maps instead of calling device-dependent accessors on the
    // (detached) cloned ops.
    DeviceResolved resolved;
    for (Operation &op : entryBlock.without_terminator()) {
      Operation *clone = funcBuilder.clone(op, mapping);
      recordDeviceResolvedRecursive(&op, clone, resolved);
    }

    // Convert the cloned AIEX txn ops (and fuse BD blockwrites) in place,
    // appending to `txn`. arith/scf left behind are lowered afterwards by the
    // upstream conversions.
    if (failed(convertTxnOps(funcOp, txnVec, resolved)))
      return failure();

    const auto &tm = deviceOp.getTargetModel();
    uint8_t devGen = llvm::isa<AIE::BaseNPU2TargetModel>(tm) ? 4 : 3;
    emitc::VerbatimOp::create(
        funcBuilder, loc,
        "aie_runtime::txn_prepend_header(txn, op_count, {0, 1, " +
            std::to_string(devGen) + ", " + std::to_string(tm.rows()) + ", " +
            std::to_string(tm.columns()) + ", " +
            std::to_string(tm.getNumMemTileRows()) + "});");
    emitc::ReturnOp::create(funcBuilder, loc, txnVec);
    return success();
  }

  LogicalResult cloneExternalConstants(AIE::RuntimeSequenceOp seqOp,
                                       OpBuilder &builder, IRMapping &mapping) {
    llvm::DenseSet<Operation *> clonedConstants;
    WalkResult prescan = seqOp.walk([&](Operation *innerOp) -> WalkResult {
      for (Value operand : innerOp->getOperands()) {
        if (mapping.contains(operand))
          continue;
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || seqOp->isAncestor(defOp))
          continue;
        auto constOp = dyn_cast<arith::ConstantOp>(defOp);
        if (!constOp) {
          innerOp->emitOpError(
              "uses an external value that is not an arith.constant");
          return WalkResult::interrupt();
        }
        if (clonedConstants.insert(defOp).second) {
          Operation *newConst = builder.clone(*defOp, mapping);
          mapping.map(constOp.getResult(), newConst->getResult(0));
        }
      }
      return WalkResult::advance();
    });
    return prescan.wasInterrupted() ? failure() : success();
  }

  // Convert the AIEX txn ops cloned into `funcOp` into aie_runtime emitc
  // calls, in place. The runtime-sequence body (including scf.for/scf.if
  // regions) was already cloned verbatim, so a post-order walk reaches txn ops
  // at any nesting depth without reconstructing control flow by hand. BD
  // blockwrite fusion is handled per-block before the generic walk.
  LogicalResult convertTxnOps(emitc::FuncOp funcOp, Value txnVec,
                              const DeviceResolved &resolved) {
    // First fuse BD blockwrites within each block. Collect blocks up front so
    // the in-place erasis during fusion don't perturb iteration.
    SmallVector<Block *> blocks;
    funcOp.walk([&](Block *block) { blocks.push_back(block); });
    for (Block *block : blocks)
      if (failed(fuseBlockWrites(*block, txnVec, resolved)))
        return failure();

    // Then convert the remaining standalone AIEX txn ops. Gather first; the
    // conversion erases ops, which a live walk dislikes. memref.get_global ops
    // are handled last: a blockwrite reads its data through get_global, so the
    // blockwrites must be converted before their get_global is dropped.
    SmallVector<Operation *> txnOps;
    SmallVector<Operation *> getGlobals;
    funcOp.walk([&](Operation *op) {
      if (isa<memref::GetGlobalOp>(op))
        getGlobals.push_back(op);
      else if (isa<AIEX::NpuWrite32Op, AIEX::NpuMaskWrite32Op, AIEX::NpuSyncOp,
                   AIEX::NpuAddressPatchOp, AIEX::NpuBlockWriteOp, AIE::EndOp>(
                   op))
        txnOps.push_back(op);
    });
    for (Operation *op : txnOps)
      if (failed(convertOneTxnOp(op, txnVec, resolved)))
        return failure();
    for (Operation *op : getGlobals)
      if (failed(convertOneTxnOp(op, txnVec, resolved)))
        return failure();

    // Any AIEX op left behind (e.g. npu.push_queue, npu.writebd,
    // npu.control_packet) has no C++ TXN representation.
    WalkResult leftover = funcOp.walk([&](Operation *op) {
      if (isa<AIEX::AIEXDialect>(op->getDialect())) {
        op->emitOpError("not supported in dynamic TXN C++ generation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (leftover.wasInterrupted())
      return failure();
    return success();
  }

  // Fuse, within a single block, each npu.blockwrite together with the
  // bd_group-tagged npu.write32 word overrides and the matching
  // npu.address_patch into a single txn_append_blockwrite plus
  // txn_append_address_patch. The fused emitc calls are inserted at the
  // blockwrite and the consumed ops are erased. write32s are matched by their
  // bd_group attribute (robust to reordering), the address_patch by its
  // address (blockAddr + 4).
  LogicalResult fuseBlockWrites(Block &block, Value txnVec,
                                const DeviceResolved &resolved) {
    SmallVector<AIEX::NpuBlockWriteOp> blockWrites;
    for (Operation &op : block)
      if (auto bw = dyn_cast<AIEX::NpuBlockWriteOp>(op))
        blockWrites.push_back(bw);

    for (AIEX::NpuBlockWriteOp blockWrite : blockWrites) {
      uint32_t blockAddr = blockWrite.getAddress();
      if (auto it = resolved.absoluteAddr.find(blockWrite);
          it != resolved.absoluteAddr.end())
        blockAddr = it->second;

      SmallVector<std::pair<uint32_t, AIEX::NpuWrite32Op>> dynWrite32s;
      AIEX::NpuAddressPatchOp matchedPatch = nullptr;
      for (Operation &op : block) {
        if (auto patch = dyn_cast<AIEX::NpuAddressPatchOp>(op)) {
          if (patch.getAddr() == blockAddr + 4) {
            matchedPatch = patch;
            break;
          }
          continue;
        }
        if (auto w32 = dyn_cast<AIEX::NpuWrite32Op>(op)) {
          auto bdGroupAttr = w32.getBdGroupAttr();
          if (bdGroupAttr &&
              bdGroupAttr.getValue().getZExtValue() == blockAddr) {
            auto addrConst =
                w32.getAddress().getDefiningOp<arith::ConstantOp>();
            auto addrAttr = addrConst
                                ? dyn_cast<IntegerAttr>(addrConst.getValue())
                                : nullptr;
            if (addrAttr) {
              uint64_t w32Addr = addrAttr.getValue().getZExtValue();
              uint32_t wordIdx =
                  static_cast<uint32_t>((w32Addr - blockAddr) / 4);
              dynWrite32s.push_back({wordIdx, w32});
            }
          }
        }
      }

      // No dynamic overrides: leave the blockwrite for the generic conversion.
      if (!matchedPatch || dynWrite32s.empty())
        continue;

      auto dataIt = resolved.blockWriteData.find(blockWrite);
      if (dataIt == resolved.blockWriteData.end())
        return failure();
      DenseIntElementsAttr data = dataIt->second;

      SmallVector<std::pair<uint32_t, Value>> dynamicWords;
      for (auto &[wordIdx, w32] : dynWrite32s)
        dynamicWords.push_back({wordIdx, w32.getValue()});

      // Insert the fused calls at the address_patch, which follows the
      // blockwrite and all the dynamic-word value definitions in program order.
      // Building here keeps every referenced SSA value dominating its use (the
      // override values are defined between the blockwrite and the patch).
      OpBuilder builder(matchedPatch);
      emitTxnBlockWriteDynamicWords(builder, blockWrite.getLoc(), txnVec,
                                    blockAddr, data, dynamicWords);
      emitTxnAddressPatch(builder, matchedPatch.getLoc(), txnVec,
                          matchedPatch.getAddr(), matchedPatch.getArgIdx(),
                          matchedPatch.getArgPlus());

      matchedPatch.erase();
      for (auto &[wordIdx, w32] : dynWrite32s)
        w32.erase();
      blockWrite.erase();
    }
    return success();
  }

  // Convert one already-cloned AIEX txn op into emitc aie_runtime calls,
  // inserted in place, then erase the op. Operands are the cloned SSA values,
  // so they are used directly (no remapping). scf/arith are left untouched for
  // the upstream conversions; only AIEX txn ops are handled here.
  LogicalResult convertOneTxnOp(Operation *op, Value txnVec,
                                const DeviceResolved &resolved) {
    Location opLoc = op->getLoc();
    OpBuilder builder(op);

    if (auto write32 = dyn_cast<AIEX::NpuWrite32Op>(op)) {
      // A compile-time-constant address resolvable through buffer/column/row is
      // emitted as the absolute address (resolved before cloning); otherwise
      // the SSA address operand is used directly.
      Value addrVal;
      if (auto it = resolved.absoluteAddr.find(op);
          it != resolved.absoluteAddr.end())
        addrVal = createU32Constant(builder, opLoc, it->second);
      else
        addrVal = write32.getAddress();
      emitTxnWrite32(builder, opLoc, txnVec, addrVal, write32.getValue());
      op->erase();
      return success();
    }

    if (auto maskWrite = dyn_cast<AIEX::NpuMaskWrite32Op>(op)) {
      Value addrVal;
      if (auto it = resolved.absoluteAddr.find(op);
          it != resolved.absoluteAddr.end())
        addrVal = createU32Constant(builder, opLoc, it->second);
      else
        addrVal = maskWrite.getAddress();
      emitTxnMaskWrite32(builder, opLoc, txnVec, addrVal, maskWrite.getValue(),
                         maskWrite.getMask());
      op->erase();
      return success();
    }

    if (auto syncOp = dyn_cast<AIEX::NpuSyncOp>(op)) {
      emitTxnSync(builder, opLoc, txnVec, syncOp.getColumn(), syncOp.getRow(),
                  syncOp.getDirection(), syncOp.getChannel(),
                  syncOp.getColumnNum(), syncOp.getRowNum());
      op->erase();
      return success();
    }

    if (auto addrPatch = dyn_cast<AIEX::NpuAddressPatchOp>(op)) {
      emitTxnAddressPatch(builder, opLoc, txnVec, addrPatch.getAddr(),
                          addrPatch.getArgIdx(), addrPatch.getArgPlus());
      op->erase();
      return success();
    }

    if (auto blockWrite = dyn_cast<AIEX::NpuBlockWriteOp>(op)) {
      uint32_t addr = blockWrite.getAddress();
      if (auto it = resolved.absoluteAddr.find(op);
          it != resolved.absoluteAddr.end())
        addr = it->second;
      auto dataIt = resolved.blockWriteData.find(op);
      if (dataIt == resolved.blockWriteData.end())
        return failure();
      emitTxnBlockWrite(builder, opLoc, txnVec, addr, dataIt->second);
      op->erase();
      return success();
    }

    if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(op)) {
      for (Operation *user : getGlobal->getUsers()) {
        if (!isa<AIEX::NpuBlockWriteOp>(user))
          return op->emitOpError(
              "unsupported memref.get_global use in TXN EmitC conversion");
      }
      // Blockwrite lowering inlines the referenced data; the get_global has no
      // remaining users once blockwrites are converted, so drop it (the
      // module-level memref.global is erased afterwards).
      op->erase();
      return success();
    }

    if (isa<AIE::EndOp>(op)) {
      op->erase();
      return success();
    }

    return op->emitOpError("unsupported op in TXN EmitC conversion");
  }

  LogicalResult lowerRemainingUnrealizedCasts(ModuleOp moduleOp) {
    SmallVector<UnrealizedConversionCastOp> initialCasts;
    moduleOp.walk(
        [&](UnrealizedConversionCastOp cast) { initialCasts.push_back(cast); });
    reconcileUnrealizedCasts(initialCasts);

    SmallVector<UnrealizedConversionCastOp> remainingCasts;
    moduleOp.walk([&](UnrealizedConversionCastOp cast) {
      remainingCasts.push_back(cast);
    });

    for (auto cast : remainingCasts) {
      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
        return cast->emitOpError("unsupported unrealized conversion cast arity "
                                 "after EmitC conversion");

      Value input = cast.getOperands().front();
      Type srcType = input.getType();
      Type dstType = cast.getResult(0).getType();
      if (srcType == dstType) {
        cast.getResult(0).replaceAllUsesWith(input);
        cast.erase();
        continue;
      }

      OpBuilder builder(cast);
      Value replacement;
      auto ptrDiffTy = emitc::PtrDiffTType::get(moduleOp.getContext());

      if (srcType.isIndex() && isa<emitc::SizeTType>(dstType)) {
        Value signedSize =
            emitc::CastOp::create(builder, cast.getLoc(), ptrDiffTy, input);
        replacement =
            emitc::CastOp::create(builder, cast.getLoc(), dstType, signedSize);
      } else if (isa<emitc::SizeTType>(srcType) && dstType.isIndex()) {
        Value signedSize =
            emitc::CastOp::create(builder, cast.getLoc(), ptrDiffTy, input);
        replacement =
            emitc::CastOp::create(builder, cast.getLoc(), dstType, signedSize);
      } else {
        return cast->emitOpError(
            "unsupported unrealized conversion cast after EmitC conversion");
      }

      cast.getResult(0).replaceAllUsesWith(replacement);
      cast.erase();
    }

    SmallVector<UnrealizedConversionCastOp> finalCasts;
    moduleOp.walk(
        [&](UnrealizedConversionCastOp cast) { finalCasts.push_back(cast); });
    if (!finalCasts.empty()) {
      finalCasts.front()->emitOpError("unresolved unrealized conversion casts "
                                      "remain after EmitC conversion");
      return failure();
    }
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::createConvertAIEXToEmitCPass() {
  return std::make_unique<ConvertAIEXToEmitCPass>();
}
