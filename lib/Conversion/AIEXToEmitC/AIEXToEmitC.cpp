//===- AIEXToEmitC.cpp - AIEX to EmitC conversion ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
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
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
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
void emitTxnWrite32(OpBuilder &builder, Location loc, Value txnVec,
                           Value addr, Value val) {
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
void emitTxnSync(OpBuilder &builder, Location loc, Value txnVec,
                        Value col, Value row, Value dir, Value chan, Value ncol,
                        Value nrow) {
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
                                uint32_t addr, int32_t argIdx, int32_t argPlus,
                                Value dynArgPlus) {
  auto addrVal = createU32Constant(builder, loc, addr);
  auto idxVal = createU32Constant(builder, loc, static_cast<uint32_t>(argIdx));
  Value plusVal;
  if (dynArgPlus) {
    plusVal = dynArgPlus;
  } else {
    plusVal = createU32Constant(builder, loc, static_cast<uint32_t>(argPlus));
  }
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

// Emit: blockwrite with a runtime-provided first payload word.
// This is used to fold the common
//   blockwrite(payload_with_word0_placeholder)
//   address_patch(base+4, ...)
//   write32(base, dynamic_len)
// sequence back into the exact static blockwrite layout while keeping the
// first payload word parameterized.
void emitTxnBlockWriteDynamicFirstWord(OpBuilder &builder, Location loc,
                                              Value txnVec, uint32_t addr,
                                              DenseIntElementsAttr data,
                                              Value dynamicFirstWord) {
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
  auto zeroIndex = createU32Constant(builder, loc, 0);
  auto firstElem = emitc::SubscriptOp::create(
      builder, loc, cast<TypedValue<emitc::ArrayType>>(arrayVar.getResult()),
      ValueRange{zeroIndex});
  emitc::AssignOp::create(builder, loc, firstElem.getResult(),
                          castToU32(builder, loc, dynamicFirstWord));

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

    IRMapping mapping;
    unsigned paramIdx = 0;
    for (auto arg : entryBlock.getArguments()) {
      if (isa<MemRefType, UnrankedMemRefType>(arg.getType()))
        continue;
      mapping.map(arg, funcBlock->getArgument(paramIdx++));
    }

    if (failed(cloneExternalConstants(seqOp, funcBuilder, mapping)))
      return failure();

    if (failed(cloneBlock(entryBlock, funcBuilder, mapping, txnVec)))
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

  LogicalResult cloneBlock(Block &block, OpBuilder &builder, IRMapping &mapping,
                           Value txnVec) {
    for (auto it = block.begin(), e = block.end(); it != e; ++it) {
      Operation *op = &*it;

      auto blockWrite = dyn_cast<AIEX::NpuBlockWriteOp>(op);
      if (!blockWrite) {
        if (failed(cloneOp(builder, op, mapping, txnVec)))
          return failure();
        continue;
      }

      auto nextIt = std::next(it);
      if (nextIt == e) {
        if (failed(cloneOp(builder, op, mapping, txnVec)))
          return failure();
        continue;
      }

      auto addrPatch = dyn_cast<AIEX::NpuAddressPatchOp>(&*nextIt);
      if (!addrPatch) {
        if (failed(cloneOp(builder, op, mapping, txnVec)))
          return failure();
        continue;
      }

      auto nextNextIt = std::next(nextIt);
      if (nextNextIt == e) {
        if (failed(cloneOp(builder, op, mapping, txnVec)))
          return failure();
        continue;
      }

      auto write32 = dyn_cast<AIEX::NpuWrite32Op>(&*nextNextIt);
      if (!write32 || !write32.hasDynamicOperands()) {
        if (failed(cloneOp(builder, op, mapping, txnVec)))
          return failure();
        continue;
      }

      auto dynAddrConst =
          write32.getDynAddress().getDefiningOp<arith::ConstantOp>();
      auto dynAddrAttr = dynAddrConst
                             ? dyn_cast<IntegerAttr>(dynAddrConst.getValue())
                             : nullptr;

      uint32_t blockAddr = blockWrite.getAddress();
      if (auto absAddr = blockWrite.getAbsoluteAddress())
        blockAddr = *absAddr;
      if (!dynAddrAttr || dynAddrAttr.getValue().getZExtValue() != blockAddr) {
        if (failed(cloneOp(builder, op, mapping, txnVec)))
          return failure();
        continue;
      }
      if (addrPatch.getAddr() != blockAddr + 4) {
        if (failed(cloneOp(builder, op, mapping, txnVec)))
          return failure();
        continue;
      }

      auto data = blockWrite.getDataWords();
      if (!data) {
        return failure();
      }

      emitTxnBlockWriteDynamicFirstWord(
          builder, blockWrite.getLoc(), txnVec, blockAddr, data,
          mapping.lookupOrDefault(write32.getDynValue()));
      Value dynPlus = addrPatch.getDynArgPlus();
      if (dynPlus)
        dynPlus = mapping.lookupOrDefault(dynPlus);
      emitTxnAddressPatch(builder, addrPatch.getLoc(), txnVec,
                          addrPatch.getAddr(), addrPatch.getArgIdx(),
                          addrPatch.getArgPlus(), dynPlus);

      // The addrPatch (nextIt) and write32 (nextNextIt) are consumed by the
      // fusion — their data is folded into emitTxnBlockWriteDynamicFirstWord
      // and emitTxnAddressPatch. Setting it = nextNextIt lets the loop's
      // ++it advance past the write32.
      it = nextNextIt;
    }
    return success();
  }

  LogicalResult cloneScfFor(OpBuilder &builder, scf::ForOp forOp,
                            IRMapping &mapping, Value txnVec) {
    Location loc = forOp.getLoc();
    SmallVector<Value> initArgs;
    for (Value initArg : forOp.getInitArgs())
      initArgs.push_back(mapping.lookupOrDefault(initArg));

    auto newFor = scf::ForOp::create(
        builder, loc, mapping.lookupOrDefault(forOp.getLowerBound()),
        mapping.lookupOrDefault(forOp.getUpperBound()),
        mapping.lookupOrDefault(forOp.getStep()), initArgs);

    for (auto [oldResult, newResult] :
         llvm::zip(forOp.getResults(), newFor.getResults()))
      mapping.map(oldResult, newResult);

    Block &newBody = newFor.getRegion().front();
    newBody.getOperations().clear();
    IRMapping nestedMapping = mapping;
    nestedMapping.map(forOp.getInductionVar(), newFor.getInductionVar());
    for (auto [oldArg, newArg] :
         llvm::zip(forOp.getRegionIterArgs(), newFor.getRegionIterArgs()))
      nestedMapping.map(oldArg, newArg);

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&newBody);
    return cloneBlock(forOp.getRegion().front(), bodyBuilder, nestedMapping,
                      txnVec);
  }

  LogicalResult cloneScfIf(OpBuilder &builder, scf::IfOp ifOp,
                           IRMapping &mapping, Value txnVec) {
    Location loc = ifOp.getLoc();
    auto newIf = scf::IfOp::create(builder, loc, ifOp.getResultTypes(),
                                   mapping.lookupOrDefault(ifOp.getCondition()),
                                   !ifOp.getElseRegion().empty());

    for (auto [oldResult, newResult] :
         llvm::zip(ifOp.getResults(), newIf.getResults()))
      mapping.map(oldResult, newResult);

    Block &thenBlock = newIf.getThenRegion().front();
    thenBlock.getOperations().clear();
    OpBuilder thenBuilder = OpBuilder::atBlockBegin(&thenBlock);
    if (failed(cloneBlock(ifOp.getThenRegion().front(), thenBuilder, mapping,
                          txnVec)))
      return failure();

    if (!ifOp.getElseRegion().empty()) {
      Block &elseBlock = newIf.getElseRegion().front();
      elseBlock.getOperations().clear();
      OpBuilder elseBuilder = OpBuilder::atBlockBegin(&elseBlock);
      if (failed(cloneBlock(ifOp.getElseRegion().front(), elseBuilder, mapping,
                            txnVec)))
        return failure();
    }

    return success();
  }

  /// Clone a source operation into the generated function, converting only the
  /// AIEX TXN ops directly to EmitC.
  LogicalResult cloneOp(OpBuilder &builder, Operation *op, IRMapping &mapping,
                        Value txnVec) {
    Location opLoc = op->getLoc();

    // AIEX write32 - handles both static and dynamic forms.
    if (auto write32 = dyn_cast<AIEX::NpuWrite32Op>(op)) {
      Value addrVal, valVal;
      if (write32.hasDynamicOperands()) {
        addrVal = mapping.lookupOrDefault(write32.getDynAddress());
        valVal = mapping.lookupOrDefault(write32.getDynValue());
      } else {
        uint32_t addr = write32.getAddress();
        if (auto absAddr = write32.getAbsoluteAddress())
          addr = *absAddr;
        addrVal = createU32Constant(builder, opLoc, addr);
        valVal = createU32Constant(builder, opLoc, write32.getValue());
      }
      emitTxnWrite32(builder, opLoc, txnVec, addrVal, valVal);
      return success();
    }

    // AIEX maskwrite32 - handles both static and dynamic forms.
    if (auto maskWrite = dyn_cast<AIEX::NpuMaskWrite32Op>(op)) {
      Value addrVal, valVal, maskVal;
      if (maskWrite.hasDynamicOperands()) {
        addrVal = mapping.lookupOrDefault(maskWrite.getDynAddress());
        valVal = mapping.lookupOrDefault(maskWrite.getDynValue());
        maskVal = mapping.lookupOrDefault(maskWrite.getDynMask());
      } else {
        uint32_t addr = maskWrite.getAddress();
        if (auto absAddr = maskWrite.getAbsoluteAddress())
          addr = *absAddr;
        addrVal = createU32Constant(builder, opLoc, addr);
        valVal = createU32Constant(builder, opLoc, maskWrite.getValue());
        maskVal = createU32Constant(builder, opLoc, maskWrite.getMask());
      }
      emitTxnMaskWrite32(builder, opLoc, txnVec, addrVal, valVal, maskVal);
      return success();
    }

    // AIEX sync - handles both static and dynamic forms.
    if (auto syncOp = dyn_cast<AIEX::NpuSyncOp>(op)) {
      Value col, row, dir, chan, ncol, nrow;
      if (syncOp.hasDynamicOperands()) {
        col = mapping.lookupOrDefault(syncOp.getDynColumn());
        row = mapping.lookupOrDefault(syncOp.getDynRow());
        dir = mapping.lookupOrDefault(syncOp.getDynDirection());
        chan = mapping.lookupOrDefault(syncOp.getDynChannel());
        ncol = mapping.lookupOrDefault(syncOp.getDynColumnNum());
        nrow = mapping.lookupOrDefault(syncOp.getDynRowNum());
      } else {
        col = createU32Constant(builder, opLoc, syncOp.getColumn());
        row = createU32Constant(builder, opLoc, syncOp.getRow());
        dir = createU32Constant(builder, opLoc,
                                static_cast<uint32_t>(syncOp.getDirection()));
        chan = createU32Constant(builder, opLoc, syncOp.getChannel());
        ncol = createU32Constant(builder, opLoc, syncOp.getColumnNum());
        nrow = createU32Constant(builder, opLoc, syncOp.getRowNum());
      }
      emitTxnSync(builder, opLoc, txnVec, col, row, dir, chan, ncol, nrow);
      return success();
    }

    if (auto addrPatch = dyn_cast<AIEX::NpuAddressPatchOp>(op)) {
      Value dynPlus = addrPatch.getDynArgPlus();
      if (dynPlus)
        dynPlus = mapping.lookupOrDefault(dynPlus);
      emitTxnAddressPatch(builder, opLoc, txnVec, addrPatch.getAddr(),
                          addrPatch.getArgIdx(), addrPatch.getArgPlus(),
                          dynPlus);
      return success();
    }

    if (auto blockWrite = dyn_cast<AIEX::NpuBlockWriteOp>(op)) {
      uint32_t addr = blockWrite.getAddress();
      if (auto absAddr = blockWrite.getAbsoluteAddress())
        addr = *absAddr;
      auto data = blockWrite.getDataWords();
      if (!data)
        return failure();
      emitTxnBlockWrite(builder, opLoc, txnVec, addr, data);
      return success();
    }

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return cloneScfFor(builder, forOp, mapping, txnVec);
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return cloneScfIf(builder, ifOp, mapping, txnVec);
    }

    if (auto getGlobal = dyn_cast<memref::GetGlobalOp>(op)) {
      for (Operation *user : getGlobal->getUsers()) {
        if (!isa<AIEX::NpuBlockWriteOp>(user)) {
          return op->emitOpError(
              "unsupported memref.get_global use in TXN EmitC conversion");
        }
      }
      // Blockwrite lowering consumes the referenced data directly; cloning the
      // memref.get_global would leave a dangling reference once module-level
      // memref.global ops are erased.
      return success();
    }

    if (auto minOp = dyn_cast<arith::MinSIOp>(op)) {
      Value lhs = mapping.lookupOrDefault(minOp.getLhs());
      Value rhs = mapping.lookupOrDefault(minOp.getRhs());
      auto cmp = builder.create<arith::CmpIOp>(opLoc, arith::CmpIPredicate::slt,
                                               lhs, rhs);
      auto select =
          builder.create<arith::SelectOp>(opLoc, cmp.getResult(), lhs, rhs);
      mapping.map(minOp.getResult(), select.getResult());
      return success();
    }

    if (auto maxOp = dyn_cast<arith::MaxSIOp>(op)) {
      Value lhs = mapping.lookupOrDefault(maxOp.getLhs());
      Value rhs = mapping.lookupOrDefault(maxOp.getRhs());
      auto cmp = builder.create<arith::CmpIOp>(opLoc, arith::CmpIPredicate::sgt,
                                               lhs, rhs);
      auto select =
          builder.create<arith::SelectOp>(opLoc, cmp.getResult(), lhs, rhs);
      mapping.map(maxOp.getResult(), select.getResult());
      return success();
    }

    if (isa<AIE::EndOp>(op))
      return success();

    if (isa<AIEX::NpuControlPacketOp, AIEX::NpuPushQueueOp,
            AIEX::NpuWriteBdOp>(op))
      return op->emitOpError("not supported in dynamic TXN C++ generation");

    if (op->getNumRegions() != 0)
      return op->emitOpError(
          "unsupported region operation in TXN EmitC conversion");

    Operation *cloned = builder.clone(*op, mapping);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), cloned->getResults()))
      mapping.map(oldResult, newResult);
    return success();
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
