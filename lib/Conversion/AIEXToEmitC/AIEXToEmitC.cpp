//===- AIEXToEmitC.cpp - AIEX to EmitC conversion ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

namespace xilinx {
#define GEN_PASS_DEF_CONVERTAIEXTOEMITC
#include "aie/Conversion/Passes.h.inc"
} // namespace xilinx

using namespace mlir;
using namespace xilinx;

namespace {

// Shared helper: get the emitc opaque type for uint32_t
static emitc::OpaqueType getU32Type(MLIRContext *ctx) {
  return emitc::OpaqueType::get(ctx, "uint32_t");
}

// Shared helper: create a uint32_t constant
static Value createU32Constant(OpBuilder &builder, Location loc, uint32_t val) {
  auto u32Type = getU32Type(builder.getContext());
  return emitc::ConstantOp::create(
      builder, loc, u32Type,
      emitc::OpaqueAttr::get(builder.getContext(), std::to_string(val) + "u"));
}

// Shared helper: get the emitc opaque type for int32_t
static emitc::OpaqueType getI32Type(MLIRContext *ctx) {
  return emitc::OpaqueType::get(ctx, "int32_t");
}

// Shared helper: cast an SSA value to int32_t for signed operations
static Value castToI32(OpBuilder &builder, Location loc, Value val) {
  auto i32Type = getI32Type(builder.getContext());
  if (val.getType() == i32Type)
    return val;
  return emitc::CastOp::create(builder, loc, i32Type, val);
}

// Shared helper: cast an SSA value to uint32_t using static_cast
static Value castToU32(OpBuilder &builder, Location loc, Value val) {
  auto u32Type = getU32Type(builder.getContext());
  if (val.getType() == u32Type)
    return val;
  return emitc::CastOp::create(builder, loc, u32Type, val);
}

// Emit: op_count++
static void emitIncrementOpCount(OpBuilder &builder, Location loc,
                                 Value /*opCountLval*/) {
  emitc::VerbatimOp::create(builder, loc, "op_count++;");
}

// Emit: aie_runtime::txn_append_write32(txn, addr, val)
static void emitTxnWrite32(OpBuilder &builder, Location loc, Value txnVec,
                           Value addr, Value val, Value opCountLval) {
  auto u32 = castToU32(builder, loc, addr);
  auto u32v = castToU32(builder, loc, val);
  emitc::CallOpaqueOp::create(builder, loc, TypeRange{},
                              "aie_runtime::txn_append_write32",
                              ValueRange{txnVec, u32, u32v});
  emitIncrementOpCount(builder, loc, opCountLval);
}

// Emit: aie_runtime::txn_append_maskwrite32(txn, addr, val, mask)
static void emitTxnMaskWrite32(OpBuilder &builder, Location loc, Value txnVec,
                               Value addr, Value val, Value mask,
                               Value opCountLval) {
  auto u32a = castToU32(builder, loc, addr);
  auto u32v = castToU32(builder, loc, val);
  auto u32m = castToU32(builder, loc, mask);
  emitc::CallOpaqueOp::create(builder, loc, TypeRange{},
                              "aie_runtime::txn_append_maskwrite32",
                              ValueRange{txnVec, u32a, u32v, u32m});
  emitIncrementOpCount(builder, loc, opCountLval);
}

// Emit: aie_runtime::txn_append_sync(txn, col, row, dir, chan, ncol, nrow)
static void emitTxnSync(OpBuilder &builder, Location loc, Value txnVec,
                        Value col, Value row, Value dir, Value chan, Value ncol,
                        Value nrow, Value opCountLval) {
  auto u32col = castToU32(builder, loc, col);
  auto u32row = castToU32(builder, loc, row);
  auto u32dir = castToU32(builder, loc, dir);
  auto u32chan = castToU32(builder, loc, chan);
  auto u32ncol = castToU32(builder, loc, ncol);
  auto u32nrow = castToU32(builder, loc, nrow);
  emitc::CallOpaqueOp::create(
      builder, loc, TypeRange{}, "aie_runtime::txn_append_sync",
      ValueRange{txnVec, u32col, u32row, u32dir, u32chan, u32ncol, u32nrow});
  emitIncrementOpCount(builder, loc, opCountLval);
}

// Emit: aie_runtime::txn_append_address_patch(txn, addr, arg_idx, arg_plus)
static void emitTxnAddressPatch(OpBuilder &builder, Location loc, Value txnVec,
                                uint32_t addr, int32_t argIdx, int32_t argPlus,
                                Value dynArgPlus, Value opCountLval) {
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
  emitIncrementOpCount(builder, loc, opCountLval);
}

// Emit: aie_runtime::txn_append_blockwrite(txn, addr, data, count)
// For blockwrite, we emit the data as an inline array literal.
static void emitTxnBlockWrite(OpBuilder &builder, Location loc, Value txnVec,
                              uint32_t addr, DenseIntElementsAttr data,
                              Value opCountLval) {
  // Build inline array data string: "uint32_t data_N[] = {0x..., 0x..., ...};"
  std::string arrayStr = "{";
  bool first = true;
  for (auto d : data) {
    if (!first)
      arrayStr += ", ";
    uint32_t word = d.getZExtValue();
    llvm::raw_string_ostream ss(arrayStr);
    ss << llvm::format("0x%08Xu", word);
    first = false;
  }
  arrayStr += "}";

  // Emit blockwrite via VerbatimOp since arrays don't map cleanly to emitc.
  std::string stmt = "{\n  static const uint32_t _bd_data[] = " + arrayStr +
                     ";\n  aie_runtime::txn_append_blockwrite(txn, " +
                     std::to_string(addr) + "u, _bd_data, " +
                     std::to_string(data.size()) + ");\n}";
  emitc::VerbatimOp::create(builder, loc, stmt);
  emitIncrementOpCount(builder, loc, opCountLval);
}

/// The main pass that converts AIEX runtime sequence ops to EmitC.
/// Rather than using the MLIR conversion framework (which requires type
/// conversion and legality checks for all ops), we do a direct IR walk and
/// build EmitC ops into a new emitc.func, since the source and target are
/// structurally different (RuntimeSequenceOp → emitc.func with TXN calls).
struct ConvertAIEXToEmitCPass
    : xilinx::impl::ConvertAIEXToEmitCBase<ConvertAIEXToEmitCPass> {

  // Stack of yield target variables. When processing an scf.for/scf.if with
  // results, the parent pushes mutable variable Values here, and the
  // scf.yield handler assigns into them.
  SmallVector<SmallVector<Value>> yieldTargetStack;
  // Current yield targets (top of stack, or empty).
  ArrayRef<Value> yieldTargets;

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

    if (sequences.empty()) {
      moduleOp.emitError("No runtime sequences found");
      return signalPassFailure();
    }

    // We'll build a new module body with emitc ops.
    OpBuilder builder(ctx);

    for (auto &[seqOp, deviceOp] : sequences) {
      Location loc = seqOp.getLoc();
      std::string seqName = seqOp.getSymName().str();

      // Extract device info for TXN header before we erase the device.
      uint8_t devGen = 3;
      uint8_t numRows = 6;
      uint8_t numCols = 5;
      uint8_t numMemTileRows = 1;
      if (deviceOp) {
        const auto &tm = deviceOp.getTargetModel();
        numRows = tm.rows();
        numCols = tm.columns();
        numMemTileRows = tm.getNumMemTileRows();
        if (llvm::isa<AIE::BaseNPU2TargetModel>(tm))
          devGen = 4;
      }

      // Determine function parameters from the runtime sequence block args.
      Block &entryBlock = seqOp.getBody().front();
      SmallVector<Type> paramTypes;
      SmallVector<std::string> paramNames;

      // First arg is always the txn vector reference (injected).
      // Remaining args come from the runtime sequence.
      for (auto arg : entryBlock.getArguments()) {
        Type origType = arg.getType();
        // Skip memref args (buffer references used for address patching).
        if (isa<MemRefType, UnrankedMemRefType>(origType))
          continue;
        // Map index/integer types to appropriate C++ types.
        Type emitcType;
        if (origType.isIndex()) {
          emitcType = emitc::OpaqueType::get(ctx, "size_t");
        } else if (auto intType = dyn_cast<IntegerType>(origType)) {
          if (intType.getWidth() <= 32)
            emitcType = getU32Type(ctx);
          else
            emitcType = emitc::OpaqueType::get(ctx, "uint64_t");
        } else {
          emitcType = emitc::OpaqueType::get(ctx, "auto");
        }
        paramTypes.push_back(emitcType);
        paramNames.push_back("arg" + std::to_string(arg.getArgNumber()));
      }

      // Build the emitc.func.
      // Signature: std::vector<uint32_t> generate_txn_<name>(<params...>)
      auto txnVecType = emitc::OpaqueType::get(ctx, "std::vector<uint32_t>");
      auto funcType = FunctionType::get(ctx, paramTypes, {txnVecType});

      builder.setInsertionPointToEnd(moduleOp.getBody());
      auto funcOp = emitc::FuncOp::create(builder, loc,
                                          "generate_txn_" + seqName, funcType);
      funcOp.setSpecifiersAttr(builder.getArrayAttr({}));

      // Create function body.
      Block *funcBlock = funcOp.addEntryBlock();
      builder.setInsertionPointToStart(funcBlock);

      // Emit: #include "aie/Runtime/TxnEncoding.h" as part of the module
      // (will be handled by the translation, not in the function body).

      // Emit: std::vector<uint32_t> txn;
      // Use VerbatimOp for the txn variable so blockwrite VerbatimOps can
      // reference it by the known name "txn".
      emitc::VerbatimOp::create(builder, loc, "std::vector<uint32_t> txn;");
      // Create an opaque literal referring to the named variable for use in
      // emitc::CallOpaqueOp calls.
      Value txnVec = emitc::LiteralOp::create(builder, loc, txnVecType, "txn");

      // Emit: uint32_t op_count = 0;
      // Tracked purely via VerbatimOp - no SSA value needed since we only
      // increment it (via VerbatimOp) and use it in the footer (via
      // VerbatimOp).
      emitc::VerbatimOp::create(builder, loc, "uint32_t op_count = 0;");
      // Dummy value passed to helpers (unused since emitIncrementOpCount
      // emits VerbatimOp directly).
      Value opCountLval = txnVec; // placeholder, not used

      // Map runtime sequence block args to function params.
      IRMapping argMapping;
      unsigned paramIdx = 0;
      for (auto arg : entryBlock.getArguments()) {
        if (isa<MemRefType, UnrankedMemRefType>(arg.getType()))
          continue;
        argMapping.map(arg, funcBlock->getArgument(paramIdx++));
      }

      // Pre-scan: find values used inside the runtime_sequence but defined
      // outside it (e.g., constants hoisted to the device region by
      // canonicalization). Create EmitC equivalents for them.
      seqOp.walk([&](Operation *innerOp) {
        for (Value operand : innerOp->getOperands()) {
          if (argMapping.contains(operand))
            continue;
          Operation *defOp = operand.getDefiningOp();
          if (!defOp)
            continue;
          // Check if defOp is outside the runtime_sequence.
          if (!seqOp->isAncestor(defOp)) {
            // Create EmitC equivalent for this external value.
            if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
              auto emitResult = emitOp(builder, constOp, loc, txnVec,
                                       opCountLval, argMapping);
              (void)emitResult;
            }
          }
        }
      });

      // Walk the runtime sequence body and emit corresponding emitc ops.
      auto result = emitOpsForBlock(builder, entryBlock, loc, txnVec,
                                    opCountLval, argMapping);
      if (failed(result)) {
        signalPassFailure();
        return;
      }

      // Emit: txn_prepend_header with device-specific info, then return.
      emitc::VerbatimOp::create(
          builder, loc,
          "aie_runtime::txn_prepend_header(txn, op_count, {0, 1, " +
              std::to_string(devGen) + ", " + std::to_string(numRows) + ", " +
              std::to_string(numCols) + ", " + std::to_string(numMemTileRows) +
              "});");

      // Emit: return txn;
      emitc::ReturnOp::create(builder, loc, txnVec);
    }

    // Remove the original runtime sequences and their parent device ops.
    // We keep the emitc.func ops and remove everything else.
    // Note: translateToCpp only emits emitc ops inside emitc.func, so
    // leftover aie.device/runtime_sequence ops are harmless. However,
    // leaving them causes verifier issues, so we need to erase them.
    //
    // To safely erase, we must ensure no cross-references from EmitC ops
    // back to old Values. Walk all EmitC funcs and clone any operands
    // that point to ops defined outside the func (i.e., in the old IR).
    // Fix cross-references: EmitC ops may reference Values defined in the
    // old IR (e.g., arith.constant values that weren't mapped through
    // argMapping). For each such reference, create an EmitC constant
    // with the same value directly in the EmitC function.
    for (auto funcOp : moduleOp.getOps<emitc::FuncOp>()) {
      // Cache: old Value -> new EmitC Value, to avoid duplicates.
      DenseMap<Value, Value> fixupCache;
      funcOp.walk([&](Operation *emitcOp) {
        for (auto &operand : emitcOp->getOpOperands()) {
          Value val = operand.get();
          if (!val)
            continue;
          Operation *defOp = val.getDefiningOp();
          if (!defOp)
            continue;
          if (defOp->getParentOfType<emitc::FuncOp>())
            continue;
          // Value is defined outside EmitC. Replace with EmitC equivalent.
          auto it = fixupCache.find(val);
          if (it != fixupCache.end()) {
            operand.set(it->second);
            continue;
          }
          OpBuilder localBuilder(emitcOp);
          Value replacement;
          if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
              auto *ctx = localBuilder.getContext();
              int64_t intVal = intAttr.getInt();
              Type origType = constOp.getType();
              Type emitcType;
              if (origType.isIndex())
                emitcType = emitc::OpaqueType::get(ctx, "size_t");
              else if (auto iType = dyn_cast<IntegerType>(origType))
                emitcType = iType.getWidth() <= 32
                                ? emitc::OpaqueType::get(ctx, "uint32_t")
                                : emitc::OpaqueType::get(ctx, "uint64_t");
              else
                emitcType = emitc::OpaqueType::get(ctx, "uint32_t");
              std::string valStr =
                  std::to_string(static_cast<uint64_t>(intVal));
              if (emitcType == emitc::OpaqueType::get(ctx, "uint32_t"))
                valStr += "u";
              replacement = emitc::ConstantOp::create(
                  localBuilder, defOp->getLoc(), emitcType,
                  emitc::OpaqueAttr::get(ctx, valStr));
            }
          }
          if (!replacement) {
            // Fallback: just create a 0u constant.
            replacement = emitc::ConstantOp::create(
                localBuilder, defOp->getLoc(),
                emitc::OpaqueType::get(localBuilder.getContext(), "uint32_t"),
                emitc::OpaqueAttr::get(localBuilder.getContext(),
                                       "0u /* fixup */"));
          }
          fixupCache[val] = replacement;
          operand.set(replacement);
        }
      });
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
    emitc::IncludeOp::create(builder, moduleOp.getLoc(), "vector",
                             /*is_standard=*/true);
  }

private:
  /// Recursively emit EmitC ops for all operations in a block.
  LogicalResult emitOpsForBlock(OpBuilder &builder, Block &block, Location loc,
                                Value txnVec, Value opCountLval,
                                IRMapping &argMapping) {
    for (auto &op : block.getOperations()) {
      if (failed(emitOp(builder, &op, loc, txnVec, opCountLval, argMapping)))
        return failure();
    }
    return success();
  }

  /// Emit a single operation.
  LogicalResult emitOp(OpBuilder &builder, Operation *op, Location loc,
                       Value txnVec, Value opCountLval, IRMapping &argMapping) {
    Location opLoc = op->getLoc();

    // AIEX write32 - handles both static and dynamic forms.
    if (auto write32 = dyn_cast<AIEX::NpuWrite32Op>(op)) {
      Value addrVal, valVal;
      if (write32.hasDynamicOperands()) {
        addrVal = argMapping.lookupOrDefault(write32.getDynAddress());
        valVal = argMapping.lookupOrDefault(write32.getDynValue());
      } else {
        uint32_t addr = write32.getAddress();
        if (auto absAddr = write32.getAbsoluteAddress())
          addr = *absAddr;
        addrVal = createU32Constant(builder, opLoc, addr);
        valVal = createU32Constant(builder, opLoc, write32.getValue());
      }
      emitTxnWrite32(builder, opLoc, txnVec, addrVal, valVal, opCountLval);
      return success();
    }

    // AIEX maskwrite32 - handles both static and dynamic forms.
    if (auto maskWrite = dyn_cast<AIEX::NpuMaskWrite32Op>(op)) {
      Value addrVal, valVal, maskVal;
      if (maskWrite.hasDynamicOperands()) {
        addrVal = argMapping.lookupOrDefault(maskWrite.getDynAddress());
        valVal = argMapping.lookupOrDefault(maskWrite.getDynValue());
        maskVal = argMapping.lookupOrDefault(maskWrite.getDynMask());
      } else {
        uint32_t addr = maskWrite.getAddress();
        if (auto absAddr = maskWrite.getAbsoluteAddress())
          addr = *absAddr;
        addrVal = createU32Constant(builder, opLoc, addr);
        valVal = createU32Constant(builder, opLoc, maskWrite.getValue());
        maskVal = createU32Constant(builder, opLoc, maskWrite.getMask());
      }
      emitTxnMaskWrite32(builder, opLoc, txnVec, addrVal, valVal, maskVal,
                         opCountLval);
      return success();
    }

    // AIEX sync - handles both static and dynamic forms.
    if (auto syncOp = dyn_cast<AIEX::NpuSyncOp>(op)) {
      Value col, row, dir, chan, ncol, nrow;
      if (syncOp.hasDynamicOperands()) {
        col = argMapping.lookupOrDefault(syncOp.getDynColumn());
        row = argMapping.lookupOrDefault(syncOp.getDynRow());
        dir = argMapping.lookupOrDefault(syncOp.getDynDirection());
        chan = argMapping.lookupOrDefault(syncOp.getDynChannel());
        ncol = argMapping.lookupOrDefault(syncOp.getDynColumnNum());
        nrow = argMapping.lookupOrDefault(syncOp.getDynRowNum());
      } else {
        col = createU32Constant(builder, opLoc, syncOp.getColumn());
        row = createU32Constant(builder, opLoc, syncOp.getRow());
        dir = createU32Constant(builder, opLoc,
                                static_cast<uint32_t>(syncOp.getDirection()));
        chan = createU32Constant(builder, opLoc, syncOp.getChannel());
        ncol = createU32Constant(builder, opLoc, syncOp.getColumnNum());
        nrow = createU32Constant(builder, opLoc, syncOp.getRowNum());
      }
      emitTxnSync(builder, opLoc, txnVec, col, row, dir, chan, ncol, nrow,
                  opCountLval);
      return success();
    }

    if (auto addrPatch = dyn_cast<AIEX::NpuAddressPatchOp>(op)) {
      Value dynPlus = addrPatch.getDynArgPlus();
      if (dynPlus)
        dynPlus = argMapping.lookupOrDefault(dynPlus);
      emitTxnAddressPatch(builder, opLoc, txnVec, addrPatch.getAddr(),
                          addrPatch.getArgIdx(), addrPatch.getArgPlus(),
                          dynPlus, opCountLval);
      return success();
    }

    if (auto blockWrite = dyn_cast<AIEX::NpuBlockWriteOp>(op)) {
      uint32_t addr = blockWrite.getAddress();
      if (auto absAddr = blockWrite.getAbsoluteAddress())
        addr = *absAddr;
      auto data = blockWrite.getDataWords();
      emitTxnBlockWrite(builder, opLoc, txnVec, addr, data, opCountLval);
      return success();
    }

    // Arithmetic ops - emit inline C++ expressions.
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      auto *ctx = builder.getContext();
      Value result;
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t intVal = intAttr.getInt();
        Type origType = constOp.getType();
        Type emitcType;
        if (origType.isIndex()) {
          emitcType = emitc::OpaqueType::get(ctx, "size_t");
        } else if (auto iType = dyn_cast<IntegerType>(origType)) {
          emitcType = iType.getWidth() <= 32
                          ? getU32Type(ctx)
                          : emitc::OpaqueType::get(ctx, "uint64_t");
        } else {
          emitcType = getU32Type(ctx);
        }
        std::string valStr;
        if (intVal < 0) {
          // For negative, use cast
          valStr = "static_cast<" +
                   cast<emitc::OpaqueType>(emitcType).getValue().str() + ">(" +
                   std::to_string(intVal) + ")";
        } else {
          valStr = std::to_string(static_cast<uint64_t>(intVal));
          if (emitcType == getU32Type(ctx))
            valStr += "u";
        }
        result = emitc::ConstantOp::create(builder, opLoc, emitcType,
                                           emitc::OpaqueAttr::get(ctx, valStr));
      } else {
        // Unsupported constant type - just emit 0.
        result = createU32Constant(builder, opLoc, 0);
      }
      argMapping.map(constOp.getResult(), result);
      return success();
    }

    if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(addOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(addOp.getRhs());
      // Unify types.
      Type resType = lhs.getType();
      Value result = emitc::AddOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(addOp.getResult(), result);
      return success();
    }

    if (auto subOp = dyn_cast<arith::SubIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(subOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(subOp.getRhs());
      Type resType = lhs.getType();
      Value result = emitc::SubOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(subOp.getResult(), result);
      return success();
    }

    if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(mulOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(mulOp.getRhs());
      Type resType = lhs.getType();
      Value result = emitc::MulOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(mulOp.getResult(), result);
      return success();
    }

    if (auto divOp = dyn_cast<arith::DivUIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(divOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(divOp.getRhs());
      Type resType = lhs.getType();
      Value result = emitc::DivOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(divOp.getResult(), result);
      return success();
    }

    if (auto remOp = dyn_cast<arith::RemUIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(remOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(remOp.getRhs());
      Type resType = lhs.getType();
      Value result = emitc::RemOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(remOp.getResult(), result);
      return success();
    }

    if (auto shlOp = dyn_cast<arith::ShLIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(shlOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(shlOp.getRhs());
      Type resType = lhs.getType();
      Value result =
          emitc::BitwiseLeftShiftOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(shlOp.getResult(), result);
      return success();
    }

    if (auto shrOp = dyn_cast<arith::ShRUIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(shrOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(shrOp.getRhs());
      Type resType = lhs.getType();
      Value result =
          emitc::BitwiseRightShiftOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(shrOp.getResult(), result);
      return success();
    }

    if (auto orOp = dyn_cast<arith::OrIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(orOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(orOp.getRhs());
      Type resType = lhs.getType();
      Value result =
          emitc::BitwiseOrOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(orOp.getResult(), result);
      return success();
    }

    if (auto andOp = dyn_cast<arith::AndIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(andOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(andOp.getRhs());
      Type resType = lhs.getType();
      Value result =
          emitc::BitwiseAndOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(andOp.getResult(), result);
      return success();
    }

    if (auto xorOp = dyn_cast<arith::XOrIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(xorOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(xorOp.getRhs());
      Type resType = lhs.getType();
      Value result =
          emitc::BitwiseXorOp::create(builder, opLoc, resType, lhs, rhs);
      argMapping.map(xorOp.getResult(), result);
      return success();
    }

    if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(cmpOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(cmpOp.getRhs());

      // For signed predicates, cast operands to int32_t before comparing.
      // Our representation uses uint32_t for all i32 values, but signed
      // comparisons require signed arithmetic to handle negative values
      // correctly.
      bool isSigned = false;
      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::sge:
        isSigned = true;
        break;
      default:
        break;
      }
      if (isSigned) {
        lhs = castToI32(builder, opLoc, lhs);
        rhs = castToI32(builder, opLoc, rhs);
      }

      // Map arith::CmpIPredicate to emitc::CmpPredicate.
      emitc::CmpPredicate emitcPred;
      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        emitcPred = emitc::CmpPredicate::eq;
        break;
      case arith::CmpIPredicate::ne:
        emitcPred = emitc::CmpPredicate::ne;
        break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        emitcPred = emitc::CmpPredicate::lt;
        break;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        emitcPred = emitc::CmpPredicate::le;
        break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        emitcPred = emitc::CmpPredicate::gt;
        break;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        emitcPred = emitc::CmpPredicate::ge;
        break;
      default:
        emitcPred = emitc::CmpPredicate::eq;
        break;
      }

      auto i1Type = emitc::OpaqueType::get(builder.getContext(), "bool");
      Value result =
          emitc::CmpOp::create(builder, opLoc, i1Type, emitcPred, lhs, rhs);
      argMapping.map(cmpOp.getResult(), result);
      return success();
    }

    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      Value cond = argMapping.lookupOrDefault(selectOp.getCondition());
      Value trueVal = argMapping.lookupOrDefault(selectOp.getTrueValue());
      Value falseVal = argMapping.lookupOrDefault(selectOp.getFalseValue());
      Type resType = trueVal.getType();
      // emitc.conditional requires i1 condition; cast if needed.
      if (!cond.getType().isInteger(1))
        cond = emitc::CastOp::create(
            builder, opLoc, IntegerType::get(builder.getContext(), 1), cond);
      Value result = emitc::ConditionalOp::create(builder, opLoc, resType, cond,
                                                  trueVal, falseVal);
      argMapping.map(selectOp.getResult(), result);
      return success();
    }

    if (auto indexCast = dyn_cast<arith::IndexCastOp>(op)) {
      Value input = argMapping.lookupOrDefault(indexCast.getIn());
      Type origResultType = indexCast.getType();
      auto *ctx = builder.getContext();
      Type emitcType;
      if (origResultType.isIndex()) {
        emitcType = emitc::OpaqueType::get(ctx, "size_t");
      } else if (auto iType = dyn_cast<IntegerType>(origResultType)) {
        emitcType = iType.getWidth() <= 32
                        ? getU32Type(ctx)
                        : emitc::OpaqueType::get(ctx, "uint64_t");
      } else {
        emitcType = getU32Type(ctx);
      }
      Value result = emitc::CastOp::create(builder, opLoc, emitcType, input);
      argMapping.map(indexCast.getResult(), result);
      return success();
    }

    if (auto minOp = dyn_cast<arith::MinSIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(minOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(minOp.getRhs());
      // arith.minsi is a SIGNED operation. Cast to int32_t before std::min
      // to preserve correct signed semantics (uint32_t would wrap on negative).
      auto i32Type = getI32Type(builder.getContext());
      Value lhsSigned = castToI32(builder, opLoc, lhs);
      Value rhsSigned = castToI32(builder, opLoc, rhs);
      auto callOp =
          emitc::CallOpaqueOp::create(builder, opLoc, i32Type, "std::min",
                                      ValueRange{lhsSigned, rhsSigned});
      // Cast result back to the original type (uint32_t in our representation)
      Value result = emitc::CastOp::create(builder, opLoc, lhs.getType(),
                                           callOp.getResult(0));
      argMapping.map(minOp.getResult(), result);
      return success();
    }

    if (auto maxOp = dyn_cast<arith::MaxSIOp>(op)) {
      Value lhs = argMapping.lookupOrDefault(maxOp.getLhs());
      Value rhs = argMapping.lookupOrDefault(maxOp.getRhs());
      // arith.maxsi is a SIGNED operation. Cast to int32_t before std::max.
      auto i32Type = getI32Type(builder.getContext());
      Value lhsSigned = castToI32(builder, opLoc, lhs);
      Value rhsSigned = castToI32(builder, opLoc, rhs);
      auto callOp =
          emitc::CallOpaqueOp::create(builder, opLoc, i32Type, "std::max",
                                      ValueRange{lhsSigned, rhsSigned});
      Value result = emitc::CastOp::create(builder, opLoc, lhs.getType(),
                                           callOp.getResult(0));
      argMapping.map(maxOp.getResult(), result);
      return success();
    }

    if (auto truncOp = dyn_cast<arith::TruncIOp>(op)) {
      Value input = argMapping.lookupOrDefault(truncOp.getIn());
      Type resultType = truncOp.getType();
      Value result = emitc::CastOp::create(builder, opLoc, resultType, input);
      argMapping.map(truncOp.getResult(), result);
      return success();
    }

    if (auto extOp = dyn_cast<arith::ExtUIOp>(op)) {
      Value input = argMapping.lookupOrDefault(extOp.getIn());
      Type resultType = extOp.getType();
      Value result = emitc::CastOp::create(builder, opLoc, resultType, input);
      argMapping.map(extOp.getResult(), result);
      return success();
    }

    if (auto extOp = dyn_cast<arith::ExtSIOp>(op)) {
      Value input = argMapping.lookupOrDefault(extOp.getIn());
      Type resultType = extOp.getType();
      Value result = emitc::CastOp::create(builder, opLoc, resultType, input);
      argMapping.map(extOp.getResult(), result);
      return success();
    }

    // SCF control flow.
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return emitScfFor(builder, forOp, txnVec, opCountLval, argMapping);
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return emitScfIf(builder, ifOp, txnVec, opCountLval, argMapping);
    }

    // scf.yield: assign yielded values to the corresponding mutable variables
    // (set up by emitScfFor/emitScfIf).
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      for (auto [yieldVal, resultVar] :
           llvm::zip(yieldOp.getOperands(), yieldTargets)) {
        Value mapped = argMapping.lookupOrDefault(yieldVal);
        emitc::AssignOp::create(builder, opLoc, resultVar, mapped);
      }
      return success();
    }

    if (isa<AIE::EndOp>(op))
      return success();

    // Ignore unknown ops with a comment.
    emitc::VerbatimOp::create(
        builder, opLoc,
        "/* unsupported: " + op->getName().getStringRef().str() + " */");
    return success();
  }

  /// Emit an scf.for as emitc.for.
  LogicalResult emitScfFor(OpBuilder &builder, scf::ForOp forOp, Value txnVec,
                           Value opCountLval, IRMapping &argMapping) {
    Location loc = forOp.getLoc();

    Value lb = argMapping.lookupOrDefault(forOp.getLowerBound());
    Value ub = argMapping.lookupOrDefault(forOp.getUpperBound());
    Value step = argMapping.lookupOrDefault(forOp.getStep());

    // Handle iter_args: create mutable variables before the loop,
    // initialize them with the init values, and map the iter_args and
    // for-op results to those variables.
    SmallVector<Value> iterVars;
    for (auto [initVal, iterArg] :
         llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
      Value mappedInit = argMapping.lookupOrDefault(initVal);
      // Create a mutable variable: type var = init;
      auto lvalType = emitc::LValueType::get(mappedInit.getType());
      auto var = emitc::VariableOp::create(
          builder, loc, lvalType,
          emitc::OpaqueAttr::get(builder.getContext(), "{}"));
      emitc::AssignOp::create(builder, loc, var.getResult(), mappedInit);
      iterVars.push_back(var.getResult());
      argMapping.map(iterArg, var.getResult());
    }

    // Create emitc.for.
    auto emitcFor = emitc::ForOp::create(builder, loc, lb, ub, step);

    // Map the induction variable.
    argMapping.map(forOp.getInductionVar(), emitcFor.getInductionVar());

    // Push yield targets for the loop body.
    auto savedTargets = yieldTargets;
    yieldTargetStack.push_back(iterVars);
    yieldTargets = yieldTargetStack.back();

    // Emit body.
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(emitcFor.getBody());

      // Emit loads for iter_arg variables at the top of the loop body.
      // This gives each iteration a snapshot of the current variable value.
      for (auto [iterArg, var] :
           llvm::zip(forOp.getRegionIterArgs(), iterVars)) {
        auto loadedType = cast<emitc::LValueType>(var.getType()).getValueType();
        Value loaded = emitc::LoadOp::create(builder, loc, loadedType, var);
        argMapping.map(iterArg, loaded);
      }

      for (auto &bodyOp : forOp.getBody()->getOperations()) {
        if (failed(
                emitOp(builder, &bodyOp, loc, txnVec, opCountLval, argMapping)))
          return failure();
      }
    }

    // Pop yield targets.
    yieldTargetStack.pop_back();
    yieldTargets = savedTargets;

    // Map the for-op results to loaded values from the mutable variables.
    for (auto [result, var] : llvm::zip(forOp.getResults(), iterVars)) {
      auto loadedType = cast<emitc::LValueType>(var.getType()).getValueType();
      Value loaded = emitc::LoadOp::create(builder, loc, loadedType, var);
      argMapping.map(result, loaded);
    }

    return success();
  }

  /// Emit an scf.if as emitc.if, with support for result values.
  LogicalResult emitScfIf(OpBuilder &builder, scf::IfOp ifOp, Value txnVec,
                          Value opCountLval, IRMapping &argMapping) {
    Location loc = ifOp.getLoc();

    Value cond = argMapping.lookupOrDefault(ifOp.getCondition());
    // emitc.if requires i1 condition; cast if needed.
    if (!cond.getType().isInteger(1))
      cond = emitc::CastOp::create(
          builder, loc, IntegerType::get(builder.getContext(), 1), cond);

    // If the scf.if has results, create mutable variables before the if.
    SmallVector<Value> resultVars;
    for (auto result : ifOp.getResults()) {
      auto lvalType = emitc::LValueType::get(result.getType());
      auto var = emitc::VariableOp::create(
          builder, loc, lvalType,
          emitc::OpaqueAttr::get(builder.getContext(), "{}"));
      resultVars.push_back(var.getResult());
    }

    bool hasElse = !ifOp.getElseRegion().empty();
    auto emitcIf = emitc::IfOp::create(builder, loc, cond, hasElse);

    // Push yield targets for the if body.
    auto savedTargets = yieldTargets;
    yieldTargetStack.push_back(resultVars);
    yieldTargets = yieldTargetStack.back();

    // Emit then body.
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&emitcIf.getThenRegion().front());
      for (auto &thenOp : ifOp.getThenRegion().front().getOperations()) {
        if (failed(
                emitOp(builder, &thenOp, loc, txnVec, opCountLval, argMapping)))
          return failure();
      }
      // Add emitc.yield terminator for the then block.
      emitc::YieldOp::create(builder, loc);
    }

    // Emit else body.
    if (hasElse) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&emitcIf.getElseRegion().front());
      for (auto &elseOp : ifOp.getElseRegion().front().getOperations()) {
        if (failed(
                emitOp(builder, &elseOp, loc, txnVec, opCountLval, argMapping)))
          return failure();
      }
      // Add emitc.yield terminator for the else block.
      emitc::YieldOp::create(builder, loc);
    }

    // Pop yield targets.
    yieldTargetStack.pop_back();
    yieldTargets = savedTargets;

    // Map the if-op results to loaded values from the mutable variables.
    for (auto [result, var] : llvm::zip(ifOp.getResults(), resultVars)) {
      auto loadedType = cast<emitc::LValueType>(var.getType()).getValueType();
      Value loaded = emitc::LoadOp::create(builder, loc, loadedType, var);
      argMapping.map(result, loaded);
    }

    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::createConvertAIEXToEmitCPass() {
  return std::make_unique<ConvertAIEXToEmitCPass>();
}
