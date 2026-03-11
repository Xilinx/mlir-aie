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
                                Value opCountLval) {
  auto addrVal = createU32Constant(builder, loc, addr);
  auto idxVal = createU32Constant(builder, loc, static_cast<uint32_t>(argIdx));
  auto plusVal =
      createU32Constant(builder, loc, static_cast<uint32_t>(argPlus));
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
      // increment it (via VerbatimOp) and use it in the footer (via VerbatimOp).
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

    // Dynamic AIEX ops.
    if (auto dynWrite = dyn_cast<AIEX::NpuDynWrite32Op>(op)) {
      Value addr = argMapping.lookupOrDefault(dynWrite.getAddress());
      Value val = argMapping.lookupOrDefault(dynWrite.getValue());
      emitTxnWrite32(builder, opLoc, txnVec, addr, val, opCountLval);
      return success();
    }

    if (auto dynMask = dyn_cast<AIEX::NpuDynMaskWrite32Op>(op)) {
      Value addr = argMapping.lookupOrDefault(dynMask.getAddress());
      Value val = argMapping.lookupOrDefault(dynMask.getValue());
      Value mask = argMapping.lookupOrDefault(dynMask.getMask());
      emitTxnMaskWrite32(builder, opLoc, txnVec, addr, val, mask, opCountLval);
      return success();
    }

    if (auto dynSync = dyn_cast<AIEX::NpuDynSyncOp>(op)) {
      Value col = argMapping.lookupOrDefault(dynSync.getColumn());
      Value row = argMapping.lookupOrDefault(dynSync.getRow());
      Value dir = argMapping.lookupOrDefault(dynSync.getDirection());
      Value chan = argMapping.lookupOrDefault(dynSync.getChannel());
      Value ncol = argMapping.lookupOrDefault(dynSync.getColumnNum());
      Value nrow = argMapping.lookupOrDefault(dynSync.getRowNum());
      emitTxnSync(builder, opLoc, txnVec, col, row, dir, chan, ncol, nrow,
                  opCountLval);
      return success();
    }

    // Static AIEX ops - use constants for attribute values.
    if (auto write32 = dyn_cast<AIEX::NpuWrite32Op>(op)) {
      uint32_t addr = write32.getAddress();
      if (auto absAddr = write32.getAbsoluteAddress())
        addr = *absAddr;
      Value addrVal = createU32Constant(builder, opLoc, addr);
      Value valVal = createU32Constant(builder, opLoc, write32.getValue());
      emitTxnWrite32(builder, opLoc, txnVec, addrVal, valVal, opCountLval);
      return success();
    }

    if (auto maskWrite = dyn_cast<AIEX::NpuMaskWrite32Op>(op)) {
      uint32_t addr = maskWrite.getAddress();
      if (auto absAddr = maskWrite.getAbsoluteAddress())
        addr = *absAddr;
      Value addrVal = createU32Constant(builder, opLoc, addr);
      Value valVal = createU32Constant(builder, opLoc, maskWrite.getValue());
      Value maskVal = createU32Constant(builder, opLoc, maskWrite.getMask());
      emitTxnMaskWrite32(builder, opLoc, txnVec, addrVal, valVal, maskVal,
                         opCountLval);
      return success();
    }

    if (auto syncOp = dyn_cast<AIEX::NpuSyncOp>(op)) {
      Value col = createU32Constant(builder, opLoc, syncOp.getColumn());
      Value row = createU32Constant(builder, opLoc, syncOp.getRow());
      Value dir = createU32Constant(
          builder, opLoc, static_cast<uint32_t>(syncOp.getDirection()));
      Value chan = createU32Constant(builder, opLoc, syncOp.getChannel());
      Value ncol = createU32Constant(builder, opLoc, syncOp.getColumnNum());
      Value nrow = createU32Constant(builder, opLoc, syncOp.getRowNum());
      emitTxnSync(builder, opLoc, txnVec, col, row, dir, chan, ncol, nrow,
                  opCountLval);
      return success();
    }

    if (auto addrPatch = dyn_cast<AIEX::NpuAddressPatchOp>(op)) {
      emitTxnAddressPatch(builder, opLoc, txnVec, addrPatch.getAddr(),
                          addrPatch.getArgIdx(), addrPatch.getArgPlus(),
                          opCountLval);
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

    // SCF control flow.
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return emitScfFor(builder, forOp, txnVec, opCountLval, argMapping);
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return emitScfIf(builder, ifOp, txnVec, opCountLval, argMapping);
    }

    // Terminators - skip.
    if (isa<scf::YieldOp>(op) || isa<AIE::EndOp>(op))
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

    // Create emitc.for.
    auto emitcFor = emitc::ForOp::create(builder, loc, lb, ub, step);

    // Map the induction variable.
    argMapping.map(forOp.getInductionVar(), emitcFor.getInductionVar());

    // Emit body.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(emitcFor.getBody());

    for (auto &bodyOp : forOp.getBody()->getOperations()) {
      if (failed(
              emitOp(builder, &bodyOp, loc, txnVec, opCountLval, argMapping)))
        return failure();
    }

    return success();
  }

  /// Emit an scf.if as emitc.if.
  LogicalResult emitScfIf(OpBuilder &builder, scf::IfOp ifOp, Value txnVec,
                          Value opCountLval, IRMapping &argMapping) {
    Location loc = ifOp.getLoc();

    Value cond = argMapping.lookupOrDefault(ifOp.getCondition());

    bool hasElse = !ifOp.getElseRegion().empty();
    auto emitcIf = emitc::IfOp::create(builder, loc, cond, hasElse);

    // Emit then body.
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&emitcIf.getThenRegion().front());
      for (auto &thenOp : ifOp.getThenRegion().front().getOperations()) {
        if (failed(
                emitOp(builder, &thenOp, loc, txnVec, opCountLval, argMapping)))
          return failure();
      }
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
    }

    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::createConvertAIEXToEmitCPass() {
  return std::make_unique<ConvertAIEXToEmitCPass>();
}
