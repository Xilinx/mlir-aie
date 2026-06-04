//===- AIELowerParameters.cpp - Lower parameter ops to scratchpad ---------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIELOWERPARAMETERS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

namespace {

/// Returns true iff the parameter-sync preamble (use_lock acquire) should be
/// emitted for this core. Reads the explicit attribute if present; otherwise
/// defaults to true iff any 'aiex.read_parameter' op is present in the core
/// body.
static bool shouldEmitParameterSyncPreamble(CoreOp coreOp) {
  if (auto attr = coreOp.getEmitParameterSyncPreambleAttr()) {
    return attr.getValue();
  }
  bool found = false;
  coreOp.getBody().walk([&](ReadParameterOp) { found = true; });
  return found;
}

/// Returns true iff the parameter-sync preamble (create_scratchpad + set_lock)
/// should be emitted into this sequence. Reads the explicit attribute if
/// present; otherwise defaults to true iff the parent device contains any
/// ReadParameterOp in a core, or any 'offset_parameter' attribute on a DMA BD,
/// and the runtime sequence does not already contain a
// `aiex.sync_parameters_from_host` marker.
static bool shouldEmitParameterSyncPreamble(RuntimeSequenceOp seqOp) {
  if (auto attr = seqOp.getEmitParameterSyncPreambleAttr()) {
    return attr.getValue();
  }
  bool hasManualSync = false;
  seqOp.walk([&](SyncParametersFromHostOp) { hasManualSync = true; });
  if (hasManualSync) {
    return false;
  }
  auto device = seqOp->getParentOfType<DeviceOp>();
  if (!device) {
    return false;
  }
  bool found = false;
  device.walk([&](Operation *op) {
    if (found) {
      return;
    }
    if (llvm::isa<ReadParameterOp>(op) || op->hasAttr("offset_parameter") ||
        op->hasAttr("offset_state_table_idx")) {
      found = true;
    }
  });
  return found;
}

struct AIELowerParametersPass
    : public xilinx::AIEX::impl::AIELowerParametersBase<
          AIELowerParametersPass> {
  using AIELowerParametersBase::AIELowerParametersBase;

  // For each read_parameter of a unique parameter, create a 2xi32 buffer and
  // store a reference to it on the ReadParameterOp as the `buffer` attribute.
  void allocateBuffers(DeviceOp device, OpBuilder &builder) {
    MLIRContext *ctx = device.getContext();
    unsigned uniquingCounter = 0;

    DenseMap<std::pair<StringRef, Operation *>, BufferOp> seen;

    device.walk([&](ReadParameterOp readOp) {
      auto coreOp = readOp->getParentOfType<CoreOp>();
      TileOp tile = coreOp.getTileOp();
      StringRef paramName = readOp.getParameter();
      auto key = std::make_pair(paramName, tile.getOperation());

      if (seen.count(key)) {
        readOp.setBufferAttr(
            FlatSymbolRefAttr::get(ctx, *seen[key].getSymName()));
        return;
      }

      builder.setInsertionPointAfter(tile);
      // Buffer must be 8 bytes: update_from_scratchpad always writes a 48-bit
      // value across two 32-bit registers at [RegOff] and [RegOff+4]. The
      // firmware masks Reg[0] with 0xFFFFFFFC (lower 2 bits forced to 0)
      // because it was designed for 4-byte-aligned DMA BD addresses. The host
      // library's `ParameterScratchpad::write` left-shifts by 2 before
      // writing to the scratchpad, and the core right-shifts by 2 after
      // loading. Note that this limits effective parameter values to 30
      // bits.
      auto bufType = MemRefType::get({2}, builder.getI32Type());
      std::string prefix =
          ("__param_" + paramName + "_" + std::to_string(tile.getCol()) + "_" +
           std::to_string(tile.getRow()) + "_")
              .str();
      std::string bufName =
          AIE::generateUniqueSymbolName(device, prefix, uniquingCounter);
      auto buf =
          BufferOp::create(builder, readOp.getLoc(), bufType, tile,
                           builder.getStringAttr(bufName), /*address=*/nullptr,
                           /*initial_value=*/nullptr, /*mem_bank=*/nullptr);
      seen[key] = buf;

      readOp.setBufferAttr(
          FlatSymbolRefAttr::get(ctx, *seen[key].getSymName()));
    });
  }

  // Lower each read_parameter to: load from buffer[0], shift right by 2, and
  // cast to the result type.  The buffer to use comes from the `buffer`
  // attribute set by allocateBuffers().
  void lowerReadParameters(DeviceOp device, OpBuilder &builder) {
    SmallVector<ReadParameterOp> readOps;
    device.walk([&](ReadParameterOp op) { readOps.push_back(op); });

    for (auto readOp : readOps) {
      FlatSymbolRefAttr bufRef = readOp.getBufferAttr();
      auto buf = device.lookupSymbol<BufferOp>(bufRef.getAttr());

      builder.setInsertionPoint(readOp);
      Value c0 = arith::ConstantIndexOp::create(builder, readOp.getLoc(), 0);
      Value raw = memref::LoadOp::create(builder, readOp.getLoc(), buf, c0);
      Value c2 = arith::ConstantOp::create(builder, readOp.getLoc(),
                                           builder.getI32IntegerAttr(2));
      Value decoded = arith::ShRUIOp::create(builder, readOp.getLoc(), raw, c2);

      Type resultType = readOp.getResult().getType();
      Value result = decoded;
      if (resultType.isInteger() && resultType != builder.getI32Type()) {
        result = arith::TruncIOp::create(builder, readOp.getLoc(), resultType,
                                         decoded);
      } else if (resultType.isBF16()) {
        Value masked = arith::TruncIOp::create(builder, readOp.getLoc(),
                                               builder.getI16Type(), decoded);
        result = arith::BitcastOp::create(builder, readOp.getLoc(), resultType,
                                          masked);
      }

      readOp.getResult().replaceAllUsesWith(result);
      readOp.erase();
    }
  }

  // For each core in `device` with shouldEmitParameterSyncPreamble()==true,
  // create an aie.lock (no lockID; AIEAssignLockIDs will assign one later) and
  // insert aie.use_lock(Acquire, 1) at the top of the core body.  Returns the
  // list of lock values created (one per qualifying core, in walk order).
  SmallVector<Value> emitCorePreambles(DeviceOp device, OpBuilder &builder) {
    SmallVector<Value> syncLocks;
    device.walk([&](CoreOp coreOp) {
      if (!shouldEmitParameterSyncPreamble(coreOp)) {
        return;
      }

      TileOp tile = coreOp.getTileOp();
      builder.setInsertionPointAfter(tile);
      auto lockOp = LockOp::create(
          builder, coreOp.getLoc(), builder.getIndexType(), tile.getResult(),
          /*lockID=*/IntegerAttr{}, builder.getI32IntegerAttr(0),
          /*sym_name=*/StringAttr{});
      syncLocks.push_back(lockOp.getResult());

      Block &bodyBlock = coreOp.getBody().front();
      builder.setInsertionPointToStart(&bodyBlock);
      UseLockOp::create(builder, coreOp.getLoc(), lockOp.getResult(),
                        LockAction::Acquire, 1);

      // Mark as done so the pass is idempotent.
      coreOp.setEmitParameterSyncPreambleAttr(builder.getBoolAttr(false));
    });
    return syncLocks;
  }

  // For each runtime sequence in `device` with shouldEmitParameterSyncPreamble
  // ()==true, insert a marker SyncParametersFromHostOp at the start of the
  // sequence body.
  void emitSequencePreambles(DeviceOp device, OpBuilder &builder) {
    device.walk([&](RuntimeSequenceOp seqOp) {
      if (!shouldEmitParameterSyncPreamble(seqOp)) {
        return;
      }

      Block &body = seqOp.getBody().front();
      builder.setInsertionPointToStart(&body);
      SyncParametersFromHostOp::create(builder, seqOp.getLoc());

      // Mark as done so the pass is idempotent.
      seqOp.setEmitParameterSyncPreambleAttr(builder.getBoolAttr(false));
    });
  }

  // Lower a sync_parameters_from_host marker op in place to its component ops,
  // using the parameter information for the device that contains it:
  //   1. npu.create_scratchpad(scratchpadSize)
  //   2. For each (state_idx, buffer_ref) in paramEntries: npu.write32(0) +
  //      npu.update_from_scratchpad
  //   3. set_lock(lock, 1) for each lock in syncLocks
  void lowerSyncParametersOp(
      SyncParametersFromHostOp syncOp, uint32_t scratchpadSize,
      const SmallVector<std::pair<uint8_t, FlatSymbolRefAttr>> &paramEntries,
      const SmallVector<Value> &syncLocks) {
    OpBuilder builder(syncOp);
    Location loc = syncOp.getLoc();

    NpuCreateScratchpadOp::create(builder, loc, scratchpadSize);

    for (auto &[stateIdx, bufRef] : paramEntries) {
      // Zero the destination before the additive UpdateScratchpad.
      NpuWrite32Op::create(builder, loc, /*address=*/0, /*value=*/0, bufRef,
                           /*column=*/nullptr, /*row=*/nullptr);
      NpuUpdateFromScratchpadOp::create(
          builder, loc, stateIdx, StateTableFunc::Incr,
          /*func_arg=*/static_cast<uint32_t>(0),
          /*address=*/static_cast<uint32_t>(0), bufRef,
          /*column=*/nullptr, /*row=*/nullptr);
    }

    for (Value lock : syncLocks) {
      SetLockOp::create(builder, loc, lock, builder.getI32IntegerAttr(1));
    }

    syncOp.erase();
  }

  // Lower every sync_parameters_from_host op in `device` to its component
  // ops in place, using parameter info for this device only.
  void lowerSyncParametersOps(
      DeviceOp device, uint32_t scratchpadSize,
      const SmallVector<std::pair<uint8_t, FlatSymbolRefAttr>> &paramEntries,
      const SmallVector<Value> &syncLocks) {
    SmallVector<SyncParametersFromHostOp> syncOps;
    device.walk([&](SyncParametersFromHostOp op) { syncOps.push_back(op); });
    for (SyncParametersFromHostOp syncOp : syncOps) {
      lowerSyncParametersOp(syncOp, scratchpadSize, paramEntries, syncLocks);
    }
  }

  // Emit a single params.txt for the whole module.
  //
  // Format (one entry per line, easily parsed with std::ifstream >>):
  //   <num_parameters>
  //   <name> <state_table_idx> <type> <kind>
  //   ...
  // where kind is "core" (shift-2 encoded, for read_parameter) or "addr"
  // (raw, for offset_parameter on DMA ops).
  LogicalResult emitParamsFile(ArrayRef<ParameterOp> allParams) {
    if (outputParamsFile.empty())
      return success();

    std::error_code ec;
    llvm::raw_fd_ostream out(outputParamsFile, ec);
    if (ec) {
      return emitError(UnknownLoc::get(&getContext()),
                       "failed to open params output file '")
             << outputParamsFile << "': " << ec.message();
    }

    out << allParams.size() << "\n";
    for (auto p : allParams) {
      std::string typeStr;
      llvm::raw_string_ostream os(typeStr);
      p.getType().print(os);
      StringRef kindStr =
          p.getKind().value() == ParameterKind::Addr ? "addr" : "core";
      out << p.getSymName() << " "
          << static_cast<unsigned>(p.getStateTableIdx().value()) << " "
          << typeStr << " " << kindStr << "\n";
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    // Step 1: collect every parameter in the module.
    SmallVector<ParameterOp> allParams;
    moduleOp.walk([&](ParameterOp p) { allParams.push_back(p); });

    if (allParams.size() > 32) {
      InFlightDiagnostic diag =
          moduleOp.emitError("Module declares ")
          << allParams.size()
          << " parameters but the scratchpad supports at most 32. The "
             "scratchpad is a single hardware resource shared by all PDIs "
             "loaded by a runtime sequence.";
      for (auto p : allParams)
        diag.attachNote(p.getLoc()) << "parameter '" << p.getSymName() << "'";
      return signalPassFailure();
    }

    // Step 2: determine each parameter's kind from its usage, erroring on
    // mixed use.  A parameter is "core" if any aiex.read_parameter references
    // it; "addr" if any DMA op references it via offset_parameter.  If both,
    // emit an error.
    DenseMap<StringRef, bool> usedAsCore;
    DenseMap<StringRef, bool> usedAsAddr;
    moduleOp.walk(
        [&](ReadParameterOp op) { usedAsCore[op.getParameter()] = true; });
    auto markAddr = [&](Operation *op, FlatSymbolRefAttr ref) {
      if (ref)
        usedAsAddr[ref.getValue()] = true;
    };
    moduleOp.walk([&](NpuDmaMemcpyNdOp op) {
      markAddr(op, op.getOffsetParameterAttr());
    });
    moduleOp.walk(
        [&](AIE::DMABDOp op) { markAddr(op, op.getOffsetParameterAttr()); });

    for (auto p : allParams) {
      StringRef name = p.getSymName();
      bool core = usedAsCore.lookup(name);
      bool addr = usedAsAddr.lookup(name);
      if (core && addr) {
        p.emitError("parameter '")
            << name
            << "' is used both as an aiex.read_parameter source (core) and "
               "as a DMA offset_parameter (addr); a parameter must have a "
               "single kind";
        return signalPassFailure();
      }
      p.setKindAttr(ParameterKindAttr::get(
          &getContext(), addr ? ParameterKind::Addr : ParameterKind::Core));
    }

    // Step 3: assign global state_table_idx in walk order, 0..N-1.
    for (auto [i, p] : llvm::enumerate(allParams)) {
      p.setStateTableIdxAttr(builder.getIntegerAttr(
          builder.getIntegerType(8, /*isSigned=*/false), i));
    }

    // Step 3b: rewrite DMA `offset_parameter` symbol references to a plain
    // `offset_state_table_idx` integer attribute, so downstream `aie`-dialect
    // passes do not need to resolve `aiex.parameter` symbols.
    auto rewriteOffsetParam = [&](Operation *op, FlatSymbolRefAttr ref) {
      if (!ref) {
        return success();
      }
      auto paramOp = moduleOp.lookupSymbol<ParameterOp>(ref.getAttr());
      if (!paramOp) {
        op->emitOpError("offset_parameter '")
            << ref.getValue()
            << "' not found. Declare it at module scope with aiex.parameter.";
        return failure();
      }
      if (!paramOp.getType().isInteger(32)) {
        auto err = op->emitOpError("offset_parameter '")
                   << ref.getValue() << "' must have type i32, got "
                   << paramOp.getType() << ".";
        err.attachNote(paramOp.getLoc()) << "Parameter declared here.";
        return failure();
      }
      uint8_t stateIdx =
          static_cast<uint8_t>(paramOp.getStateTableIdx().value());
      op->setAttr("offset_state_table_idx",
                  builder.getIntegerAttr(
                      builder.getIntegerType(8, /*isSigned=*/false), stateIdx));
      op->removeAttr("offset_parameter");
      return success();
    };
    WalkResult rewriteResult = moduleOp.walk([&](Operation *op) {
      if (auto dmaOp = dyn_cast<NpuDmaMemcpyNdOp>(op)) {
        if (failed(rewriteOffsetParam(op, dmaOp.getOffsetParameterAttr()))) {
          return WalkResult::interrupt();
        }
      } else if (auto bdOp = dyn_cast<AIE::DMABDOp>(op)) {
        if (failed(rewriteOffsetParam(op, bdOp.getOffsetParameterAttr()))) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (rewriteResult.wasInterrupted()) {
      return signalPassFailure();
    }

    // Step 4: per-device lowering.
    SmallVector<DeviceOp> devices;
    moduleOp.walk([&](DeviceOp d) { devices.push_back(d); });
    for (auto d : devices) {
      allocateBuffers(d, builder);

      // Collect unique (stateIdx, bufferRef) pairs for core-kind parameters in
      // this device.  Buffer attrs are set by allocateBuffers() above.
      SmallVector<std::pair<uint8_t, FlatSymbolRefAttr>> paramEntries;
      DenseSet<StringRef> seenBufs;
      d.walk([&](ReadParameterOp readOp) {
        FlatSymbolRefAttr bufRef = readOp.getBufferAttr();
        if (!bufRef || !seenBufs.insert(bufRef.getValue()).second) {
          return;
        }
        auto paramOp =
            moduleOp.lookupSymbol<ParameterOp>(readOp.getParameter());
        uint8_t stateIdx =
            static_cast<uint8_t>(paramOp.getStateTableIdx().value());
        paramEntries.push_back({stateIdx, bufRef});
      });

      // Emit lock + use_lock preambles in cores, then insert a marker
      // SyncParametersFromHostOp at the start of each qualifying runtime
      // sequence.  Immediately lower every sync_parameters_from_host op in
      // the device (both the preamble-inserted ones and any user-written
      // ones) using this device's parameter info.
      SmallVector<Value> syncLocks = emitCorePreambles(d, builder);
      emitSequencePreambles(d, builder);
      uint32_t scratchpadSize = static_cast<uint32_t>(allParams.size() * 4);
      lowerSyncParametersOps(d, scratchpadSize, paramEntries, syncLocks);

      lowerReadParameters(d, builder);
    }

    // Step 5: emit the single params.txt for the module.
    if (failed(emitParamsFile(allParams))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIELowerParametersPass() {
  return std::make_unique<AIELowerParametersPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIELowerParametersPass(AIELowerParametersOptions options) {
  return std::make_unique<AIELowerParametersPass>(std::move(options));
}
