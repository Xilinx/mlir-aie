//===- AIELowerParameters.cpp - Lower parameter ops to scratchpad ---------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers aiex.parameter, aiex.read_parameter, and
// aiex.sync_parameters_from_host ops to the lower-level scratchpad ops
// (create_scratchpad, write32, update_from_scratchpad).
//
// State table indices are assigned globally across the entire module: every
// aiex.parameter in any aie.device gets a unique index in [0, 32).  This is
// necessary because the scratchpad is a single hardware resource shared by all
// PDIs loaded by a runtime sequence.
//
// A single params.txt file is emitted describing every parameter in the module
// (name, global state_table_idx, type, and kind: "core" or "addr").
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

      if (!seen.count(key)) {
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
            ("__param_" + paramName + "_" + std::to_string(tile.getCol()) +
             "_" + std::to_string(tile.getRow()) + "_")
                .str();
        std::string bufName =
            AIE::generateUniqueSymbolName(device, prefix, uniquingCounter);
        auto buf = BufferOp::create(
            builder, readOp.getLoc(), bufType, tile,
            builder.getStringAttr(bufName), /*address=*/nullptr,
            /*initial_value=*/nullptr, /*mem_bank=*/nullptr);
        seen[key] = buf;
      }

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
      Value c0 = builder.create<arith::ConstantIndexOp>(readOp.getLoc(), 0);
      Value raw = builder.create<memref::LoadOp>(readOp.getLoc(), buf, c0);
      Value c2 = builder.create<arith::ConstantOp>(
          readOp.getLoc(), builder.getI32IntegerAttr(2));
      Value decoded = builder.create<arith::ShRUIOp>(readOp.getLoc(), raw, c2);

      Type resultType = readOp.getResult().getType();
      Value result = decoded;
      if (resultType != builder.getI32Type()) {
        if (resultType.isInteger()) {
          result = builder.create<arith::TruncIOp>(readOp.getLoc(), resultType,
                                                   decoded);
        } else if (resultType.isBF16()) {
          Value masked = builder.create<arith::TruncIOp>(
              readOp.getLoc(), builder.getI16Type(), decoded);
          result = builder.create<arith::BitcastOp>(readOp.getLoc(), resultType,
                                                    masked);
        } else if (resultType.isF32()) {
          result = builder.create<arith::BitcastOp>(readOp.getLoc(), resultType,
                                                    decoded);
        }
      }

      readOp.getResult().replaceAllUsesWith(result);
      readOp.erase();
    }
  }

  // Lower sync_parameters_from_host to create_scratchpad +
  // update_from_scratchpad sequences. `scratchpadSlots` is the total number of
  // parameters in the whole module (the scratchpad is a single hardware
  // resource shared across all PDIs loaded by a runtime sequence).
  void lowerSyncOps(DeviceOp device, OpBuilder &builder,
                    unsigned scratchpadSlots) {
    auto module = device->getParentOfType<ModuleOp>();
    // Collect unique (stateIdx, bufferRef) pairs from ReadParameterOp attrs.
    SmallVector<std::pair<uint8_t, FlatSymbolRefAttr>> paramEntries;
    DenseSet<StringRef> seenBufs;
    device.walk([&](ReadParameterOp readOp) {
      FlatSymbolRefAttr bufRef = readOp.getBufferAttr();
      if (!seenBufs.insert(bufRef.getValue()).second)
        return;
      auto paramOp = module.lookupSymbol<ParameterOp>(readOp.getParameter());
      uint8_t stateIdx =
          static_cast<uint8_t>(paramOp.getStateTableIdx().value());
      paramEntries.push_back({stateIdx, bufRef});
    });

    device.walk([&](SyncParametersFromHostOp syncOp) {
      builder.setInsertionPoint(syncOp);
      Location loc = syncOp.getLoc();

      NpuCreateScratchpadOp::create(builder, loc,
                                    static_cast<uint32_t>(scratchpadSlots * 4));

      for (auto &[stateIdx, bufRef] : paramEntries) {
        NpuUpdateFromScratchpadOp::create(
            builder, loc, stateIdx, StateTableFunc::Incr,
            /*func_arg=*/static_cast<uint32_t>(0),
            /*address=*/static_cast<uint32_t>(0), bufRef, /*column=*/nullptr,
            /*row=*/nullptr);
      }

      syncOp.erase();
    });
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
    if (ec)
      return emitError(UnknownLoc::get(&getContext()),
                       "failed to open params output file '")
             << outputParamsFile << "': " << ec.message();

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
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    // Step 1: collect every parameter in the module.
    SmallVector<ParameterOp> allParams;
    module.walk([&](ParameterOp p) { allParams.push_back(p); });

    if (allParams.size() > 32) {
      InFlightDiagnostic diag =
          module.emitError("Module declares ")
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
    module.walk(
        [&](ReadParameterOp op) { usedAsCore[op.getParameter()] = true; });
    auto markAddr = [&](Operation *op, FlatSymbolRefAttr ref) {
      if (ref)
        usedAsAddr[ref.getValue()] = true;
    };
    module.walk([&](NpuDmaMemcpyNdOp op) {
      markAddr(op, op.getOffsetParameterAttr());
    });
    module.walk(
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

    unsigned totalParams = allParams.size();

    // Step 4: per-device lowering.
    SmallVector<DeviceOp> devices;
    module.walk([&](DeviceOp d) { devices.push_back(d); });
    for (auto d : devices) {
      allocateBuffers(d, builder);
      lowerSyncOps(d, builder, totalParams);
      lowerReadParameters(d, builder);
    }

    // Step 5: emit the single params.txt for the module.
    if (failed(emitParamsFile(allParams)))
      return signalPassFailure();

    // ParameterOps are kept around so that later passes (e.g. DMA lowering)
    // can resolve `offset_parameter` symbol references back to their
    // state_table_idx / kind / type. `--aiex-standard-lowering` will remove
    // them at the end of the pipeline.
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
