//===- AIECoreDMARearm.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower aie.dma_rearm onto the writes that arm a DMA channel.
//
// The writes are not spelled out here. aie-rt is asked to arm the channel
// inside a transaction, and the transaction it records is read back and
// replayed as stores the core makes over its processor bus. A restatement of
// the register map would be one more copy to keep in step with aie-rt.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Runtime/TransactionBinary.h"
#include "aie/Targets/AIERT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIECOREDMAREARM
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

/// The core reaches its own tile's registers through a window at this offset;
/// see the write_tm intrinsic, whose own default says the same.
constexpr uint64_t kTileMemoryMappedBase = 0x80000;

/// The name of the store intrinsic for this architecture.
static std::optional<std::string> writeTmIntrinsic(AIEArch arch) {
  switch (arch) {
  case AIEArch::AIE2:
    return "llvm.aie2.write.tm";
  case AIEArch::AIE2p:
    return "llvm.aie2p.write.tm";
  default:
    return std::nullopt;
  }
}

/// The writes aie-rt makes to arm `op`'s channel, as (tile offset, value).
static LogicalResult
rearmWrites(DmaRearmOp op, int col, int row, const AIETargetModel &targetModel,
            SmallVectorImpl<std::pair<uint64_t, uint32_t>> &writes) {
  AIERTControl ctl(targetModel);
  if (failed(ctl.setIOBackend(/*aieSim=*/false, /*xaieDebug=*/false)))
    return op.emitOpError("could not start aie-rt to collect its writes");
  ctl.startTransaction();
  if (failed(ctl.pushToBdQueueAndEnable(*op.getOperation(), col, row,
                                        op.getChannel(), op.getDirection(),
                                        op.getBdId(), op.getRepeatCount(),
                                        /*padValue=*/0, op.getOutOfOrder())))
    return failure();

  std::vector<uint8_t> txn = ctl.exportSerializedTransaction();
  std::vector<TransactionBinaryOperation> ops;
  if (!parseTransactionBinary(txn, ops))
    return op.emitOpError("could not read back what aie-rt recorded");

  // Only this tile's registers are the core's to write, and only a plain
  // store is. Anything else means aie-rt arms a channel in a way this
  // lowering does not carry, and silently dropping it would leave the
  // channel unarmed.
  uint64_t tileBase =
      (static_cast<uint64_t>(col) << targetModel.getColumnShift()) |
      (static_cast<uint64_t>(row) << targetModel.getRowShift());
  uint64_t tileSpan = 1ull << targetModel.getRowShift();
  for (const TransactionBinaryOperation &txnOp : ops) {
    if (txnOp.cmd.Opcode != XAie_TxnOpcode::XAIE_IO_WRITE)
      return op.emitOpError("arming this channel takes a ")
             << static_cast<int>(txnOp.cmd.Opcode)
             << " transaction command, which a core cannot make; only plain "
                "register writes lower here";
    uint64_t addr = txnOp.cmd.RegOff;
    if (addr < tileBase || addr >= tileBase + tileSpan)
      return op.emitOpError("arming this channel writes ")
             << addr << ", outside tile (" << col << ", " << row
             << "); a core reaches only its own tile's registers";
    writes.emplace_back(addr - tileBase, txnOp.cmd.Value);
  }
  if (writes.empty())
    return op.emitOpError("aie-rt recorded no writes for this arming");
  return success();
}

struct AIECoreDMARearmPass
    : public xilinx::AIE::impl::AIECoreDMARearmBase<AIECoreDMARearmPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const AIETargetModel &targetModel = device.getTargetModel();
    std::optional<std::string> intrinsic =
        writeTmIntrinsic(targetModel.getTargetArch());

    SmallVector<DmaRearmOp> rearms;
    device.walk([&](DmaRearmOp op) { rearms.push_back(op); });
    if (rearms.empty())
      return;

    if (!intrinsic) {
      rearms.front().emitOpError(
          "this device has no store a core can make to its own tile's "
          "registers");
      return signalPassFailure();
    }

    OpBuilder builder(device);
    MLIRContext *ctx = device.getContext();
    Type i32 = builder.getI32Type();
    Type ptr = LLVM::LLVMPointerType::get(ctx);

    // Declared inside the device, which is a symbol table of its own, so
    // that the calls resolve here; outlining the cores later takes the
    // declaration out to the module along with them.
    auto decl = device.lookupSymbol<func::FuncOp>(*intrinsic);
    if (!decl) {
      OpBuilder declBuilder(builder);
      declBuilder.setInsertionPointToStart(device.getBody());
      decl = func::FuncOp::create(declBuilder, device.getLoc(), *intrinsic,
                                  FunctionType::get(ctx, {i32, ptr}, {}));
      decl.setPrivate();
    }

    for (DmaRearmOp op : rearms) {
      CoreOp core = op->getParentOfType<CoreOp>();
      TileOp tile = core.getTileOp();
      SmallVector<std::pair<uint64_t, uint32_t>> writes;
      if (failed(rearmWrites(op, tile.getCol(), tile.getRow(), targetModel,
                             writes)))
        return signalPassFailure();

      // The configuration enables the bus for this core before it runs.
      core->setAttr(kProcessorBusAttrName, builder.getUnitAttr());

      builder.setInsertionPoint(op);
      for (auto [offset, value] : writes) {
        Value valueOp = arith::ConstantOp::create(
            builder, op.getLoc(), i32,
            builder.getI32IntegerAttr(static_cast<int32_t>(value)));
        Value addrOp = arith::ConstantOp::create(
            builder, op.getLoc(), i32,
            builder.getI32IntegerAttr(
                static_cast<int32_t>(kTileMemoryMappedBase + offset)));
        Value addrPtr =
            LLVM::IntToPtrOp::create(builder, op.getLoc(), ptr, addrOp);
        // A call to a declaration, not llvm.call_intrinsic: the intrinsic is
        // the AIE backend's, and translation to LLVM IR runs against an LLVM
        // that does not have it. A named callee is what the other AIE
        // intrinsics the cores use are reached through too.
        func::CallOp::create(builder, op.getLoc(), decl,
                             ValueRange{valueOp, addrPtr});
      }
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIECoreDMARearmPass() {
  return std::make_unique<AIECoreDMARearmPass>();
}
