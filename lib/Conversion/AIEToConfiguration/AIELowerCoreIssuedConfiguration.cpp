//===- AIELowerCoreIssuedConfiguration.cpp ----------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower the configuration a core makes onto the accesses it makes itself.
//
// The writes that push a task are not spelled out here. aie-rt is asked to
// push inside a transaction, and the transaction it records is read back and
// replayed as stores the core makes over its processor bus. A restatement of
// the register map would be one more copy to keep in step with aie-rt.

#include "TransactionBinary.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Targets/AIERT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIELOWERCOREISSUEDCONFIGURATION
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

/// The core reaches its own tile's registers through a window at this offset;
/// see the write_tm intrinsic, whose own default says the same.
constexpr uint64_t kTileMemoryMappedBase = 0x80000;

/// The names of the store and load intrinsics for this architecture.
struct TileAccessIntrinsics {
  std::string write, read;
};

static std::optional<TileAccessIntrinsics> tileAccessIntrinsics(AIEArch arch) {
  switch (arch) {
  case AIEArch::AIE2:
    return TileAccessIntrinsics{"llvm.aie2.write.tm", "llvm.aie2.read.tm"};
  case AIEArch::AIE2p:
    return TileAccessIntrinsics{"llvm.aie2p.write.tm", "llvm.aie2p.read.tm"};
  default:
    return std::nullopt;
  }
}

/// The writes aie-rt makes to push `op`'s task, as (tile offset, value).
static LogicalResult
pushWrites(DmaPushQueueOp op, int col, int row,
           const AIETargetModel &targetModel,
           SmallVectorImpl<std::pair<uint64_t, uint32_t>> &writes) {
  if (!op.getBdId())
    return op.emitOpError("names its descriptor by tag; run "
                          "--aie-assign-bd-ids first");
  AIERTControl ctl(targetModel);
  if (failed(ctl.setIOBackend(/*aieSim=*/false, /*xaieDebug=*/false)))
    return op.emitOpError("could not start aie-rt to collect its writes");
  ctl.startTransaction();
  // Only the task queue. Rewriting the channel's control register while the
  // channel runs loses the state it keeps between packets, and the channel
  // this pushes onto is already configured and running.
  if (failed(ctl.pushToBdQueueAndEnable(
          *op.getOperation(), col, row, op.getChannel(), op.getDirection(),
          op.getBdId().value_or(0), op.getRepeatCount(), /*padValue=*/0,
          op.getOutOfOrder(), /*configureChannel=*/false)))
    return failure();

  std::vector<uint8_t> txn = ctl.exportSerializedTransaction();
  std::vector<TransactionBinaryOperation> ops;
  if (!parseTransactionBinary(txn, ops))
    return op.emitOpError("could not read back what aie-rt recorded");

  // Only this tile's registers are the core's to write, and only a plain
  // store is. Anything else means aie-rt pushes a task in a way this lowering
  // does not carry, and silently dropping it would leave the channel without
  // the task.
  uint64_t tileBase =
      (static_cast<uint64_t>(col) << targetModel.getColumnShift()) |
      (static_cast<uint64_t>(row) << targetModel.getRowShift());
  uint64_t tileSpan = 1ull << targetModel.getRowShift();
  for (const TransactionBinaryOperation &txnOp : ops) {
    if (txnOp.cmd.Opcode != XAie_TxnOpcode::XAIE_IO_WRITE)
      return op.emitOpError("pushing this task takes a ")
             << static_cast<int>(txnOp.cmd.Opcode)
             << " transaction command, which a core cannot make; only plain "
                "register writes lower here";
    uint64_t addr = txnOp.cmd.RegOff;
    if (addr < tileBase || addr >= tileBase + tileSpan)
      return op.emitOpError("pushing this task writes ")
             << addr << ", outside tile (" << col << ", " << row
             << "); a core reaches only its own tile's registers";
    writes.emplace_back(addr - tileBase, txnOp.cmd.Value);
  }
  if (writes.empty())
    return op.emitOpError("aie-rt recorded no writes for this push");
  return success();
}

struct AIELowerCoreIssuedConfigurationPass
    : public xilinx::AIE::impl::AIELowerCoreIssuedConfigurationBase<
          AIELowerCoreIssuedConfigurationPass> {

  /// A declaration of `name`, inside the device, which is a symbol table of
  /// its own, so that the calls resolve here; outlining the cores later takes
  /// the declaration out to the module along with them.
  func::FuncOp declare(DeviceOp device, StringRef name, TypeRange argTypes,
                       TypeRange resultTypes) {
    if (auto decl = device.lookupSymbol<func::FuncOp>(name))
      return decl;
    OpBuilder builder(device.getContext());
    builder.setInsertionPointToStart(device.getBody());
    auto decl = func::FuncOp::create(
        builder, device.getLoc(), name,
        FunctionType::get(device.getContext(), argTypes, resultTypes));
    decl.setPrivate();
    return decl;
  }

  /// A pointer to `offset` in this tile's register window.
  Value tilePointer(OpBuilder &builder, Location loc, uint64_t offset) {
    Type i32 = builder.getI32Type();
    Type ptr = LLVM::LLVMPointerType::get(builder.getContext());
    Value addr = arith::ConstantOp::create(
        builder, loc, i32,
        builder.getI32IntegerAttr(
            static_cast<int32_t>(kTileMemoryMappedBase + offset)));
    return LLVM::IntToPtrOp::create(builder, loc, ptr, addr);
  }

  LogicalResult lowerPush(DmaPushQueueOp op, const AIETargetModel &targetModel,
                          func::FuncOp write) {
    CoreOp core = op->getParentOfType<CoreOp>();
    TileOp tile = core.getTileOp();
    SmallVector<std::pair<uint64_t, uint32_t>> writes;
    if (failed(
            pushWrites(op, tile.getCol(), tile.getRow(), targetModel, writes)))
      return failure();
    // The configuration enables the bus for this core before it runs.
    core.setProcessorBus(true);

    OpBuilder builder(op);
    Type i32 = builder.getI32Type();
    for (auto [offset, value] : writes) {
      Value valueOp = arith::ConstantOp::create(
          builder, op.getLoc(), i32,
          builder.getI32IntegerAttr(static_cast<int32_t>(value)));
      Value addrPtr = tilePointer(builder, op.getLoc(), offset);
      // A call to a declaration, not llvm.call_intrinsic: the intrinsic is
      // the AIE backend's, and translation to LLVM IR runs against an LLVM
      // that does not have it. A named callee is what the other AIE
      // intrinsics the cores use are reached through too.
      func::CallOp::create(builder, op.getLoc(), write,
                           ValueRange{valueOp, addrPtr});
    }
    op.erase();
    return success();
  }

  LogicalResult lowerPoll(DmaPollForQueueSpaceOp op,
                          const AIETargetModel &targetModel,
                          func::FuncOp read) {
    CoreOp core = op->getParentOfType<CoreOp>();
    TileOp tile = core.getTileOp();
    std::optional<uint32_t> statusAddr = targetModel.getDmaStatusAddress(
        tile.getCol(), tile.getRow(), op.getChannel(), op.getDirection());
    if (!statusAddr)
      return op.emitOpError("this tile reports no DMA status register to read");
    uint32_t mask = targetModel.getDmaTaskQueueSizeMask();
    if (mask == 0)
      return op.emitOpError(
          "this tile reports no task-queue field in its DMA status");
    core.setProcessorBus(true);

    OpBuilder builder(op);
    Location loc = op.getLoc();
    Type i32 = builder.getI32Type();
    uint32_t localAddr = *statusAddr & ((1u << targetModel.getRowShift()) - 1);
    Value addrPtr = tilePointer(builder, loc, localAddr);
    Value maskOp = arith::ConstantOp::create(
        builder, loc, i32,
        builder.getI32IntegerAttr(static_cast<int32_t>(mask)));
    // The field counts the tasks the queue holds, so it reads the depth
    // exactly when the queue has no room left.
    uint32_t full = targetModel.getDmaTaskQueueDepth()
                    << llvm::countr_zero(mask);
    Value fullOp = arith::ConstantOp::create(
        builder, loc, i32,
        builder.getI32IntegerAttr(static_cast<int32_t>(full)));

    // Read the field until it falls below the depth. The core does nothing
    // else meanwhile.
    auto loop = scf::WhileOp::create(builder, loc, TypeRange{}, ValueRange{});
    OpBuilder::InsertionGuard guard(builder);
    Block *before = builder.createBlock(&loop.getBefore());
    builder.setInsertionPointToStart(before);
    Value status = func::CallOp::create(builder, loc, read, ValueRange{addrPtr})
                       .getResult(0);
    Value size = arith::AndIOp::create(builder, loc, status, maskOp);
    Value isFull = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         size, fullOp);
    scf::ConditionOp::create(builder, loc, isFull, ValueRange{});

    Block *after = builder.createBlock(&loop.getAfter());
    builder.setInsertionPointToStart(after);
    scf::YieldOp::create(builder, loc, ValueRange{});

    op.erase();
    return success();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    const AIETargetModel &targetModel = device.getTargetModel();

    SmallVector<DmaPushQueueOp> pushes;
    SmallVector<DmaPollForQueueSpaceOp> polls;
    device.walk([&](Operation *op) {
      if (auto push = dyn_cast<DmaPushQueueOp>(op))
        pushes.push_back(push);
      else if (auto poll = dyn_cast<DmaPollForQueueSpaceOp>(op))
        polls.push_back(poll);
    });
    if (pushes.empty() && polls.empty())
      return;

    std::optional<TileAccessIntrinsics> intrinsics =
        tileAccessIntrinsics(targetModel.getTargetArch());
    if (!intrinsics) {
      Operation *first = pushes.empty() ? polls.front().getOperation()
                                        : pushes.front().getOperation();
      first->emitOpError("this device has no access a core can make to its "
                         "own tile's registers");
      return signalPassFailure();
    }

    OpBuilder builder(device.getContext());
    Type i32 = builder.getI32Type();
    Type ptr = LLVM::LLVMPointerType::get(device.getContext());

    if (!polls.empty()) {
      func::FuncOp read = declare(device, intrinsics->read, {ptr}, {i32});
      for (DmaPollForQueueSpaceOp op : polls)
        if (failed(lowerPoll(op, targetModel, read)))
          return signalPassFailure();
    }
    if (!pushes.empty()) {
      func::FuncOp write =
          declare(device, intrinsics->write, {i32, ptr}, TypeRange{});
      for (DmaPushQueueOp op : pushes)
        if (failed(lowerPush(op, targetModel, write)))
          return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIELowerCoreIssuedConfigurationPass() {
  return std::make_unique<AIELowerCoreIssuedConfigurationPass>();
}
