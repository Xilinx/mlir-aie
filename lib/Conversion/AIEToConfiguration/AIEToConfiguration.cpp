//===- AIEToConfiguration.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Targets/AIERT.h"

#include "llvm/Support/Debug.h"
#include <llvm/ADT/APInt.h>

extern "C" {
#include "xaiengine/xaiegbl_defs.h"
// above needs to go first for u32, u64 typedefs
#include "xaiengine/xaie_txn.h"
}

#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#define DEBUG_TYPE "aie-convert-to-config"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// A TransactionBinaryOperation encapsulates an aie-rt XAie_TxnCmd struct and
// any additional metadata needed for custom operations that do not map cleanly
// onto the core command fields.
struct TransactionBinaryOperation {
  struct XAie_TxnCmd cmd = {};

  struct SyncPayload {
    int32_t column;
    int32_t row;
    int32_t direction;
    int32_t channel;
    int32_t columnCount;
    int32_t rowCount;
  };

  struct LoadPdiPayload {
    uint32_t id;
    uint32_t size;
    uint64_t address;
  };

  struct AddressPatchPayload {
    uint32_t action;
    uint32_t addr;
    int32_t argIdx;
    int32_t argPlus;
  };

  std::optional<SyncPayload> sync;
  std::optional<LoadPdiPayload> loadPdi;
  std::optional<AddressPatchPayload> addressPatch;

  TransactionBinaryOperation() = default;

  TransactionBinaryOperation(XAie_TxnOpcode opc, uint32_t mask, uint64_t addr,
                             uint32_t value, const uint8_t *data,
                             uint32_t size) {
    cmd.Opcode = opc;
    cmd.Mask = mask;
    cmd.RegOff = addr;
    cmd.Value = value;
    cmd.DataPtr = reinterpret_cast<uint64_t>(data);
    cmd.Size = size;
  }
};

constexpr size_t kTxnHeaderBytes = 16;

struct TxnPreemptHeader {
  uint8_t opcode;
  uint8_t level;
  uint16_t reserved;
};

struct TxnLoadPdiHeader {
  uint8_t opcode;
  uint8_t padding;
  uint16_t id;
  uint32_t size;
  uint64_t address;
};
} // namespace

// Parse a TXN binary blob. On success return the number of columns from the
// header and a vector of parsed operations. On failure return std::nullopt.
static std::optional<int>
parseTransactionBinary(const std::vector<uint8_t> &data,
                       std::vector<TransactionBinaryOperation> &ops) {

  if (data.size() < kTxnHeaderBytes) {
    llvm::errs() << "Transaction binary is too small for header\n";
    return std::nullopt;
  }

  uint32_t major = data[0];
  uint32_t minor = data[1];
  uint32_t num_cols = data[4];

  uint32_t num_ops, txn_size;
  std::memcpy(&num_ops, &data[8], 4);
  std::memcpy(&txn_size, &data[12], 4);

  LLVM_DEBUG(llvm::dbgs() << "Major: " << major << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Minor: " << minor << "\n");
  LLVM_DEBUG(llvm::dbgs() << "DevGen: " << data[2] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumRows: " << data[3] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumCols: " << num_cols << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumMemTileRows: " << data[5] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NumOps: " << num_ops << "\n");
  LLVM_DEBUG(llvm::dbgs() << "TxnSize: " << txn_size << " bytes\n");

  size_t i = kTxnHeaderBytes;

  auto requireBytes = [&](size_t offset, size_t length) -> bool {
    if (offset + length > data.size()) {
      llvm::errs() << "Transaction binary truncated while parsing opcode\n";
      return false;
    }
    return true;
  };

  auto read32 = [&](size_t offset) -> uint32_t {
    uint32_t value;
    std::memcpy(&value, data.data() + offset, sizeof(uint32_t));
    return value;
  };

  // Convert opcode from uint8 to a validated opcode byte
  auto convertOpcode = [](uint8_t opc) -> std::optional<uint8_t> {
    switch (opc) {
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_WRITE):
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_BLOCKWRITE):
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_MASKWRITE):
    case 0x6: // XAie_TxnOpcode::XAIE_IO_PREEMPT
    case 0x8: // XAie_TxnOpcode::XAIE_IO_LOAD_PDI
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT):
    case static_cast<uint8_t>(XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH):
      return opc;
    default:
      llvm::errs() << "Unhandled opcode: " << std::to_string(opc) << "\n";
      return std::nullopt;
    }
  };

  // Parse the binary blob. There are two versions supported, 0.1 and 1.0.
  // For both versions, build a list of TransactionBinaryOperation objects
  // representing the parsed operations.
  if (major == 0 && minor == 1) {
    while (i < data.size()) {
      auto maybeOpcode = convertOpcode(data[i]);
      if (!maybeOpcode)
        return std::nullopt;
      XAie_TxnOpcode opcode = static_cast<XAie_TxnOpcode>(*maybeOpcode);
      LLVM_DEBUG(llvm::dbgs() << "opcode: " + std::to_string(opcode) << "\n");

      TransactionBinaryOperation op;
      op.cmd.Opcode = opcode;

      switch (opcode) {
      case XAie_TxnOpcode::XAIE_IO_WRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: WRITE (0x00)\n");
        if (!requireBytes(i, 24))
          return std::nullopt;
        uint32_t addrLo = read32(i + 8);
        uint32_t addrHi = read32(i + 12);
        uint32_t value = read32(i + 16);
        uint32_t opSize = read32(i + 20);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        uint64_t addr = (static_cast<uint64_t>(addrHi) << 32) | addrLo;
        op.cmd.RegOff = addr;
        op.cmd.Value = value;
        op.cmd.Size = 0;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_BLOCKWRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: BLOCKWRITE (0x01)\n");
        if (!requireBytes(i, 16))
          return std::nullopt;
        uint32_t addr = read32(i + 8);
        uint32_t opSize = read32(i + 12);
        if (opSize < 16 || !requireBytes(i, opSize))
          return std::nullopt;
        const uint8_t *payload = data.data() + i + 16;
        uint32_t payloadBytes = opSize - 16;
        op.cmd.RegOff = addr;
        op.cmd.DataPtr = reinterpret_cast<uint64_t>(payload);
        op.cmd.Size = payloadBytes;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_MASKWRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: MASKWRITE (0x03)\n");
        if (!requireBytes(i, 28))
          return std::nullopt;
        uint32_t addrLo = read32(i + 8);
        uint32_t addrHi = read32(i + 12);
        uint32_t value = read32(i + 16);
        uint32_t mask = read32(i + 20);
        uint32_t opSize = read32(i + 24);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        uint64_t addr = (static_cast<uint64_t>(addrHi) << 32) | addrLo;
        op.cmd.RegOff = addr;
        op.cmd.Value = value;
        op.cmd.Mask = mask;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT: {
        uint32_t opSize = read32(i + 4);
        if (opSize < 16 || !requireBytes(i, opSize))
          return std::nullopt;
        uint32_t descriptor = read32(i + 8);
        uint32_t config = read32(i + 12);
        TransactionBinaryOperation::SyncPayload payload{
            /*column=*/static_cast<int32_t>((descriptor >> 16) & 0xff),
            /*row=*/static_cast<int32_t>((descriptor >> 8) & 0xff),
            /*direction=*/static_cast<int32_t>(descriptor & 0xff),
            /*channel=*/static_cast<int32_t>((config >> 24) & 0xff),
            /*columnCount=*/static_cast<int32_t>((config >> 16) & 0xff),
            /*rowCount=*/static_cast<int32_t>((config >> 8) & 0xff)};
        op.sync = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case 0x8: { // XAie_TxnOpcode::XAIE_IO_LOAD_PDI
        LLVM_DEBUG(llvm::dbgs() << "opcode: LOAD_PDI (0x08)\n");
        constexpr size_t opSize = sizeof(TxnLoadPdiHeader);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        TxnLoadPdiHeader header;
        std::memcpy(&header, data.data() + i, opSize);
        TransactionBinaryOperation::LoadPdiPayload payload{
            header.id, header.size, header.address};
        op.loadPdi = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH: {
        uint32_t opSize = read32(i + 4);
        if (opSize < 44 || !requireBytes(i, opSize))
          return std::nullopt;
        uint32_t action = read32(i + 20);
        uint32_t addr = read32(i + 24);
        int32_t argIdx = static_cast<int32_t>(read32(i + 32));
        int32_t argPlus = static_cast<int32_t>(read32(i + 40));
        TransactionBinaryOperation::AddressPatchPayload payload{
            action, addr, argIdx, argPlus};
        op.addressPatch = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case 0x6: { // XAie_TxnOpcode::XAIE_IO_PREEMPT
        LLVM_DEBUG(llvm::dbgs() << "opcode: PREEMPT (0x06)\n");
        constexpr size_t opSize = sizeof(TxnPreemptHeader);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        auto header =
            reinterpret_cast<const TxnPreemptHeader *>(data.data() + i);
        op.cmd.Value = header->level;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      default:
        llvm::errs() << "Unhandled opcode: " << std::to_string(opcode)
                     << " for v0.1 transaction\n";
        return std::nullopt;
      }

      ops.push_back(std::move(op));
    }
  } else if (major == 1 && minor == 0) {
    while (i < data.size()) {
      auto maybeOpcode = convertOpcode(data[i]);
      if (!maybeOpcode)
        return std::nullopt;
      XAie_TxnOpcode opcode = static_cast<XAie_TxnOpcode>(*maybeOpcode);
      LLVM_DEBUG(llvm::dbgs() << "opcode: " + std::to_string(opcode) << "\n");

      TransactionBinaryOperation op;
      op.cmd.Opcode = opcode;

      switch (opcode) {
      case XAie_TxnOpcode::XAIE_IO_WRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: WRITE (0x00)\n");
        if (!requireBytes(i, 12))
          return std::nullopt;
        uint32_t addr = read32(i + 4);
        uint32_t value = read32(i + 8);
        op.cmd.RegOff = addr;
        op.cmd.Value = value;
        op.cmd.Size = 0;
        i += 12;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_BLOCKWRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: BLOCKWRITE (0x01)\n");
        if (!requireBytes(i, 12))
          return std::nullopt;
        uint32_t addr = read32(i + 4);
        uint32_t opSize = read32(i + 8);
        if (opSize < 12 || !requireBytes(i, opSize))
          return std::nullopt;
        const uint8_t *payload = data.data() + i + 12;
        uint32_t payloadBytes = opSize - 12;
        op.cmd.RegOff = addr;
        op.cmd.DataPtr = reinterpret_cast<uint64_t>(payload);
        op.cmd.Size = payloadBytes;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_MASKWRITE: {
        LLVM_DEBUG(llvm::dbgs() << "opcode: MASKWRITE (0x03)\n");
        if (!requireBytes(i, 16))
          return std::nullopt;
        uint32_t addr = read32(i + 4);
        uint32_t value = read32(i + 8);
        uint32_t mask = read32(i + 12);
        op.cmd.RegOff = addr;
        op.cmd.Value = value;
        op.cmd.Mask = mask;
        op.cmd.Size = 0;
        i += 16;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT: {
        uint32_t opSize = read32(i + 4);
        if (opSize < 16 || !requireBytes(i, opSize))
          return std::nullopt;
        uint32_t descriptor = read32(i + 8);
        uint32_t config = read32(i + 12);
        TransactionBinaryOperation::SyncPayload payload{
            /*column=*/static_cast<int32_t>((descriptor >> 16) & 0xff),
            /*row=*/static_cast<int32_t>((descriptor >> 8) & 0xff),
            /*direction=*/static_cast<int32_t>(descriptor & 0xff),
            /*channel=*/static_cast<int32_t>((config >> 24) & 0xff),
            /*columnCount=*/static_cast<int32_t>((config >> 16) & 0xff),
            /*rowCount=*/static_cast<int32_t>((config >> 8) & 0xff)};
        op.sync = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case 0x8: { // XAie_TxnOpcode::XAIE_IO_LOAD_PDI
        LLVM_DEBUG(llvm::dbgs() << "opcode: LOAD_PDI (0x08)\n");
        constexpr size_t opSize = sizeof(TxnLoadPdiHeader);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        TxnLoadPdiHeader header;
        std::memcpy(&header, data.data() + i, opSize);
        TransactionBinaryOperation::LoadPdiPayload payload{
            header.id, header.size, header.address};
        op.loadPdi = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH: {
        uint32_t opSize = read32(i + 4);
        if (opSize < 44 || !requireBytes(i, opSize))
          return std::nullopt;
        uint32_t action = read32(i + 20);
        uint32_t addr = read32(i + 24);
        int32_t argIdx = static_cast<int32_t>(read32(i + 32));
        int32_t argPlus = static_cast<int32_t>(read32(i + 40));
        TransactionBinaryOperation::AddressPatchPayload payload{
            action, addr, argIdx, argPlus};
        op.addressPatch = payload;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      case 0x6: { // XAie_TxnOpcode::XAIE_IO_PREEMPT
        LLVM_DEBUG(llvm::dbgs() << "opcode: PREEMPT (0x06)\n");
        constexpr size_t opSize = sizeof(TxnPreemptHeader);
        if (!requireBytes(i, opSize))
          return std::nullopt;
        auto header =
            reinterpret_cast<const TxnPreemptHeader *>(data.data() + i);
        op.cmd.Value = header->level;
        op.cmd.Size = opSize;
        i += opSize;
        break;
      }
      default:
        llvm::errs() << "Unhandled opcode: " << std::to_string(opcode)
                     << " for v1.0 transaction\n";
        return std::nullopt;
      }

      ops.push_back(std::move(op));
    }
  } else {
    llvm::errs() << "Unsupported TXN binary version: " << major << "." << minor
                 << "\n";
    return std::nullopt;
  }

  return num_cols;
}

static LogicalResult generateTransactions(AIERTControl &ctl,
                                          const StringRef workDirPath,
                                          DeviceOp &targetOp, bool aieSim,
                                          bool enableElfs, bool enableInit,
                                          bool enableCores) {
  if (enableElfs && !targetOp.getOps<CoreOp>().empty() &&
      failed(ctl.addAieElfs(targetOp, workDirPath, aieSim)))
    return failure();
  if (enableInit && failed(ctl.addInitConfig(targetOp)))
    return failure();
  if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
      failed(ctl.addCoreEnable(targetOp)))
    return failure();
  return success();
}

// Translate vector of TransactionBinaryOperation to a sequence of transaction
// ops (npu.write32, npu.maskwrite32, npu.blockwrite).
static LogicalResult
emitTransactionOps(OpBuilder &builder,
                   std::vector<TransactionBinaryOperation> &operations,
                   std::vector<memref::GlobalOp> &global_data) {

  auto loc = builder.getUnknownLoc();

  // create the txn ops
  for (auto [op, payload] : llvm::zip(operations, global_data)) {

    if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_WRITE) {
      builder.create<AIEX::NpuWrite32Op>(loc, op.cmd.RegOff, op.cmd.Value,
                                         nullptr, nullptr, nullptr);
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
      auto memref = builder.create<memref::GetGlobalOp>(loc, payload.getType(),
                                                        payload.getName());
      builder.create<AIEX::NpuBlockWriteOp>(
          loc, builder.getUI32IntegerAttr(op.cmd.RegOff), memref.getResult(),
          nullptr, nullptr, nullptr);
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
      builder.create<AIEX::NpuMaskWrite32Op>(loc, op.cmd.RegOff, op.cmd.Value,
                                             op.cmd.Mask, nullptr, nullptr,
                                             nullptr);
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_TCT) {
      if (!op.sync) {
        llvm::errs() << "Missing sync payload while emitting transaction\n";
        return failure();
      }
      const TransactionBinaryOperation::SyncPayload &sync = *op.sync;
      builder.create<AIEX::NpuSyncOp>(
          loc, builder.getI32IntegerAttr(sync.column),
          builder.getI32IntegerAttr(sync.row),
          builder.getI32IntegerAttr(sync.direction),
          builder.getI32IntegerAttr(sync.channel),
          builder.getI32IntegerAttr(sync.columnCount),
          builder.getI32IntegerAttr(sync.rowCount));
    } else if (op.cmd.Opcode == 0x8 /* XAie_TxnOpcode::XAIE_IO_LOAD_PDI */) {
      if (!op.loadPdi) {
        llvm::errs() << "Missing load_pdi payload while emitting transaction\n";
        return failure();
      }
      const TransactionBinaryOperation::LoadPdiPayload &payloadInfo =
          *op.loadPdi;
      auto idAttr =
          builder.getI32IntegerAttr(static_cast<int32_t>(payloadInfo.id));
      IntegerAttr sizeAttr =
          builder.getI32IntegerAttr(static_cast<int32_t>(payloadInfo.size));

      auto ui64Ty =
          IntegerType::get(builder.getContext(), 64, IntegerType::Unsigned);
      IntegerAttr addressAttr =
          IntegerAttr::get(ui64Ty, llvm::APInt(64, payloadInfo.address));

      builder.create<AIEX::NpuLoadPdiOp>(loc, nullptr, idAttr, sizeAttr,
                                         addressAttr);
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_DDR_PATCH) {
      if (!op.addressPatch) {
        llvm::errs()
            << "Missing address_patch payload while emitting transaction\n";
        return failure();
      }
      const TransactionBinaryOperation::AddressPatchPayload &patch =
          *op.addressPatch;
      builder.create<AIEX::NpuAddressPatchOp>(
          loc, builder.getUI32IntegerAttr(patch.addr),
          builder.getI32IntegerAttr(patch.argIdx),
          builder.getI32IntegerAttr(patch.argPlus));
    } else if (op.cmd.Opcode == 0x6 /*  XAie_TxnOpcode::XAIE_IO_PREEMPT */) {
      auto ui8Ty =
          IntegerType::get(builder.getContext(), 8, IntegerType::Unsigned);
      auto levelAttr = IntegerAttr::get(ui8Ty, llvm::APInt(8, op.cmd.Value));
      builder.create<AIEX::NpuPreemptOp>(loc, levelAttr);
    } else {
      llvm::errs() << "Unhandled txn opcode: " << op.cmd.Opcode << "\n";
      return failure();
    }
  }
  return success();
}

// Translate vector of TransactionBinaryOperation to a sequence of control
// packet ops.
static LogicalResult
emitControlPacketOps(OpBuilder &builder,
                     std::vector<TransactionBinaryOperation> &operations,
                     std::vector<memref::GlobalOp> &global_data) {

  auto loc = builder.getUnknownLoc();
  auto ctx = builder.getContext();

  // create the control packet ops
  for (auto [op, payload] : llvm::zip(operations, global_data)) {

    if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_WRITE) {
      builder.create<AIEX::NpuControlPacketOp>(
          loc, builder.getUI32IntegerAttr(op.cmd.RegOff), nullptr,
          /*opcode*/ builder.getI32IntegerAttr(0),
          /*stream_id*/ builder.getI32IntegerAttr(0),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(op.cmd.Value)));
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
      if (!payload.getInitialValue())
        continue;
      auto blockWriteData =
          dyn_cast<DenseIntElementsAttr>(*payload.getInitialValue());
      if (!blockWriteData) {
        payload.emitError(
            "Global symbol initial value is not a dense int array");
        break;
      }
      auto blockWriteDataValues = blockWriteData.getValues<int32_t>();
      // Split block write data into beats of 4 or less, in int32_t.
      int currAddr = op.cmd.RegOff;
      for (size_t i = 0; i < blockWriteDataValues.size(); i += 4) {
        auto last = std::min(blockWriteDataValues.size(), i + 4);
        SmallVector<int32_t> splitData =
            SmallVector<int32_t>(blockWriteDataValues.begin() + i,
                                 blockWriteDataValues.begin() + last);
        builder.create<AIEX::NpuControlPacketOp>(
            loc, builder.getUI32IntegerAttr(currAddr), nullptr,
            /*opcode*/ builder.getI32IntegerAttr(0),
            /*stream_id*/ builder.getI32IntegerAttr(0),
            DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(splitData)));
        currAddr += splitData.size() * sizeof(int32_t);
      }

    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
      builder.create<AIEX::NpuControlPacketOp>(
          loc, builder.getUI32IntegerAttr(op.cmd.RegOff), nullptr,
          /*opcode*/ builder.getI32IntegerAttr(0),
          /*stream_id*/ builder.getI32IntegerAttr(0),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(op.cmd.Value)));
    } else {
      llvm::errs() << "Unhandled txn opcode: " << op.cmd.Opcode << "\n";
      return failure();
    }
  }
  return success();
}

// Perform bitwise or on consecutive control packets operating on the same
// address, to resolve the lack of mask write in control packets.
LogicalResult orConsecutiveWritesOnSameAddr(Block *body) {
  SmallVector<AIEX::NpuControlPacketOp> ctrlPktOps;
  body->walk(
      [&](AIEX::NpuControlPacketOp cpOp) { ctrlPktOps.push_back(cpOp); });
  if (ctrlPktOps.empty())
    return success();

  SmallVector<Operation *> erased;
  int addrBuffer = ctrlPktOps[0].getAddress();
  AIEX::NpuControlPacketOp ctrlPktBuffer = ctrlPktOps[0];
  for (size_t i = 1; i < ctrlPktOps.size(); i++) {
    int currentAddrBuffer = ctrlPktOps[i].getAddress();
    if (addrBuffer != currentAddrBuffer) {
      addrBuffer = currentAddrBuffer;
      ctrlPktBuffer = ctrlPktOps[i];
      continue;
    }
    auto bufferedData = ctrlPktBuffer.getData().value();
    auto currentData = ctrlPktOps[i].getData().value();
    SmallVector<int> newData;
    for (unsigned j = 0; j < std::max(bufferedData.size(), currentData.size());
         j++) {
      if (j < std::min(bufferedData.size(), currentData.size())) {
        newData.push_back(bufferedData[j] | currentData[j]);
        continue;
      }
      newData.push_back(j < bufferedData.size() ? bufferedData[j]
                                                : currentData[j]);
    }
    ctrlPktBuffer.getProperties().data = DenseI32ArrayAttr::get(
        ctrlPktBuffer->getContext(), ArrayRef<int>{newData});
    erased.push_back(ctrlPktOps[i]);
  }

  for (auto e : erased)
    e->erase();

  return success();
}

// an enum to represent the output type of the transaction binary
enum OutputType {
  Transaction,
  ControlPacket,
};

static LogicalResult convertTransactionOpsToMLIR(
    OpBuilder builder, AIE::DeviceOp device, OutputType outputType,
    std::vector<TransactionBinaryOperation> &operations) {

  auto loc = builder.getUnknownLoc();

  // for each blockwrite in the binary, create a GlobalOp with the data
  std::vector<memref::GlobalOp> global_data;
  for (auto &op : operations) {
    if (op.cmd.Opcode != XAIE_IO_BLOCKWRITE) {
      global_data.push_back(nullptr);
      continue;
    }
    uint32_t size = op.cmd.Size / 4;
    const uint32_t *d = reinterpret_cast<const uint32_t *>(op.cmd.DataPtr);
    std::vector<uint32_t> data32(d, d + size);

    int id = 0;
    std::string name = "blockwrite_data";
    while (device.lookupSymbol(name))
      name = "blockwrite_data_" + std::to_string(id++);

    MemRefType memrefType = MemRefType::get({size}, builder.getI32Type());
    TensorType tensorType = RankedTensorType::get({size}, builder.getI32Type());
    auto global = builder.create<memref::GlobalOp>(
        loc, name, builder.getStringAttr("private"), memrefType,
        DenseElementsAttr::get<uint32_t>(tensorType, data32), true, nullptr);
    global_data.push_back(global);
  }

  // search for npu.configure ops in runtime sequences by walking the device
  // and collect them in a vector.
  SmallVector<AIEX::NpuConfigureOp> configureOps;
  device.walk([&](AIEX::NpuConfigureOp op) { configureOps.push_back(op); });

  if (configureOps.empty()) {

    // create aiex.runtime_sequence
    int id = 0;
    std::string seq_name = "configure";
    while (device.lookupSymbol(seq_name))
      seq_name = "configure" + std::to_string(id++);
    StringAttr seq_sym_name = builder.getStringAttr(seq_name);
    auto seq = builder.create<AIEX::RuntimeSequenceOp>(loc, seq_sym_name);
    seq.getBody().push_back(new Block);

    builder.setInsertionPointToStart(&seq.getBody().front());
  } else {
    builder.setInsertionPoint(configureOps.front());
  }

  // create the txn ops
  if (outputType == OutputType::Transaction) {
    if (failed(emitTransactionOps(builder, operations, global_data)))
      return failure();
  } else if (outputType == OutputType::ControlPacket) {
    if (failed(emitControlPacketOps(builder, operations, global_data)))
      return failure();
    // resolve mask writes; control packet doesn't natively support mask write.
    if (failed(orConsecutiveWritesOnSameAddr(builder.getBlock())))
      return failure();
  } else {
    llvm_unreachable("bad output type");
  }

  if (!configureOps.empty())
    configureOps.front().erase();

  return success();
}

// Convert (disassemble) a transaction binary to MLIR. On success return a new
// ModuleOp containing a DeviceOp containing a runtime sequence with the
// transaction binary encoded as a sequence of npu.write32, npu.maskwrite32 and
// npu.blockwrite operations. On failure return std::nullopt.
std::optional<mlir::ModuleOp>
xilinx::AIE::convertTransactionBinaryToMLIR(mlir::MLIRContext *ctx,
                                            std::vector<uint8_t> &binary) {

  // parse the binary
  std::vector<TransactionBinaryOperation> operations;
  auto c = parseTransactionBinary(binary, operations);
  if (!c) {
    llvm::errs() << "Failed to parse binary\n";
    return std::nullopt;
  }
  int columns = *c;

  auto loc = mlir::UnknownLoc::get(ctx);

  // create a new ModuleOp and set the insertion point
  auto module = ModuleOp::create(loc);
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());

  // create aie.device
  std::vector<AIEDevice> devices{AIEDevice::npu1_1col, AIEDevice::npu1_2col,
                                 AIEDevice::npu1_3col, AIEDevice::npu1};
  auto device = builder.create<DeviceOp>(loc, devices[columns - 1],
                                         StringAttr::get(builder.getContext()));
  device.getRegion().emplaceBlock();
  DeviceOp::ensureTerminator(device.getBodyRegion(), builder, loc);
  builder.setInsertionPointToStart(device.getBody());

  // convert the parsed ops to MLIR
  if (failed(convertTransactionOpsToMLIR(builder, device,
                                         OutputType::Transaction, operations)))
    return std::nullopt;

  return module;
}

static LogicalResult convertAIEToConfiguration(AIE::DeviceOp device,
                                               StringRef clElfDir,
                                               OutputType outputType) {

  const AIETargetModel &targetModel =
      (const AIETargetModel &)device.getTargetModel();

  if (!targetModel.hasProperty(AIETargetModel::IsNPU))
    return failure();

  bool aieSim = false;
  bool xaieDebug = false;

  AIERTControl ctl(targetModel);
  if (failed(ctl.setIOBackend(aieSim, xaieDebug)))
    return failure();

  // start collecting transations
  ctl.startTransaction();

  bool generateElfs = clElfDir.size() > 0;
  if (failed(generateTransactions(ctl, clElfDir, device, aieSim, generateElfs,
                                  true, true)))
    return failure();

  // Export the transactions to a binary buffer
  std::vector<uint8_t> txn_data = ctl.exportSerializedTransaction();

  // parse the binary data
  std::vector<TransactionBinaryOperation> operations;
  if (!parseTransactionBinary(txn_data, operations)) {
    llvm::errs() << "Failed to parse binary\n";
    return failure();
  }

  OpBuilder builder(device.getBodyRegion());

  // convert the parsed ops to MLIR
  if (failed(
          convertTransactionOpsToMLIR(builder, device, outputType, operations)))
    return failure();

  return success();
}

namespace {

template <typename BaseClass, OutputType MyOutputType>
struct ConvertAIEToConfigurationPass : BaseClass {
  std::string &ref_clElfDir;
  std::string &ref_clDeviceName;
  ConvertAIEToConfigurationPass(std::string &clElfDir,
                                std::string &clDeviceName)
      : ref_clElfDir(clElfDir), ref_clDeviceName(clDeviceName) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    AIE::DeviceOp deviceOp = BaseClass::getOperation();
    if (!ref_clDeviceName.empty() &&
        deviceOp.getSymName() != ref_clDeviceName) {
      return;
    }
    if (failed(
            convertAIEToConfiguration(deviceOp, ref_clElfDir, MyOutputType))) {
      return BaseClass::signalPassFailure();
    }
  }
};

struct ConvertAIEToTransactionPass
    : ConvertAIEToConfigurationPass<
          ConvertAIEToTransactionBase<ConvertAIEToTransactionPass>,
          OutputType::Transaction> {
  ConvertAIEToTransactionPass()
      : ConvertAIEToConfigurationPass<
            ConvertAIEToTransactionBase<ConvertAIEToTransactionPass>,
            OutputType::Transaction>(clElfDir, clDeviceName) {}
};

struct ConvertAIEToControlPacketsPass
    : ConvertAIEToConfigurationPass<
          ConvertAIEToControlPacketsBase<ConvertAIEToControlPacketsPass>,
          OutputType::ControlPacket> {
  ConvertAIEToControlPacketsPass()
      : ConvertAIEToConfigurationPass<
            ConvertAIEToControlPacketsBase<ConvertAIEToControlPacketsPass>,
            OutputType::ControlPacket>(clElfDir, clDeviceName) {}
};

} // end anonymous namespace

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
xilinx::AIE::createConvertAIEToTransactionPass() {
  return std::make_unique<ConvertAIEToTransactionPass>();
}

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
xilinx::AIE::createConvertAIEToControlPacketsPass() {
  return std::make_unique<ConvertAIEToControlPacketsPass>();
}
