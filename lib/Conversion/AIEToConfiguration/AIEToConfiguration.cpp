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

extern "C" {
#include "xaiengine/xaiegbl_defs.h"
// above needs to go first for u32, u64 typedefs
#include "xaiengine/xaie_txn.h"
}

#include <vector>

#define DEBUG_TYPE "aie-convert-to-config"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// An TransactionBinaryOperation encapulates an aie-rt TnxCmd struct
struct TransactionBinaryOperation {
  struct XAie_TxnCmd cmd;
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
} // namespace

// Parse a TXN binary blob. On success return the number of columns from the
// header and a vector of parsed operations. On failure return std::nullopt.
static std::optional<int>
parseTransactionBinary(const std::vector<uint8_t> &data,
                       std::vector<TransactionBinaryOperation> &ops) {

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

  size_t i = 16;

  // Convert opcode from uint8 to enum
  auto convertOpcode = [](uint8_t opc) {
    switch (opc) {
    case 0:
      return XAie_TxnOpcode::XAIE_IO_WRITE;
    case 1:
      return XAie_TxnOpcode::XAIE_IO_BLOCKWRITE;
    case 3:
      return XAie_TxnOpcode::XAIE_IO_MASKWRITE;
    default:
      llvm::errs() << "Unhandled opcode: " << std::to_string(opc) << "\n";
      return XAie_TxnOpcode::XAIE_IO_CUSTOM_OP_MAX;
    }
  };

  // Parse the binary blob. There are two versions supported, 0.1 and 1.0.
  // For both versions, build a list of TransactionBinaryOperation objects
  // representing the parsed operations.
  if (major == 0 && minor == 1) {
    while (i < data.size()) {

      XAie_TxnOpcode opc = convertOpcode(data[i]);
      LLVM_DEBUG(llvm::dbgs() << "opcode: " + std::to_string(opc) + "\n");

      uint64_t addr = 0;
      uint32_t value = 0;
      uint32_t size = 0;
      uint32_t mask = 0;
      const uint8_t *data_ptr = nullptr;

      if (opc == XAie_TxnOpcode::XAIE_IO_WRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: WRITE (0x00)\n");
        uint32_t addr0, addr1;
        std::memcpy(&addr0, &data[i + 8], 4);
        std::memcpy(&addr1, &data[i + 12], 4);
        std::memcpy(&value, &data[i + 16], 4);
        std::memcpy(&size, &data[i + 20], 4);
        addr = static_cast<uint64_t>(addr1) << 32 | addr0;
        i += size;
      } else if (opc == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: BLOCKWRITE (0x01)\n");
        std::memcpy(&addr, &data[i + 8], 4);
        std::memcpy(&size, &data[i + 12], 4);
        data_ptr = data.data() + i + 16;
        i += size;
        size = size - 16;
      } else if (opc == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: MASKWRITE (0x03)\n");
        uint32_t addr0, addr1;
        std::memcpy(&addr0, &data[i + 8], 4);
        std::memcpy(&addr1, &data[i + 12], 4);
        std::memcpy(&value, &data[i + 16], 4);
        std::memcpy(&mask, &data[i + 20], 4);
        std::memcpy(&size, &data[i + 24], 4);
        addr = static_cast<uint64_t>(addr1) << 32 | addr0;
        i += size;
      } else {
        llvm::errs() << "Unhandled opcode: " << std::to_string(opc) << "\n";
        return std::nullopt;
      }
      ops.emplace_back(opc, mask, addr, value, data_ptr, size);
      LLVM_DEBUG(llvm::dbgs() << "addr: " << addr << "\n");
      LLVM_DEBUG(llvm::dbgs() << "value: " << value << "\n");
      LLVM_DEBUG(llvm::dbgs() << "size: " << size << "\n");
      LLVM_DEBUG(llvm::dbgs() << "mask: " << mask << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "data: " << reinterpret_cast<uintptr_t>(data_ptr) << "\n");
    }
  } else if (major == 1 && minor == 0) {
    while (i < data.size()) {

      XAie_TxnOpcode opc = convertOpcode(data[i]);
      LLVM_DEBUG(llvm::dbgs() << "opcode: " + std::to_string(opc) + "\n");

      uint64_t addr = 0;
      uint32_t value = 0;
      uint32_t size = 0;
      uint32_t mask = 0;
      const uint8_t *data_ptr = nullptr;

      if (opc == XAie_TxnOpcode::XAIE_IO_WRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: WRITE (0x00)\n");
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        i += 12;
      } else if (opc == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: BLOCKWRITE (0x01)\n");
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&size, &data[i + 8], 4);
        data_ptr = data.data() + i + 12;
        i += size;
        size = size - 12;
      } else if (opc == XAie_TxnOpcode::XAIE_IO_MASKWRITE) {
        LLVM_DEBUG(llvm::dbgs() << "opcode: MASKWRITE (0x03)\n");
        std::memcpy(&addr, &data[i + 4], 4);
        std::memcpy(&value, &data[i + 8], 4);
        std::memcpy(&mask, &data[i + 12], 4);
        i += 16;
      } else {
        llvm::errs() << "Unhandled opcode: " << std::to_string(opc) << "\n";
        return std::nullopt;
      }
      LLVM_DEBUG(llvm::dbgs() << "addr: " << addr << "\n");
      LLVM_DEBUG(llvm::dbgs() << "value: " << value << "\n");
      LLVM_DEBUG(llvm::dbgs() << "size: " << size << "\n");
      LLVM_DEBUG(llvm::dbgs() << "mask: " << mask << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "data: " << reinterpret_cast<uintptr_t>(data_ptr) << "\n");
      ops.emplace_back(opc, mask, addr, value, data_ptr, size);
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

  // search for aiex.configure ops in runtime sequences by walking the device
  // and collect them in a vector. If there are none, create a new runtime
  // sequence. Otherwise assume the insertion point is the first aiex.configure
  // op.
  SmallVector<AIEX::ConfigureOp> configureOps;
  device.walk([&](AIEX::ConfigureOp op) { configureOps.push_back(op); });

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

  if (!configureOps.empty()) {
    // splice the body into the current insertion point
    builder.getBlock()->getOperations().splice(
        builder.getInsertionPoint(),
        configureOps.front().getBody().front().getOperations());
    configureOps.front().erase();
  }
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
  auto device = builder.create<DeviceOp>(
      loc, devices[columns - 1], StringAttr::get(builder.getContext(), "main"));
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

struct ConvertAIEToTransactionPass
    : ConvertAIEToTransactionBase<ConvertAIEToTransactionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, AIEX::AIEXDialect>();
  }
  void runOnOperation() override {
    if (failed(convertAIEToConfiguration(getOperation(), clElfDir,
                                         OutputType::Transaction)))
      return signalPassFailure();
  }
};

struct ConvertAIEToControlPacketsPass
    : public ConvertAIEToControlPacketsBase<ConvertAIEToControlPacketsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, AIEX::AIEXDialect>();
  }
  void runOnOperation() override {
    if (failed(convertAIEToConfiguration(getOperation(), clElfDir,
                                         OutputType::ControlPacket)))
      return signalPassFailure();
  }
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
