//===- AIETargetCDODirect.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIERT.h"
#include "aie/Targets/AIETargets.h"
extern "C" {
#include "cdo-driver/cdo_driver.h"
}

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#ifndef NDEBUG
#define XAIE_DEBUG
#endif

extern "C" {
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaiegbl.h"
}

#define DEBUG_TYPE "aie-generate-cdo"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (cdoDebug)
    EnAXIdebug();
  setEndianness(endianness);
};

static LogicalResult
generateCDOBinary(const StringRef outputPath,
                  const std::function<LogicalResult()> &cb) {

  // TODO(newling): Get bootgen team to remove print statement in this function.
  startCDOFileStream(outputPath.str().c_str());
  FileHeader();
  // Never generate a completely empty CDO file.  If the file only contains a
  // header, then bootgen flags it as invalid.
  insertNoOpCommand(4);
  if (failed(cb()))
    return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

static LogicalResult generateCDOBinariesSeparately(AIERTControl &ctl,
                                                   const StringRef workDirPath,
                                                   DeviceOp &targetOp,
                                                   bool aieSim,
                                                   bool enableCores) {
  auto ps = std::filesystem::path::preferred_separator;

  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_elfs.bin")
              .str(),
          [&ctl, &targetOp, &workDirPath, &aieSim] {
            return ctl.addAieElfs(targetOp, workDirPath, aieSim);
          })))
    return failure();

  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_init.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addInitConfig(targetOp); })))
    return failure();

  if (enableCores &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_enable.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addCoreEnable(targetOp); })))
    return failure();

  return success();
}

static LogicalResult generateCDOUnified(AIERTControl &ctl,
                                        const StringRef workDirPath,
                                        DeviceOp &targetOp, bool aieSim,
                                        bool enableCores) {
  auto ps = std::filesystem::path::preferred_separator;

  return generateCDOBinary(
      (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo.bin").str(),
      [&ctl, &targetOp, &workDirPath, &aieSim, &enableCores] {
        if (!targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addAieElfs(targetOp, workDirPath, aieSim)))
          return failure();
        if (failed(ctl.addInitConfig(targetOp)))
          return failure();
        if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addCoreEnable(targetOp)))
          return failure();
        return success();
      });
}

static LogicalResult
translateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                     byte_ordering endianness, bool emitUnified, bool cdoDebug,
                     bool aieSim, bool xaieDebug, bool enableCores) {

  auto devOps = m.getOps<DeviceOp>();
  assert(llvm::range_size(devOps) == 1 &&
         "only exactly 1 device op supported.");
  DeviceOp targetOp = *devOps.begin();
  const BaseNPUTargetModel &targetModel =
      (const BaseNPUTargetModel &)targetOp.getTargetModel();

  // things like XAIE_MEM_TILE_ROW_START and the missing
  // shim dma on tile (0,0) are hard-coded assumptions about NPU...
  assert(targetModel.isNPU() && "Only NPU currently supported");

  AIERTControl ctl(targetModel);
  if (failed(ctl.setIOBackend(aieSim, xaieDebug)))
    return failure();
  initializeCDOGenerator(endianness, cdoDebug);

  auto result = [&]() {
    if (emitUnified) {
      return generateCDOUnified(ctl, workDirPath, targetOp, aieSim,
                                enableCores);
    }
    return generateCDOBinariesSeparately(ctl, workDirPath, targetOp, aieSim,
                                         enableCores);
  }();
  return result;
}

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

static LogicalResult generateTxn(AIERTControl &ctl, const StringRef workDirPath,
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

static LogicalResult translateToTxn(ModuleOp m, std::vector<uint8_t> &output,
                                    llvm::StringRef workDirPath, bool aieSim,
                                    bool xaieDebug, bool enableCores) {

  auto devOps = m.getOps<DeviceOp>();
  if (llvm::range_size(devOps) > 1)
    return m.emitError("only exactly 1 device op supported.");

  DeviceOp targetOp = *devOps.begin();
  const BaseNPUTargetModel &targetModel =
      (const BaseNPUTargetModel &)targetOp.getTargetModel();

  if (!targetModel.isNPU())
    return failure();

  AIERTControl ctl(targetModel);
  if (failed(ctl.setIOBackend(aieSim, xaieDebug)))
    return failure();

  // start collecting transations
  XAie_StartTransaction(&ctl.devInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);

  auto result =
      generateTxn(ctl, workDirPath, targetOp, aieSim, true, true, true);
  if (failed(result))
    return result;

  // Export the transactions to a buffer
  uint8_t *txn_ptr = XAie_ExportSerializedTransaction(&ctl.devInst, 0, 0);
  XAie_TxnHeader *hdr = (XAie_TxnHeader *)txn_ptr;
  std::vector<uint8_t> txn_data(txn_ptr, txn_ptr + hdr->TxnSize);
  output.swap(txn_data);

  return success();
}

LogicalResult xilinx::AIE::AIETranslateToCDODirect(
    ModuleOp m, llvm::StringRef workDirPath, bool bigEndian, bool emitUnified,
    bool cdoDebug, bool aieSim, bool xaieDebug, bool enableCores) {
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  return translateToCDODirect(m, workDirPath, endianness, emitUnified, cdoDebug,
                              aieSim, xaieDebug, enableCores);
}

std::optional<mlir::ModuleOp>
xilinx::AIE::AIETranslateBinaryToCtrlpkt(mlir::MLIRContext *ctx,
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
                                 AIEDevice::npu1_3col, AIEDevice::npu1_4col,
                                 AIEDevice::npu1};
  auto device = builder.create<DeviceOp>(loc, devices[columns - 1]);
  device.getRegion().emplaceBlock();
  builder.setInsertionPointToStart(device.getBody());

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

  // create aiex.runtime_sequence
  auto seq = builder.create<AIEX::RuntimeSequenceOp>(loc, nullptr);
  seq.getBody().push_back(new Block);

  // create the txn ops
  builder.setInsertionPointToStart(&seq.getBody().front());
  for (auto p : llvm::zip(operations, global_data)) {
    auto op = std::get<0>(p);
    memref::GlobalOp payload = std::get<1>(p);

    if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_WRITE) {
      builder.create<AIEX::NpuControlPacketOp>(
          loc, builder.getUI32IntegerAttr(op.cmd.RegOff), nullptr,
          /*opcode*/ builder.getI32IntegerAttr(0),
          /*stream_id*/ builder.getI32IntegerAttr(0),
          DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>(op.cmd.Value)));
    } else if (op.cmd.Opcode == XAie_TxnOpcode::XAIE_IO_BLOCKWRITE) {
      if (!std::get<1>(p).getInitialValue())
        continue;
      auto blockWriteData =
          dyn_cast<DenseIntElementsAttr>(*std::get<1>(p).getInitialValue());
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
      return std::nullopt;
    }
  }

  return module;
}

LogicalResult xilinx::AIE::AIETranslateToControlPackets(
    ModuleOp m, llvm::raw_ostream &output, llvm::StringRef workDirPath,
    bool outputBinary, bool enableSim, bool xaieDebug, bool enableCores) {
  std::vector<uint8_t> bin;
  auto result =
      translateToTxn(m, bin, workDirPath, enableSim, xaieDebug, enableCores);
  if (failed(result))
    return result;

  if (outputBinary) {
    output.write(reinterpret_cast<const char *>(bin.data()), bin.size());
    return success();
  }

  auto new_module = AIETranslateBinaryToCtrlpkt(m.getContext(), bin);
  if (!new_module)
    return failure();
  new_module->print(output);
  return success();
}
