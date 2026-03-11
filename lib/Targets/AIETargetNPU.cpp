//===- AIETargetNPU.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Runtime/TxnEncoding.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

// Helper function for reserving space in instruction vector (still used by
// appendBlockWrite and control packet translation below).
llvm::MutableArrayRef<uint32_t>
reserveAndGetTail(std::vector<uint32_t> &instructions, uint64_t tailSize) {
  auto oldSize = instructions.size();
  auto newSize = oldSize + tailSize;
  instructions.resize(newSize, 0);
  return llvm::MutableArrayRef<uint32_t>(instructions.data() + oldSize,
                                         tailSize);
}

// Thin wrappers that extract MLIR attributes and delegate to TxnEncoding.h.

void appendSync(std::vector<uint32_t> &instructions, NpuSyncOp op) {
  aie_runtime::txn_append_sync(instructions, op.getColumn(), op.getRow(),
                               static_cast<uint32_t>(op.getDirection()),
                               op.getChannel(), op.getColumnNum(),
                               op.getRowNum());
}

void appendWrite32(std::vector<uint32_t> &instructions, NpuWrite32Op op) {
  if (op.getBuffer()) {
    op.emitOpError("Cannot translate symbolic address");
    return;
  }
  aie_runtime::txn_append_write32(instructions, *op.getAbsoluteAddress(),
                                  op.getValue());
}

void appendMaskWrite32(std::vector<uint32_t> &instructions,
                       NpuMaskWrite32Op op) {
  if (op.getBuffer()) {
    op.emitOpError("Cannot translate symbolic address");
    return;
  }
  aie_runtime::txn_append_maskwrite32(instructions, *op.getAbsoluteAddress(),
                                      op.getValue(), op.getMask());
}

void appendLoadPdi(std::vector<uint32_t> &instructions, NpuLoadPdiOp op) {
  aie_runtime::txn_append_loadpdi(instructions, op.getId(), op.getSize(),
                                  op.getAddress());
}

void appendAddressPatch(std::vector<uint32_t> &instructions,
                        NpuAddressPatchOp op) {
  aie_runtime::txn_append_address_patch(instructions, op.getAddr(),
                                        op.getArgIdx(), op.getArgPlus());
}

void appendBlockWrite(std::vector<uint32_t> &instructions, NpuBlockWriteOp op) {
  std::optional<uint32_t> address = op.getAbsoluteAddress();
  DenseIntElementsAttr data = op.getDataWords();

  // Extract payload into a temporary buffer.
  std::vector<uint32_t> payload;
  payload.reserve(data.size());
  for (auto d : data)
    payload.push_back(d.getZExtValue());

  // Use encoding library for the core format, then fix up col/row field.
  aie_runtime::txn_append_blockwrite(instructions, *address, payload.data(),
                                     payload.size());

  // The encoding library leaves word[1] as 0. If col/row are present, set it.
  auto col = op.getColumn();
  auto row = op.getRow();
  if (col && row) {
    // word[1] is at position (current_size - headerSize - count + 1)
    size_t headerPos = instructions.size() - 4 - payload.size();
    instructions[headerPos + 1] = (*col & 0xff) | ((*row & 0xff) << 8);
  }
}

void appendPreempt(std::vector<uint32_t> &instructions, NpuPreemptOp op) {
  aie_runtime::txn_append_preempt(instructions, op.getLevel());
}

} // namespace

LogicalResult xilinx::AIE::AIETranslateNpuToBinary(
    mlir::ModuleOp moduleOp, std::vector<uint32_t> &instructions,
    StringRef deviceName, StringRef sequenceName) {

  DeviceOp deviceOp =
      DeviceOp::getForSymbolInModuleOrError(moduleOp, deviceName);
  if (!deviceOp) {
    return failure();
  }

  const AIETargetModel &tm = deviceOp.getTargetModel();

  // Build device info for the TXN header.
  aie_runtime::TxnDeviceInfo devInfo;
  devInfo.major = 0;
  devInfo.minor = 1;
  devInfo.devGen = llvm::isa<AIE::BaseNPU2TargetModel>(tm) ? 4 : 3;
  devInfo.numRows = tm.rows();
  devInfo.numCols = tm.columns();
  devInfo.numMemTileRows = tm.getNumMemTileRows();

  AIE::RuntimeSequenceOp seq =
      AIE::RuntimeSequenceOp::getForSymbolInDeviceOrError(deviceOp,
                                                          sequenceName);
  if (!seq) {
    return failure();
  }

  uint32_t count = 0;
  for (Block &block : seq.getBody()) {
    for (Operation &o : block) {
      llvm::TypeSwitch<Operation *>(&o)
          .Case<NpuSyncOp>([&](auto op) {
            count++;
            appendSync(instructions, op);
          })
          .Case<NpuWrite32Op>([&](auto op) {
            count++;
            appendWrite32(instructions, op);
          })
          .Case<NpuBlockWriteOp>([&](auto op) {
            count++;
            appendBlockWrite(instructions, op);
          })
          .Case<NpuMaskWrite32Op>([&](auto op) {
            count++;
            appendMaskWrite32(instructions, op);
          })
          .Case<NpuLoadPdiOp>([&](auto op) {
            count++;
            appendLoadPdi(instructions, op);
          })
          .Case<NpuAddressPatchOp>([&](auto op) {
            count++;
            appendAddressPatch(instructions, op);
          })
          .Case<NpuPreemptOp>([&](auto op) {
            count++;
            appendPreempt(instructions, op);
          });
    }
  }

  // Prepend the TXN header (inserts 4 words at the front).
  aie_runtime::txn_prepend_header(instructions, count, devInfo);
  return success();
}

LogicalResult xilinx::AIE::AIETranslateControlPacketsToUI32Vec(
    ModuleOp module, std::vector<uint32_t> &instructions, StringRef deviceName,
    StringRef sequenceName) {
  DeviceOp deviceOp =
      AIE::DeviceOp::getForSymbolInModuleOrError(module, deviceName);
  if (!deviceOp) {
    return failure();
  }
  OpBuilder builder = OpBuilder::atBlockBegin(deviceOp.getBody());
  AIE::RuntimeSequenceOp seq =
      AIE::RuntimeSequenceOp::getForSymbolInDeviceOrError(deviceOp,
                                                          sequenceName);
  if (!seq) {
    return failure();
  }

  Block &entry = seq.getBody().front();
  for (auto &o : entry) {
    auto packetOp = dyn_cast<AIEX::NpuControlPacketOp>(o);
    if (!packetOp)
      continue;

    uint32_t size = 0;
    auto data = packetOp.getData();
    if (data)
      size = data->size();

    auto words = reserveAndGetTail(instructions, 2 + size);

    if (!data && packetOp.getLength())
      size = *packetOp.getLength();

    auto parity = [](uint32_t n) {
      uint32_t p = 0;
      while (n) {
        p += n & 1;
        n >>= 1;
      }
      return (p % 2) == 0;
    };

    // stream header is attached here instead of by shim dma
    int col = packetOp.getColumnFromAddr();
    int row = packetOp.getRowFromAddr();
    auto destTile = TileOp::getOrCreate(builder, deviceOp, col, row);
    auto info = destTile->getAttrOfType<AIE::PacketInfoAttr>("controller_id");
    uint32_t hdr = 0;
    if (info)
      hdr = (info.getPktType() & 0x7) << 12 | (info.getPktId() & 0xff);
    else
      destTile->emitWarning("Expected controller_id attribute");
    words[0] = hdr | (0x1 & parity(hdr)) << 31;

    // control packet header
    uint32_t addr = packetOp.getAddress() & 0xFFFFF;
    uint32_t beats = size - 1;
    uint32_t opc = packetOp.getOpcode();
    uint32_t id = packetOp.getStreamId();
    hdr = id << 24 | opc << 22 | beats << 20 | addr;
    words[1] = hdr | (0x1 & parity(hdr)) << 31;

    // configuration data
    if (opc == 0x0 || opc == 0x2)
      for (unsigned i = 0; i < size; i++)
        words[i + 2] = data.value()[i];
  }
  return success();
}
