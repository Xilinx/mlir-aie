//===- AIETargetNPU.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

#define TXN_OPC_WRITE 0x0
#define TXN_OPC_BLOCKWRITE 0x1
#define TXN_OPC_MASKWRITE 0x3
#define TXN_OPC_TCT 0x80
#define TXN_OPC_DDR_PATCH 0x81

namespace {

// Example:
// - instructions = {3,4,5}
// - tailSize = 2
// instructions becomes {3,4,5,0,0} and
// a mutable reference to the tail {0,0} is returned.
llvm::MutableArrayRef<uint32_t>
reserveAndGetTail(std::vector<uint32_t> &instructions, uint64_t tailSize) {
  auto oldSize = instructions.size();
  auto newSize = oldSize + tailSize;
  instructions.resize(newSize, 0);
  return llvm::MutableArrayRef<uint32_t>(instructions.data() + oldSize,
                                         tailSize);
}

void appendSync(std::vector<uint32_t> &instructions, NpuSyncOp op) {

  auto words = reserveAndGetTail(instructions, 4);

  // XAIE_IO_CUSTOM_OP_TCT
  words[0] = TXN_OPC_TCT;

  words[1] = words.size() * sizeof(uint32_t); // Operation Size

  words[2] |= static_cast<uint32_t>(op.getDirection()) & 0xff;
  words[2] |= (op.getRow() & 0xff) << 8;
  words[2] |= (op.getColumn() & 0xff) << 16;

  words[3] |= (op.getRowNum() & 0xff) << 8;
  words[3] |= (op.getColumnNum() & 0xff) << 16;
  words[3] |= (op.getChannel() & 0xff) << 24;
}

void appendWrite32(std::vector<uint32_t> &instructions, NpuWrite32Op op) {

  auto words = reserveAndGetTail(instructions, 3);

  if (op.getBuffer()) {
    op.emitOpError("Cannot translate symbolic address");
    return;
  }

  // XAIE_IO_WRITE
  words[0] = TXN_OPC_WRITE;
  words[1] = op.getAddress();
  auto col = op.getColumn();
  auto row = op.getRow();
  if (col && row) {
    const AIETargetModel &tm = op->getParentOfType<DeviceOp>().getTargetModel();
    words[1] = ((*col & 0xff) << tm.getColumnShift()) |
               ((*row & 0xff) << tm.getRowShift()) | (words[1] & 0xFFFFF);
  }
  words[2] = op.getValue(); // Value
}

void appendMaskWrite32(std::vector<uint32_t> &instructions,
                       NpuMaskWrite32Op op) {

  auto words = reserveAndGetTail(instructions, 4);

  if (op.getBuffer()) {
    op.emitOpError("Cannot translate symbolic address");
    return;
  }

  // XAIE_IO_MASKWRITE
  words[0] = TXN_OPC_MASKWRITE;
  words[1] = op.getAddress();
  auto col = op.getColumn();
  auto row = op.getRow();
  if (col && row) {
    const AIETargetModel &tm = op->getParentOfType<DeviceOp>().getTargetModel();
    words[1] = ((*col & 0xff) << tm.getColumnShift()) |
               ((*row & 0xff) << tm.getRowShift()) | (words[1] & 0xFFFFF);
  }
  words[2] = op.getValue(); // Value
  words[3] = op.getMask();
}

void appendAddressPatch(std::vector<uint32_t> &instructions,
                        NpuAddressPatchOp op) {

  auto words = reserveAndGetTail(instructions, 6);

  // XAIE_IO_CUSTOM_OP_DDR_PATCH
  words[0] = TXN_OPC_DDR_PATCH;
  words[1] = words.size() * sizeof(uint32_t); // Operation Size

  words[2] = op.getAddr();

  words[3] = op.getArgIdx();

  words[4] = op.getArgPlus();
  words[5] = 0;
}

void appendBlockWrite(std::vector<uint32_t> &instructions, NpuBlockWriteOp op) {

  Value memref = op.getData();
  int64_t width = cast<MemRefType>(memref.getType()).getElementTypeBitWidth();
  if (width != 32) {
    op.emitWarning("Only 32-bit data type is supported for now");
    return;
  }

  memref::GetGlobalOp getGlobal = memref.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal) {
    op.emitError("Only MemRefs from memref.get_global are supported");
    return;
  }

  auto global = dyn_cast_if_present<memref::GlobalOp>(
      op->getParentOfType<AIE::DeviceOp>().lookupSymbol(getGlobal.getName()));
  if (!global) {
    op.emitError("Global symbol not found");
    return;
  }

  auto initVal = global.getInitialValue();
  if (!initVal) {
    op.emitError("Global symbol has no initial value");
    return;
  }

  auto data = dyn_cast<DenseIntElementsAttr>(*initVal);
  if (!data) {
    op.emitError("Global symbol initial value is not a dense int array");
    return;
  }

  auto words = reserveAndGetTail(instructions, data.size() + 3);

  // XAIE_IO_BLOCKWRITE
  words[0] = TXN_OPC_BLOCKWRITE;
  words[1] = op.getAddress();
  auto col = op.getColumn();
  auto row = op.getRow();
  if (col && row) {
    const AIETargetModel &tm = op->getParentOfType<DeviceOp>().getTargetModel();
    words[1] = ((*col & 0xff) << tm.getColumnShift()) |
               ((*row & 0xff) << tm.getRowShift()) | (words[1] & 0xFFFFF);
  }
  words[2] = words.size() * sizeof(uint32_t); // Operation Size

  unsigned i = 3;
  for (auto d : data)
    words[i++] = d.getZExtValue();
}

} // namespace

LogicalResult
xilinx::AIE::AIETranslateNpuToBinary(ModuleOp module,
                                     std::vector<uint32_t> &instructions,
                                     StringRef sequenceName) {

  auto words = reserveAndGetTail(instructions, 4);

  DeviceOp deviceOp = *module.getOps<DeviceOp>().begin();
  const AIETargetModel &tm = deviceOp.getTargetModel();

  // setup txn header
  uint8_t major = 1;
  uint8_t minor = 0;
  uint8_t devGen = 3;
  uint8_t numRows = tm.rows();
  uint8_t numCols = tm.columns();
  uint8_t numMemTileRows = tm.getNumMemTileRows();
  uint32_t count = 0;
  words[0] = (numRows << 24) | (devGen << 16) | (minor << 8) | major;
  words[1] = (numMemTileRows << 8) | numCols;

  auto sequenceOps = deviceOp.getOps<AIEX::RuntimeSequenceOp>();
  for (auto seq : sequenceOps) {
    if (sequenceName.size() && sequenceName != seq.getSymName())
      continue;
    Block &entry = seq.getBody().front();
    for (auto &o : entry) {
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
          .Case<NpuAddressPatchOp>([&](auto op) {
            count++;
            appendAddressPatch(instructions, op);
          });
    }
  }

  // write size fields of the txn header
  instructions[2] = count;
  instructions[3] = instructions.size() * sizeof(uint32_t); // size of the txn
  return success();
}

LogicalResult xilinx::AIE::AIETranslateControlPacketsToUI32Vec(
    ModuleOp module, std::vector<uint32_t> &instructions,
    StringRef sequenceName) {
  DeviceOp deviceOp = *module.getOps<DeviceOp>().begin();
  auto sequenceOps = deviceOp.getOps<AIEX::RuntimeSequenceOp>();
  for (auto seq : sequenceOps) {
    if (sequenceName.size() && sequenceName != seq.getSymName())
      continue;
    Block &entry = seq.getBody().front();
    for (auto &o : entry) {
      llvm::TypeSwitch<Operation *>(&o).Case<NpuControlPacketOp>([&](auto op) {
        uint32_t size = 0;
        auto data = op.getData();
        auto length = op.getLength();
        if (data)
          size = data->size();
        auto words = reserveAndGetTail(instructions, 1 + size);
        if (!data && length)
          size = *length;
        auto parity = [](uint32_t n) {
          uint32_t p = 0;
          while (n) {
            p += n & 1;
            n >>= 1;
          }
          return (p % 2) == 0;
        };
        uint32_t addr = op.getAddress() & 0xFFFFF;
        uint32_t beats = size - 1;
        uint32_t opc = op.getOpcode();
        uint32_t id = op.getStreamId();
        uint32_t hdr = id << 24 | opc << 22 | beats << 20 | addr;
        words[0] = hdr | (0x1 & parity(hdr)) << 31;
        if (opc == 0x0 || opc == 0x2)
          for (unsigned i = 0; i < size; i++)
            words[i + 1] = data.value()[i];
      });
    }
  }
  return success();
}
