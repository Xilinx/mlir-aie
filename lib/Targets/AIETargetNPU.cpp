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

#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

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

  // XAIE_IO_CUSTOM_OP_BEGIN
  uint32_t opCode = 0x80;
  // Wait until the number of BDs in the same channel of all tiles equal to 0
  words[0] = (opCode & 0xff);

  words[1] = 4 * 4; // Operation Size

  words[2] |= op.getDirection() & 0xff;
  words[2] |= (op.getRow() & 0xff) << 8;
  words[2] |= (op.getColumn() & 0xff) << 16;

  words[3] |= (op.getRowNum() & 0xff) << 8;
  words[3] |= (op.getColumnNum() & 0xff) << 16;
  words[3] |= (op.getChannel() & 0xff) << 24;
}

void appendWrite32(std::vector<uint32_t> &instructions, NpuWrite32Op op) {

  auto words = reserveAndGetTail(instructions, 6);

  // XAIE_IO_WRITE
  uint32_t opCode = 0;
  words[0] = (opCode & 0xff);
  words[1] = 0;
  words[2] = ((op.getColumn() & 0xff) << XAIE_COL_SHIFT) |
             ((op.getRow() & 0xff) << XAIE_ROW_SHIFT) |
             (op.getAddress() & 0xFFFFF); // ADDR_LOW
  words[3] = 0;
  words[4] = op.getValue(); // Value
  words[5] = 6 * 4;         // Operation Size
}

void appendAddressPatch(std::vector<uint32_t> &instructions,
                        NpuAddressPatchOp op) {

  auto words = reserveAndGetTail(instructions, 12);

  // XAIE_IO_CUSTOM_OP_BEGIN + 1
  uint32_t opCode = 0x81;
  words[0] = (opCode & 0xff);
  words[1] = 12 * 4;

  words[6] = op.getAddr();
  words[7] = 0;

  words[8] = op.getArgIdx();
  words[9] = 0;

  words[10] = op.getArgPlus();
  words[11] = 0;
}

void appendWriteBdShimTile(std::vector<uint32_t> &instructions,
                           NpuWriteBdExShimTileOp op) {

  auto words = reserveAndGetTail(instructions, 12);

  // XAIE_IO_BLOCKWRITE
  uint32_t opCode = 1;
  words[0] = (opCode & 0xff);
  words[1] = 0;

  // RegOff
  auto bd_id = op.getBdId();
  uint32_t bd_addr = (op.getColumn() << 25) | (0x1D000 + bd_id * 0x20);
  words[2] = bd_addr; // ADDR
  words[3] = 12 * 4;  // Operation Size

  // DMA_BDX_0
  words[4] = op.getBufferLength();

  // DMA_BDX_1
  words[5] = op.getBufferOffset();

  // DMA_BDX_2
  // En Packet , OoO BD ID , Packet ID , Packet Type
  words[6] |= (op.getEnablePacket() & 0x1) << 30;
  words[6] |= (op.getOutOfOrderId() & 0x3f) << 24;
  words[6] |= (op.getPacketId() & 0x1f) << 19;
  words[6] |= (op.getPacketType() & 0x7) << 16;

  // DMA_BDX_3
  // TODO: Secure Access
  words[7] |= (op.getD0Size() & 0x3ff) << 20;
  words[7] |= op.getD0Stride() & 0xfffff;

  // DMA_BDX_4
  words[8] = 0x80000000; // burst length;
  words[8] |= (op.getD1Size() & 0x3ff) << 20;
  words[8] |= op.getD1Stride() & 0xfffff;

  // DMA_BDX_5
  // TODO: SIMID, AxCache, AXQoS
  words[9] = op.getD2Stride() & 0xfffff;

  // DMA_BDX_6
  words[10] |= (op.getIterationCurrent() & 0x3f) << 26;
  words[10] |= (op.getIterationSize() & 0x3f) << 20;
  words[10] |= op.getIterationStride() & 0xfffff;

  // DMA_BDX_7
  // TODO: TLAST Suppress
  words[11] |= (op.getNextBd() & 0xf) << 27;
  words[11] |= (op.getUseNextBd() & 0x1) << 26;
  words[11] |= (op.getValidBd() & 0x1) << 25;
  words[11] |= (op.getLockRelVal() & 0xef) << 18;
  words[11] |= (op.getLockRelId() & 0xf) << 13;
  words[11] |= (op.getLockAcqEnable() & 0x1) << 12;
  words[11] |= (op.getLockAcqVal() & 0xef) << 5;
  words[11] |= op.getLockAcqId() & 0xf;
}

} // namespace

std::vector<uint32_t> xilinx::AIE::AIETranslateToNPU(ModuleOp module) {

  std::vector<uint32_t> instructions;

  auto words = reserveAndGetTail(instructions, 4);

  // setup txn header
  words[0] = 0x06030100;
  words[1] = 0x00000105;

  DeviceOp deviceOp = *module.getOps<DeviceOp>().begin();
  auto funcOps = deviceOp.getOps<func::FuncOp>();
  int count = 0;
  for (auto f : funcOps) {
    if (f.isDeclaration())
      continue;
    Block &entry = f.getRegion().front();
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
          .Case<NpuAddressPatchOp>([&](auto op) {
            count++;
            appendAddressPatch(instructions, op);
          })
          .Case<NpuWriteBdExShimTileOp>([&](auto op) {
            count++;
            appendWriteBdShimTile(instructions, op);
          });
    }
  }

  // write size fields of the txn header
  instructions[2] = count;
  instructions[3] = instructions.size() * sizeof(uint32_t);
  return instructions;
}

LogicalResult xilinx::AIE::AIETranslateToNPU(ModuleOp module,
                                             raw_ostream &output) {
  auto instructions = AIETranslateToNPU(module);
  for (auto w : instructions)
    output << llvm::format("%08X\n", w);
  return success();
}
