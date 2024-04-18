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

namespace {

std::vector<uint32_t> getProlog() {
  return {0x00000011, 0x01000405, 0x01000100, 0x0B590100, 0x000055FF,
          0x00000001, 0x00000010, 0x314E5A5F, 0x635F5F31, 0x676E696C,
          0x39354E5F, 0x6E693131, 0x5F727473, 0x64726F77, 0x00004573,
          0x07BD9630, 0x000055FF};
}

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

  auto words = reserveAndGetTail(instructions, 2);

  uint32_t opCode = 3;
  words[0] |= (opCode & 0xff) << 24;
  words[0] |= (op.getColumn() & 0xff) << 16;
  words[0] |= (op.getRow() & 0xff) << 8;
  words[0] |= op.getDirection() & 0x1;

  words[1] |= (op.getChannel() & 0xff) << 24;
  words[1] |= (op.getColumnNum() & 0xff) << 16;
  words[1] |= (op.getRowNum() & 0xff) << 8;
}

void appendWrite32(std::vector<uint32_t> &instructions, NpuWrite32Op op) {

  auto words = reserveAndGetTail(instructions, 3);

  uint32_t opCode = 2;
  words[0] |= (opCode & 0xff) << 24;
  words[0] |= (op.getColumn() & 0xff) << 16;
  words[0] |= (op.getRow() & 0xff) << 8;

  words[1] = op.getAddress();

  words[2] = op.getValue();
}

void appendWriteBdShimTile(std::vector<uint32_t> &instructions,
                           NpuWriteBdExShimTileOp op) {

  auto words = reserveAndGetTail(instructions, 10);

  uint32_t opCode = 6;
  words[0] |= (opCode & 0xff) << 24;
  words[0] |= (op.getColumn() & 0xff) << 16;
  words[0] |= (op.getColumnNum() & 0xff) << 8;
  words[0] |= (op.getDdrId() & 0xf) << 4;
  words[0] |= (op.getBdId() & 0xf);

  // TODO: Address Incr
  // words[1] = ...

  words[2] = op.getBufferLength();
  words[3] = op.getBufferOffset();

  // En Packet , OoO BD ID , Packet ID , Packet Type
  words[4] |= (op.getEnablePacket() & 0x1) << 30;
  words[4] |= (op.getOutOfOrderId() & 0x3f) << 24;
  words[4] |= (op.getPacketId() & 0x1f) << 19;
  words[4] |= (op.getPacketType() & 0x7) << 16;

  // TODO: Secure Access
  words[5] |= (op.getD0Size() & 0x3ff) << 20;
  words[5] |= op.getD0Stride() & 0xfffff;

  words[6] = 0x80000000; // burst length;
  words[6] |= (op.getD1Size() & 0x3ff) << 20;
  words[6] |= op.getD1Stride() & 0xfffff;

  // TODO: SIMID, AxCache, AXQoS
  words[7] = op.getD2Stride() & 0xfffff;

  words[8] |= (op.getIterationCurrent() & 0x3f) << 26;
  words[8] |= (op.getIterationSize() & 0x3f) << 20;
  words[8] |= op.getIterationStride() & 0xfffff;

  // TODO: TLAST Suppress
  words[9] |= (op.getNextBd() & 0xf) << 27;
  words[9] |= (op.getUseNextBd() & 0x1) << 26;
  words[9] |= (op.getValidBd() & 0x1) << 25;
  words[9] |= (op.getLockRelVal() & 0xef) << 18;
  words[9] |= (op.getLockRelId() & 0xf) << 13;
  words[9] |= (op.getLockAcqEnable() & 0x1) << 12;
  words[9] |= (op.getLockAcqVal() & 0xef) << 5;
  words[9] |= op.getLockAcqId() & 0xf;
}

} // namespace

std::vector<uint32_t> xilinx::AIE::AIETranslateToNPU(ModuleOp module) {

  std::vector<uint32_t> instructions = getProlog();

  DeviceOp deviceOp = *module.getOps<DeviceOp>().begin();
  auto funcOps = deviceOp.getOps<func::FuncOp>();
  for (auto f : funcOps) {
    if (f.isDeclaration())
      continue;
    Block &entry = f.getRegion().front();
    for (auto &o : entry) {
      llvm::TypeSwitch<Operation *>(&o)
          .Case<NpuSyncOp>([&](auto op) { appendSync(instructions, op); })
          .Case<NpuWrite32Op>([&](auto op) { appendWrite32(instructions, op); })
          .Case<NpuWriteBdExShimTileOp>(
              [&](auto op) { appendWriteBdShimTile(instructions, op); });
    }
  }

  return instructions;
}

LogicalResult xilinx::AIE::AIETranslateToNPU(ModuleOp module,
                                             raw_ostream &output) {
  auto instructions = AIETranslateToNPU(module);
  for (auto w : instructions)
    output << llvm::format("%08X\n", w);
  return success();
}
