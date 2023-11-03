//===- AIETargetIPU.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "AIETargets.h"

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

static void emitProlog(llvm::raw_ostream &os) {
  os << "00000011\n";
  os << "01000405\n";
  os << "01000100\n";
  os << "0B590100\n";
  os << "000055FF\n";
  os << "00000001\n";
  os << "00000010\n";
  os << "314E5A5F\n";
  os << "635F5F31\n";
  os << "676E696C\n";
  os << "39354E5F\n";
  os << "6E693131\n";
  os << "5F727473\n";
  os << "64726F77\n";
  os << "00004573\n";
  os << "07BD9630\n";
  os << "000055FF\n";
}

static void emitSync(raw_ostream &output, IpuSyncOp op) {
  std::vector<uint32_t> words(2, 0);

  uint32_t op_code = 3;
  words[0] |= (op_code & 0xff) << 24;
  words[0] |= (op.getColumn() & 0xff) << 16;
  words[0] |= (op.getRow() & 0xff) << 8;
  words[0] |= op.getDirection() & 0x1;

  words[1] |= (op.getChannel() & 0xff) << 24;
  words[1] |= (op.getColumnNum() & 0xff) << 16;
  words[1] |= (op.getRowNum() & 0xff) << 8;

  for (auto w : words)
    output << llvm::format("%08X\n", w);
}

static void emitWrite32(raw_ostream &output, IpuWrite32Op op) {
  std::vector<uint32_t> words(3, 0);

  uint32_t op_code = 2;
  words[0] |= (op_code & 0xff) << 24;
  words[0] |= (op.getColumn() & 0xff) << 16;
  words[0] |= (op.getRow() & 0xff) << 8;
  words[1] = op.getAddress();
  words[2] = op.getValue();

  for (auto w : words)
    output << llvm::format("%08X\n", w);
}

static void emitWriteBdShimTile(raw_ostream &output,
                                IpuWriteBdExShimTileOp op) {
  std::vector<uint32_t> words(10, 0);

  uint32_t op_code = 6;
  words[0] |= (op_code & 0xff) << 24;
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
  words[5] |= (op.getD0Wrap() & 0x3ff) << 20;
  words[5] |= op.getD0Stepsize() & 0xfffff;

  words[6] = 0x80000000; // burst length;
  words[6] |= (op.getD1Wrap() & 0x3ff) << 20;
  words[6] |= op.getD1Stepsize() & 0xfffff;

  // TODO: SIMID, AxCache, AXQoS
  words[7] = op.getD2Stepsize() & 0xfffff;

  words[8] |= (op.getIterationCurrent() & 0x3f) << 26;
  words[8] |= (op.getIterationWrap() & 0x3f) << 20;
  words[8] |= op.getIterationStepsize() & 0xfffff;

  // TODO: TLAST Suppress
  words[9] |= (op.getNextBd() & 0xf) << 27;
  words[9] |= (op.getUseNextBd() & 0x1) << 26;
  words[9] |= (op.getValidBd() & 0x1) << 25;
  words[9] |= (op.getLockRelVal() & 0xef) << 18;
  words[9] |= (op.getLockRelId() & 0xf) << 13;
  words[9] |= (op.getLockAcqEnable() & 0x1) << 12;
  words[9] |= (op.getLockAcqVal() & 0xef) << 5;
  words[9] |= op.getLockAcqId() & 0xf;

  for (auto w : words)
    output << llvm::format("%08X\n", w);
}

LogicalResult xilinx::AIE::AIETranslateToIPU(ModuleOp module,
                                             raw_ostream &output) {
  emitProlog(output);

  DeviceOp deviceOp = *module.getOps<DeviceOp>().begin();
  auto funcOps = deviceOp.getOps<func::FuncOp>();
  for (auto f : funcOps) {
    if (f.isDeclaration())
      continue;
    Block &entry = f.getRegion().front();
    for (auto &o : entry) {
      llvm::TypeSwitch<Operation *>(&o)
          .Case<IpuSyncOp>([&](auto op) { emitSync(output, op); })
          .Case<IpuWrite32Op>([&](auto op) { emitWrite32(output, op); })
          .Case<IpuWriteBdExShimTileOp>(
              [&](auto op) { emitWriteBdShimTile(output, op); });
    }
  }
  return success();
}
