//===- AIETargetUcCert.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
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

// NOP
void emitNop(CertNopOp op, std::string &text) { text += "  NOP\n"; }

// MASK_WRITE_32    0x6838000, 0x2, 0x2
void emitMaskWrite32(CertMaskWrite32Op op, std::string &text) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "  MASK_WRITE_32          ";
  ss << llvm::format("0x%08x, ", op.getAddress());
  ss << llvm::format("0x%08x, ", op.getMask());
  ss << llvm::format("0x%08x\n", op.getValue());
  text += ss.str();
}

// WRITE_32               0x01A0634, 0x80000004
void emitWrite32(CertWrite32Op op, std::string &text) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "  WRITE_32               ";
  ss << llvm::format("0x%08x, ", op.getAddress());
  ss << llvm::format("0x%08x\n", op.getValue());
  text += ss.str();
}

// uC_DMA_WRITE_DES_SYNC  @INPUT_row1_actor6_task6_ucbds
void emitUcDmaWriteDesSync(CertUcDmaWriteDesSyncOp op, std::string &text) {
  text += "  uC_DMA_WRITE_DES_SYNC  ";
  text += "@" + op.getSymbol().str() + "\n";
}

// WAIT_TCTS              TILE_0_1, MEM_MM2S_0, 1
void emitWaitTCTS(CertWaitTCTSOp op, std::string &text) {
  uint32_t tileId = op.getTileId();
  uint32_t channelId = op.getChannelId();
  text += "  WAIT_TCTS              ";
  text += std::to_string(tileId) + ", ";
  text += std::to_string(channelId) + ", ";
  text += std::to_string(op.getTargetTcts()) + "\n";
}

// APPLY_OFFSET_57     @mem21_bd0, 1, 0xffff
void emitApplyOffset57(CertApplyOffset57Op op, std::string &text) {
  uint16_t num_entries = op.getNumEntries();
  uint16_t offset = op.getOffset();
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "  APPLY_OFFSET_57        ";
  ss << "@" << op.getSymbol().str() << ", ";
  ss << num_entries << ", ";
  ss << llvm::format("0x%04x", offset) << "\n";
  text += ss.str();
}

// LOCAL_BARRIER       $lb0, 2
void emitLocalBarrier(CertLocalBarrierOp op, std::string &text) {
  text += "  LOCAL_BARRIER          ";
  text += "$lb" + std::to_string(op.getLocalBarrierId()) + ", ";
  text += std::to_string(op.getNumParticipants()) + "\n";
}

// REMOTE_BARRIER      $rb3, 3
void emitRemoteBarrier(CertRemoteBarrierOp op, std::string &text) {
  text += "  REMOTE_BARRIER         ";
  text += "$rb" + std::to_string(op.getRemoteBarrierId()) + ", ";
  text += std::to_string(op.getPartyMask()) + "\n";
}

// START_JOB 0
//   <body>
// END_JOB
void emitJob(CertJobOp jobOp, std::string &text, std::string &data) {

  text += "START_JOB ";
  text += std::to_string(jobOp.getJobId()) + "\n";

  for (auto &o : jobOp.getBody().front()) {
    llvm::TypeSwitch<Operation *>(&o)
        .Case<CertApplyOffset57Op>(
            [&](auto op) { emitApplyOffset57(op, text); })
        .Case<CertLocalBarrierOp>([&](auto op) { emitLocalBarrier(op, text); })
        .Case<CertRemoteBarrierOp>(
            [&](auto op) { emitRemoteBarrier(op, text); })
        .Case<CertMaskWrite32Op>([&](auto op) { emitMaskWrite32(op, text); })
        .Case<CertNopOp>([&](auto op) { emitNop(op, text); })
        .Case<CertUcDmaWriteDesSyncOp>(
            [&](auto op) { emitUcDmaWriteDesSync(op, text); })
        .Case<CertWaitTCTSOp>([&](auto op) { emitWaitTCTS(op, text); })
        .Case<CertWrite32Op>([&](auto op) { emitWrite32(op, text); });
  }

  text += "END_JOB\n\n";
}

void emitJobs(llvm::SmallVector<CertJobOp> &jobs, std::string &text,
              std::string &data) {

  llvm::sort(jobs, [](CertJobOp a, CertJobOp b) {
    return a.getJobId() < b.getJobId();
  });

  for (auto job : jobs) {
    emitJob(job, text, data);
    text += ".eop\n\n";
  }
}

void emitAttachToGroupOp(CertAttachToGroupOp groupOp, std::string &text,
                         std::string &data) {
  if (groupOp.getGroupId())
    text += ".attach_to_group " + std::to_string(groupOp.getGroupId()) + "\n\n";
  auto jobs =
      llvm::to_vector_of<CertJobOp>(groupOp.getBody().getOps<CertJobOp>());
  emitJobs(jobs, text, data);
}

void emitUcDmaBdData(CertUcDmaBdOp op, std::string &data) {
  // lookup data from operation
  auto dataSymbol = op.getRemoteAddress();

  auto global = dyn_cast_if_present<memref::GlobalOp>(
      op->getParentOfType<AIE::DeviceOp>().lookupSymbol(dataSymbol));
  if (!global) {
    op.emitError("Global symbol not found");
    return;
  }

  auto initVal = global.getInitialValue();
  if (!initVal) {
    op.emitError("Global symbol has no initial value");
    return;
  }

  auto initData = dyn_cast<DenseIntElementsAttr>(*initVal);
  if (!initData) {
    op.emitError("Global symbol initial value is not a dense int array");
    return;
  }

  // data0:
  //   .long           0x00005A00
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "  .align 16\n";
  ss << dataSymbol.str() << ":\n";
  for (auto d : initData)
    ss << llvm::format("  .long           0x%08x\n", d.getZExtValue());
  data += ss.str();
}

// UC_DMA_BD       0, 0x001A05C0, @data, 8, 0, 1
void emitUcDmaBd(CertUcDmaBdOp op, std::string &chains, std::string &data) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "  UC_DMA_BD       ";
  ss << "0, ";
  ss << llvm::format("0x%08x, ", op.getLocalAddress());
  ss << "@" + op.getRemoteAddress().str() + ", ";
  ss << op.getLength() << ", ";
  ss << "0, ";
  ss << (op.getNextBd() ? "1\n" : "0\n");
  chains += ss.str();
  emitUcDmaBdData(op, data);
}

// .align           16
// name_of_chain:
//   UC_DMA_BD       0, 0x001A05C0, @data0, 8, 0, 1
void emitUcDmaChain(CertUcDmaChainOp op, std::string &chains,
                    std::string &data) {
  chains += "  .align 16\n";
  chains += op.getName().str() + ":\n";
  for (auto &o : op.getBody().front()) {
    llvm::TypeSwitch<Operation *>(&o).Case<CertUcDmaBdOp>(
        [&](auto op) { emitUcDmaBd(op, chains, data); });
  }
}

} // namespace

LogicalResult xilinx::AIE::AIETranslateToUcDma(ModuleOp module,
                                               std::string &assembly) {

  DeviceOp deviceOp = *module.getOps<DeviceOp>().begin();

  std::vector<std::string> text;
  std::vector<std::string> data;
  std::vector<std::string> chains;

  text.push_back("\n;\n; Code\n;\n\n");
  data.push_back("\n;\n; Data\n;\n\n");
  chains.push_back("\n;\n; Data (chains)\n;\n\n");

  text[0] += ".attach_to_group 0\n\n";
  auto jobs = llvm::to_vector_of<CertJobOp>(deviceOp.getOps<CertJobOp>());
  emitJobs(jobs, text[0], data[0]);

  auto groups = llvm::to_vector_of<CertAttachToGroupOp>(
      deviceOp.getOps<CertAttachToGroupOp>());
  llvm::sort(groups, [](CertAttachToGroupOp a, CertAttachToGroupOp b) {
    return a.getGroupId() < b.getGroupId();
  });

  for (auto o : deviceOp.getBody()->getOps<CertUcDmaChainOp>())
    emitUcDmaChain(o, chains[0], data[0]);

  int group_id = 0;
  for (auto &groupOp : groups) {
    if (group_id) {
      text.push_back("\n;\n; Code\n;\n\n");
      data.push_back("\n;\n; Data\n;\n\n");
      chains.push_back("\n;\n; Data (chains)\n;\n\n");
    }
    emitAttachToGroupOp(groupOp, text[group_id], data[group_id]);
    group_id++;
  }

  for (auto const &[t, c, d] : llvm::zip(text, chains, data)) {
    if (t.size()) {
      assembly += t + "EOF\n";
    }
    if (c.size()) {
      assembly += c;
    }
    if (d.size())
      assembly += "\n" + d;
  }
  return success();
}

LogicalResult xilinx::AIE::AIETranslateToUcDma(ModuleOp module,
                                               raw_ostream &output) {
  std::string assembly;
  auto r = AIETranslateToUcDma(module, assembly);
  if (failed(r))
    return r;
  output << assembly;
  return success();
}
