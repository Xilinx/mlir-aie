//===- AIETargetUcCert.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"

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

// APPLY_OFFSET_57     @mem21_bd0, 1, 65535
void emitApplyOffset57(CertApplyOffset57Op op, std::string &text) {
  uint16_t num_entries = op.getNumEntries();
  uint16_t offset = op.getOffset();
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "  APPLY_OFFSET_57        ";
  ss << "@" << op.getSymbol().str() << ", ";
  ss << num_entries << ", ";
  ss << offset << "\n";
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

// LOAD_PDI            1, @device_config
void emitLoadPdi(CertLoadPdiOp op, std::string &text) {
  text += "  LOAD_PDI               ";
  text += std::to_string(op.getPdiId()) + ", ";
  text += "@" + op.getSymbol().str() + "\n";
}

// PREEMPT             0x0001, @save, @restore
void emitPreempt(CertPreemptOp op, std::string &text) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "  PREEMPT                ";
  ss << llvm::format("0x%04x, ", op.getId());
  ss << "@" << op.getSaveSection().str() << ", ";
  ss << "@" << op.getRestoreSection().str() << "\n";
  text += ss.str();
}

// START_JOB 0
//   <body>
// END_JOB
LogicalResult emitJob(CertJobOp jobOp, std::string &text, std::string &data) {
  LogicalResult result = success();

  text += "START_JOB ";
  text += std::to_string(jobOp.getJobId()) + "\n";

  for (auto &o : jobOp.getBody().front()) {
    llvm::TypeSwitch<Operation *>(&o)
        .Case<CertApplyOffset57Op>(
            [&](auto op) { emitApplyOffset57(op, text); })
        .Case<CertLocalBarrierOp>([&](auto op) { emitLocalBarrier(op, text); })
        .Case<CertLoadPdiOp>([&](auto op) { emitLoadPdi(op, text); })
        .Case<CertPreemptOp>([&](auto op) { emitPreempt(op, text); })
        .Case<CertRemoteBarrierOp>(
            [&](auto op) { emitRemoteBarrier(op, text); })
        .Case<CertMaskWrite32Op>([&](auto op) { emitMaskWrite32(op, text); })
        .Case<CertNopOp>([&](auto op) { emitNop(op, text); })
        .Case<CertUcDmaWriteDesSyncOp>(
            [&](auto op) { emitUcDmaWriteDesSync(op, text); })
        .Case<CertWaitTCTSOp>([&](auto op) { emitWaitTCTS(op, text); })
        .Case<CertWrite32Op>([&](auto op) { emitWrite32(op, text); })
        .Case<AIE::EndOp>([](auto) { /* implicit region terminator, skip */ })
        .Default([&](Operation *op) {
          op->emitError("Unsupported operation in CertJobOp");
          result = failure();
        });
  }

  text += "END_JOB\n\n";
  return result;
}

LogicalResult emitJobs(llvm::SmallVector<CertJobOp> &jobs, std::string &text,
                       std::string &data) {
  LogicalResult result = success();

  llvm::sort(jobs, [](CertJobOp a, CertJobOp b) {
    return a.getJobId() < b.getJobId();
  });

  for (auto job : jobs) {
    if (failed(emitJob(job, text, data)))
      result = failure();
    text += ".eop\n\n";
  }
  return result;
}

// .align 16
// <page contents>
// .eop
LogicalResult emitPage(CertPageOp pageOp, std::string &text,
                       std::string &data) {
  LogicalResult result = success();
  text += ".align 16\n";

  // Process jobs within the page
  auto jobs =
      llvm::to_vector_of<CertJobOp>(pageOp.getBody().getOps<CertJobOp>());
  llvm::sort(jobs, [](CertJobOp a, CertJobOp b) {
    return a.getJobId() < b.getJobId();
  });

  for (auto job : jobs) {
    if (failed(emitJob(job, text, data)))
      result = failure();
  }

  text += ".eop\n\n";
  return result;
}

// label:
// .include section_file.asm
// .endl label
//
// The section content is emitted to a separate file
LogicalResult emitSection(CertSectionOp sectionOp, std::string &text,
                          llvm::StringRef outputPath) {
  LogicalResult result = success();
  std::string sectionName = sectionOp.getSymName().str();
  std::string sectionFileName = sectionName + ".asm";

  // Build the section content in separate strings
  std::string sectionText;
  std::string sectionData;

  // Process pages within the section
  for (auto page : sectionOp.getBody().getOps<CertPageOp>()) {
    if (failed(emitPage(page, sectionText, sectionData)))
      result = failure();
  }

  // Process standalone jobs (not inside pages) for backwards compatibility
  auto jobs =
      llvm::to_vector_of<CertJobOp>(sectionOp.getBody().getOps<CertJobOp>());
  for (auto job : jobs) {
    if (failed(emitJob(job, sectionText, sectionData)))
      result = failure();
    sectionText += ".eop\n\n";
  }

  // Write section content to separate file
  std::string sectionFilePath;
  if (!outputPath.empty()) {
    llvm::SmallString<128> path(outputPath);
    llvm::sys::path::remove_filename(path);
    llvm::sys::path::append(path, sectionFileName);
    sectionFilePath = path.str().str();
  } else {
    sectionFilePath = sectionFileName;
  }

  std::error_code EC;
  llvm::raw_fd_ostream sectionFile(sectionFilePath, EC);
  if (EC) {
    // Don't emit a .include directive pointing at a file we failed to write.
    return sectionOp.emitError("failed to open section file '")
           << sectionFilePath << "': " << EC.message();
  }

  // Emit text section
  if (!sectionText.empty()) {
    sectionFile << ";\n;text\n;\n";
    sectionFile << sectionText;
  }

  // Emit EOF before data section
  sectionFile << "EOF\n\n";

  // Emit data section
  if (!sectionData.empty()) {
    sectionFile << ";\n;data\n;\n";
    sectionFile << sectionData;
  }

  sectionFile.close();
  if (sectionFile.has_error()) {
    std::error_code writeEC = sectionFile.error();
    // Clear the error so the raw_fd_ostream destructor doesn't report_fatal.
    sectionFile.clear_error();
    return sectionOp.emitError("failed to write section file '")
           << sectionFilePath << "': " << writeEC.message();
  }

  // In main file, emit label with .include directive
  text += sectionName + ":\n";
  text += ".include " + sectionFileName + "\n";
  text += ".endl " + sectionName + "\n\n";
  return result;
}

LogicalResult emitAttachToGroupOp(CertAttachToGroupOp groupOp,
                                  std::string &text, std::string &data,
                                  llvm::StringRef outputPath) {
  LogicalResult result = success();

  // Process sections first (they may be referenced by main code)
  for (auto section : groupOp.getBody().getOps<CertSectionOp>()) {
    if (failed(emitSection(section, text, outputPath)))
      result = failure();
  }

  if (groupOp.getGroupId())
    text += ".attach_to_group " + std::to_string(groupOp.getGroupId()) + "\n\n";

  // Process pages (if any)
  for (auto page : groupOp.getBody().getOps<CertPageOp>()) {
    if (failed(emitPage(page, text, data)))
      result = failure();
  }

  // Process standalone jobs (not inside pages) for backwards compatibility
  auto jobs =
      llvm::to_vector_of<CertJobOp>(groupOp.getBody().getOps<CertJobOp>());
  if (failed(emitJobs(jobs, text, data)))
    result = failure();
  return result;
}

void emitUcDmaBdData(CertUcDmaBdOp op, std::string &data,
                     llvm::StringSet<> &emittedGlobals) {
  // lookup data from operation
  auto dataSymbol = op.getRemoteAddress();
  if (emittedGlobals.contains(dataSymbol))
    return;

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
  emittedGlobals.insert(dataSymbol);
}

// UC_DMA_BD       0, 0x001A05C0, @data, 8, 0, 1
void emitUcDmaBd(CertUcDmaBdOp op, std::string &chains, std::string &data,
                 llvm::StringSet<> &emittedGlobals) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "  UC_DMA_BD       0, ";
  ss << llvm::format("0x%08x, ", op.getLocalAddress());
  ss << "@" + op.getRemoteAddress().str() + ", ";
  ss << op.getLength() << ", ";
  ss << "0, ";
  ss << (op.getNextBd() ? "1\n" : "0\n");
  chains += ss.str();
  emitUcDmaBdData(op, data, emittedGlobals);
}

// .align           16
// name_of_chain:
//   UC_DMA_BD       0, 0x001A05C0, @data0, 8, 0, 1
void emitUcDmaChain(CertUcDmaChainOp op, std::string &chains, std::string &data,
                    llvm::StringSet<> &emittedGlobals) {
  chains += "  .align 16\n";
  chains += op.getName().str() + ":\n";
  for (auto &o : op.getBody().front()) {
    llvm::TypeSwitch<Operation *>(&o).Case<CertUcDmaBdOp>(
        [&](auto op) { emitUcDmaBd(op, chains, data, emittedGlobals); });
  }
}

} // namespace

LogicalResult xilinx::AIE::AIETranslateToUcDma(ModuleOp module,
                                               std::string &assembly,
                                               llvm::StringRef outputPath) {

  DeviceOp deviceOp = *module.getOps<DeviceOp>().begin();

  std::vector<std::string> text;
  std::vector<std::string> data;
  std::vector<std::string> chains;
  llvm::StringSet<> emittedGlobals;

  text.push_back("\n;\n; Code\n;\n\n");
  data.push_back("\n;\n; Data\n;\n\n");
  chains.push_back("\n;\n; Data (chains)\n;\n\n");

  auto &tm = deviceOp.getTargetModel();
  text[0] += ".partition " + std::to_string(tm.columns()) + "column\n";

  // Process sections first
  text[0] += ".attach_to_group 0\n\n";

  LogicalResult result = success();

  // Process sections after attach_to_group so they're in the column
  for (auto section : deviceOp.getBody()->getOps<CertSectionOp>()) {
    if (failed(emitSection(section, text[0], outputPath)))
      result = failure();
  }

  // Process pages at device level (if any)
  for (auto page : deviceOp.getBody()->getOps<CertPageOp>()) {
    if (failed(emitPage(page, text[0], data[0])))
      result = failure();
  }

  // Process standalone jobs at device level (for backwards compatibility)
  auto jobs =
      llvm::to_vector_of<CertJobOp>(deviceOp.getBody()->getOps<CertJobOp>());
  if (failed(emitJobs(jobs, text[0], data[0])))
    result = failure();

  auto groups = llvm::to_vector_of<CertAttachToGroupOp>(
      deviceOp.getBody()->getOps<CertAttachToGroupOp>());
  llvm::sort(groups, [](CertAttachToGroupOp a, CertAttachToGroupOp b) {
    return a.getGroupId() < b.getGroupId();
  });

  for (auto o : deviceOp.getBody()->getOps<CertUcDmaChainOp>())
    emitUcDmaChain(o, chains[0], data[0], emittedGlobals);

  int group_id = 0;
  for (auto &groupOp : groups) {
    if (group_id) {
      text.push_back("\n;\n; Code\n;\n\n");
      data.push_back("\n;\n; Data\n;\n\n");
      chains.push_back("\n;\n; Data (chains)\n;\n\n");
    }
    if (failed(emitAttachToGroupOp(groupOp, text[group_id], data[group_id],
                                   outputPath)))
      result = failure();
    group_id++;
  }

  if (failed(result))
    return result;

  for (auto const &[t, c, d] : llvm::zip(text, chains, data)) {
    if (t.size()) {
      assembly += t;
      // Add final EOF after all code (sections emit their own EOF before .endl)
      assembly += "EOF\n";
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

  // Section files will be created in the current directory
  // since we can't reliably extract the output path from raw_ostream
  llvm::StringRef outputPath = "";

  auto r = AIETranslateToUcDma(module, assembly, outputPath);
  if (failed(r))
    return r;
  output << assembly;
  return success();
}
