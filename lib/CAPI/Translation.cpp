//===- Translation.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "aie-c/Translation.h"

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Targets/AIERT.h"
#include "aie/Targets/AIETargets.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace xilinx::AIE;

MlirStringRef aieTranslateAIEVecToCpp(MlirOperation moduleOp, bool aie2) {
  std::string cpp;
  llvm::raw_string_ostream os(cpp);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(xilinx::aievec::translateAIEVecToCpp(mod, aie2, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(cpp.size()));
  cpp.copy(cStr, cpp.size());
  return mlirStringRefCreate(cStr, cpp.size());
};

MlirStringRef aieTranslateModuleToLLVMIR(MlirOperation moduleOp) {
  std::string llvmir;
  llvm::raw_string_ostream os(llvmir);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(mod, llvmContext);
  if (!llvmModule)
    return mlirStringRefCreate(nullptr, 0);
  llvmModule->print(os, nullptr);
  char *cStr = static_cast<char *>(malloc(llvmir.size()));
  llvmir.copy(cStr, llvmir.size());
  return mlirStringRefCreate(cStr, llvmir.size());
}

MlirLogicalResult aieTranslateToCDODirect(MlirOperation moduleOp,
                                          MlirStringRef workDirPath,
                                          MlirStringRef deviceName,
                                          bool bigEndian, bool emitUnified,
                                          bool cdoDebug, bool aieSim,
                                          bool xaieDebug, bool enableCores) {
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  auto status = AIETranslateToCDODirect(
      mod, llvm::StringRef(workDirPath.data, workDirPath.length),
      llvm::StringRef(deviceName.data, deviceName.length), bigEndian,
      emitUnified, cdoDebug, aieSim, xaieDebug, enableCores);
  std::vector<std::string> diagnostics;
  ScopedDiagnosticHandler handler(mod.getContext(), [&](Diagnostic &d) {
    llvm::raw_string_ostream(diagnostics.emplace_back())
        << d.getLocation() << ": " << d;
  });
  if (failed(status))
    for (const auto &diagnostic : diagnostics)
      std::cerr << diagnostic << "\n";

  return wrap(status);
}

MlirOperation aieTranslateBinaryToTxn(MlirContext ctx, MlirStringRef binary) {
  std::vector<uint8_t> binaryData(binary.data, binary.data + binary.length);
  auto mod = convertTransactionBinaryToMLIR(unwrap(ctx), binaryData);
  if (!mod)
    return wrap(ModuleOp().getOperation());
  return wrap(mod->getOperation());
}

MlirStringRef aieTranslateNpuToBinary(MlirOperation moduleOp,
                                      MlirStringRef deviceNameMlir,
                                      MlirStringRef sequenceNameMlir) {
  std::vector<uint32_t> insts;
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  llvm::StringRef deviceName(deviceNameMlir.data, deviceNameMlir.length);
  llvm::StringRef sequenceName(sequenceNameMlir.data, sequenceNameMlir.length);
  if (failed(AIETranslateNpuToBinary(mod, insts, deviceName, sequenceName)))
    return mlirStringRefCreate(nullptr, 0);
  size_t insts_size = insts.size() * sizeof(uint32_t);
  char *cStr = static_cast<char *>(malloc(insts_size));
  memcpy(cStr, insts.data(), insts_size);
  return mlirStringRefCreate(cStr, insts_size);
}

MlirStringRef aieTranslateControlPacketsToUI32Vec(MlirOperation moduleOp,
                                                  MlirStringRef deviceName) {
  std::vector<uint32_t> insts;
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateControlPacketsToUI32Vec(
          mod, insts, llvm::StringRef(deviceName.data, deviceName.length))))
    return mlirStringRefCreate(nullptr, 0);
  size_t insts_size = insts.size() * sizeof(uint32_t);
  char *cStr = static_cast<char *>(malloc(insts_size));
  memcpy(cStr, insts.data(), insts_size);
  return mlirStringRefCreate(cStr, insts_size);
}

MlirStringRef aieTranslateToXAIEV2(MlirOperation moduleOp,
                                   MlirStringRef deviceName) {
  std::string xaie;
  llvm::raw_string_ostream os(xaie);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToXAIEV2(
          mod, os, llvm::StringRef(deviceName.data, deviceName.length))))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(xaie.size()));
  xaie.copy(cStr, xaie.size());
  return mlirStringRefCreate(cStr, xaie.size());
}

MlirStringRef aieTranslateToHSA(MlirOperation moduleOp,
                                MlirStringRef deviceName) {
  std::string xaie;
  llvm::raw_string_ostream os(xaie);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToHSA(
          mod, os, llvm::StringRef(deviceName.data, deviceName.length))))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(xaie.size()));
  xaie.copy(cStr, xaie.size());
  return mlirStringRefCreate(cStr, xaie.size());
}

MlirStringRef aieTranslateToBCF(MlirOperation moduleOp, int col, int row,
                                MlirStringRef deviceName) {
  std::string bcf;
  llvm::raw_string_ostream os(bcf);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToBCF(
          mod, os, col, row,
          llvm::StringRef(deviceName.data, deviceName.length))))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(bcf.size()));
  bcf.copy(cStr, bcf.size());
  return mlirStringRefCreate(cStr, bcf.size());
}

MlirStringRef aieLLVMLink(MlirStringRef *modules, int nModules) {
  std::string ll;
  llvm::raw_string_ostream os(ll);
  std::vector<std::string> files;
  files.reserve(nModules);
  for (int i = 0; i < nModules; ++i)
    files.emplace_back(modules[i].data, modules[i].length);
  if (failed(AIELLVMLink(os, files)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(ll.size()));
  ll.copy(cStr, ll.size());
  return mlirStringRefCreate(cStr, ll.size());
}

DEFINE_C_API_PTR_METHODS(AieRtControl, xilinx::AIE::AIERTControl)

AieRtControl getAieRtControl(AieTargetModel tm) {
  // unwrap the target model
  const AIETargetModel &targetModel =
      *reinterpret_cast<const AIETargetModel *>(tm.d);
  AIERTControl *ctl = new AIERTControl(targetModel);
  return wrap(ctl);
}

void freeAieRtControl(AieRtControl aieCtl) {
  AIERTControl *ctl = unwrap(aieCtl);
  delete ctl;
}

void aieRtDmaUpdateBdAddr(AieRtControl aieCtl, int col, int row, size_t addr,
                          size_t bdId) {
  AIERTControl *ctl = unwrap(aieCtl);
  ctl->dmaUpdateBdAddr(col, row, addr, bdId);
}

void aieRtStartTransaction(AieRtControl aieCtl) {
  AIERTControl *ctl = unwrap(aieCtl);
  ctl->startTransaction();
}

void aieRtExportSerializedTransaction(AieRtControl aieCtl) {
  AIERTControl *ctl = unwrap(aieCtl);
  ctl->exportSerializedTransaction();
}