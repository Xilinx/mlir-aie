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
#include "aie/Targets/AIETargets.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
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
                                          bool bigEndian, bool emitUnified,
                                          bool cdoDebug, bool aieSim,
                                          bool xaieDebug, bool enableCores) {
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  auto status = AIETranslateToCDODirect(
      mod, llvm::StringRef(workDirPath.data, workDirPath.length), bigEndian,
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

MlirStringRef aieTranslateToNPU(MlirOperation moduleOp) {
  std::string npu;
  llvm::raw_string_ostream os(npu);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToNPU(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(npu.size()));
  npu.copy(cStr, npu.size());
  return mlirStringRefCreate(cStr, npu.size());
}

MlirStringRef aieTranslateToXAIEV2(MlirOperation moduleOp) {
  std::string xaie;
  llvm::raw_string_ostream os(xaie);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToXAIEV2(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(xaie.size()));
  xaie.copy(cStr, xaie.size());
  return mlirStringRefCreate(cStr, xaie.size());
}

MlirStringRef aieTranslateToHSA(MlirOperation moduleOp) {
  std::string xaie;
  llvm::raw_string_ostream os(xaie);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToHSA(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(xaie.size()));
  xaie.copy(cStr, xaie.size());
  return mlirStringRefCreate(cStr, xaie.size());
}

MlirStringRef aieTranslateToBCF(MlirOperation moduleOp, int col, int row) {
  std::string bcf;
  llvm::raw_string_ostream os(bcf);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToBCF(mod, os, col, row)))
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
