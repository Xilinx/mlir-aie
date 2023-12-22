//===- Translation.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aie-c/Translation.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;
using namespace mlir;
using namespace xilinx::AIE;

MlirStringRef aieTranslateAIEVecToCpp(MlirOperation op, bool aieml) {
  std::string cpp;
  llvm::raw_string_ostream os(cpp);
  mlir::Operation *op_ = unwrap(op);
  if (failed(xilinx::aievec::translateAIEVecToCpp(op_, aieml, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(cpp.size()));
  cpp.copy(cStr, cpp.size());
  return mlirStringRefCreate(cStr, cpp.size());
};

MlirStringRef aieTranslateModuleToLLVMIR(MlirOperation op) {
  std::string llvmir;
  llvm::raw_string_ostream os(llvmir);
  Operation *op_ = unwrap(op);
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(op_, llvmContext);
  if (!llvmModule)
    return mlirStringRefCreate(nullptr, 0);
  llvmModule->print(os, nullptr);
  char *cStr = static_cast<char *>(malloc(llvmir.size()));
  llvmir.copy(cStr, llvmir.size());
  return mlirStringRefCreate(cStr, llvmir.size());
}

MlirStringRef aieTranslateToCDO(MlirOperation op) {
  std::string cdo;
  llvm::raw_string_ostream os(cdo);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(op));
  if (failed(AIETranslateToCDO(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(cdo.size()));
  cdo.copy(cStr, cdo.size());
  return mlirStringRefCreate(cStr, cdo.size());
}

MlirStringRef aieTranslateToIPU(MlirOperation op) {
  std::string ipu;
  llvm::raw_string_ostream os(ipu);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(op));
  if (failed(AIETranslateToIPU(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(ipu.size()));
  ipu.copy(cStr, ipu.size());
  return mlirStringRefCreate(cStr, ipu.size());
}

MlirStringRef aieTranslateToXAIEV2(MlirOperation op) {
  std::string xaie;
  llvm::raw_string_ostream os(xaie);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(op));
  if (failed(AIETranslateToXAIEV2(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(xaie.size()));
  xaie.copy(cStr, xaie.size());
  return mlirStringRefCreate(cStr, xaie.size());
}

MlirStringRef aieTranslateToBCF(MlirOperation op, int col, int row) {
  std::string bcf;
  llvm::raw_string_ostream os(bcf);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(op));
  if (failed(AIETranslateToBCF(mod, os, col, row)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(bcf.size()));
  bcf.copy(cStr, bcf.size());
  return mlirStringRefCreate(cStr, bcf.size());
}
