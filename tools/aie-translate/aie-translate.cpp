//===- aie-translate.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

using namespace mlir;

int main(int argc, char **argv) {
//  registerAllDialects(getContext());
//#  registry.insert<scf::SCFDialect>();

  registerAllTranslations();
  xilinx::AIE::registerAIETranslations();
  xilinx::aievec::registerAIEVecToCppTranslation();

  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
