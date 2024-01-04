//===- aie-translate.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

#ifndef NDEBUG

namespace {
struct CreateDebug {
  static void *call() {
    return new llvm::cl::opt<bool, true>(
        "debug", llvm::cl::desc("Enable debug output"), llvm::cl::Hidden,
        llvm::cl::location(llvm::DebugFlag));
  }
};
} // namespace

static llvm::ManagedStatic<llvm::cl::opt<bool, true>, CreateDebug> Debug;

#endif

int main(int argc, char **argv) {
#ifndef NDEBUG
  *Debug;
  if (Debug->getNumOccurrences())
    llvm::DebugFlag = true;
#endif

  registerAllTranslations();
  xilinx::AIE::registerAIETranslations();
  xilinx::aievec::registerAIEVecToCppTranslation();

  return failed(mlirTranslateMain(argc, argv, "AIE Translation Tool"));
}
