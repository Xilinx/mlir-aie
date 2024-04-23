//===- XCLBinGen.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===---------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <string>

#pragma once

namespace xilinx {

struct XCLBinGenConfig {
  std::string TargetArch;
  std::string PeanoDir;
  std::string InstallDir;
  std::string AIEToolsDir;
  std::string TempDir;
  bool Verbose;
  std::string HostArch;
  std::string XCLBinKernelName;
  std::string XCLBinKernelID;
  std::string XCLBinInstanceName;
  bool UseChess = false;
  bool DisableThreading = false;
  bool PrintIRAfterAll = false;
  bool PrintIRBeforeAll = false;
  bool PrintIRModuleScope = false;
};

void findVitis(XCLBinGenConfig &TK);

mlir::LogicalResult aie2xclbin(mlir::MLIRContext *ctx, mlir::ModuleOp moduleOp,
                               XCLBinGenConfig &TK, llvm::StringRef OutputNPU,
                               llvm::StringRef OutputXCLBin);

} // namespace xilinx
