//===- ADFPasses.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef ADF_PASSES
#define ADF_PASSES

include "mlir/Pass/PassBase.td"

def ADFGenerateCppGraph : Pass<"adf-generate-cpp-graph", "ModuleOp"> {
  let summary = "ADF c++ graph generation";
  let description = [{
    This pass generates an ADF C++ graph and the external kernel function
    header file.

  }];

  let constructor = "xilinx::ADF::createADFGenerateCppGraphPass()";
  let dependentDialects = [
    "xilinx::ADF::ADFDialect",
    "StandardOpsDialect",
  ];
}

#endif 