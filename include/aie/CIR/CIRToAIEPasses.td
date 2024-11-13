//===- CIRToAIEpasses.td -----------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CIR_PASSES
#define AIE_CIR_PASSES

include "mlir/Pass/PassBase.td"

def CIRToAIEPrepare : Pass<"cir-to-aie-prepare", "mlir::ModuleOp"> {
  let summary = "Prepare some transformations of some AIE++ patterns in CIR into AIE equivalent operations";
  let description = [{
    Some AIE constructs are represented in AIE++ by C++ abstractions expressed
    in normal C++. These are compiled to CIR as plain C++ constructs.
    This pass recognize these AIE++ C++ constructs and replace them by
    equivalent AIE operations.
  }];

  let constructor = "xilinx::AIE::CIR::createCIRToAIEPreparePass()";
  let dependentDialects = [
    "cir::CIRDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def CIRToAIE : Pass<"cir-to-aie", "mlir::ModuleOp"> {
  let summary = "Transform some AIE++ patterns in CIR into AIE equivalent operations";
  let description = [{
    Some AIE constructs are represented in AIE++ by C++ abstractions expressed
    in normal C++. These are compiled to CIR as plain C++ constructs.
    This pass recognize these AIE++ C++ constructs and replace them by
    equivalent AIE operations.
  }];

  let constructor = "xilinx::AIE::CIR::createCIRToAIEPass()";
  let dependentDialects = [
    "cir::CIRDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

#endif
