//===- AIEVecDialect.td - AIE vector op definitions--------*- tablegen -*-====//
//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines AIE vector dialect.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_DIALECT
#define AIEVEC_DIALECT

include "mlir/IR/OpBase.td"

def AIEVec_Dialect : Dialect {
  let name = "aievec";
  let summary = "Types and operations for AIE vector dialect";
  let cppNamespace = "::xilinx::aievec";

  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // AIEVEC_DIALECT
