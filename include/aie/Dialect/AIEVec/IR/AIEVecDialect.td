//===- AIEVecDialect.td - AIE vector op definitions--------*- tablegen -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
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
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}
#endif // AIEVEC_DIALECT
