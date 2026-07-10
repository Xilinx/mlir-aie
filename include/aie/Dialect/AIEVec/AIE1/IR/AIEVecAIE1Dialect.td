//===- AIEVecAIE1Dialect.td - AIE1 vector op definitions---*- tablegen -*-====//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines AIE1 vector dialect.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_AIE1_DIALECT
#define AIEVEC_AIE1_DIALECT

include "mlir/IR/OpBase.td"

def AIEVec_AIE1_Dialect : Dialect {
  let name = "aievec_aie1";
  let summary = "Types and operations for AIE1 vector dialect";
  let cppNamespace = "::xilinx::aievec::aie1";
}

#endif // AIEVEC_AIE1_DIALECT
