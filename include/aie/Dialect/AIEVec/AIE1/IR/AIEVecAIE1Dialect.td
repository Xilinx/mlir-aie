//===- AIEVecAIE1Dialect.td - AIE1 vector op definitions---*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
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
