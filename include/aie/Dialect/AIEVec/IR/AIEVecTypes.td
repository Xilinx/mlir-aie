//===- AIEVecTypes.td - AIE vector type definitions --------*- tablegen -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines the set of new AIE vector types
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_TYPES
#define AIEVEC_TYPES

include "aie/Dialect/AIEVec/IR/AIEVecDialect.td"

include "mlir/IR/AttrTypeBase.td"

// Base class for AIE vector types.
class AIEVec_Type<string name, string typeMnemonic, 
               list<Trait> traits = [],
               string baseCppClass = "::xilinx::aievec::AIEVecType">
      : TypeDef<AIEVec_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

#endif // AIEVEC_TYPES
