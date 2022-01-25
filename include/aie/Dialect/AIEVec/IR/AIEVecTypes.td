//===- AIEVecTypes.td - AIE vector type definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// 
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// Defines the set of new AIE vector types
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_TYPES
#define AIEVEC_TYPES

include "aie/Dialect/AIEVec/IR/AIEVecDialect.td"

// Base class for AIE vector types.
class AIEVec_Type<string name, string typeMnemonic, 
               list<Trait> traits = [],
               string baseCppClass = "::xilinx::aievec::AIEVecType">
      : TypeDef<AIEVec_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Accumulation Types
//===----------------------------------------------------------------------===//

def AIEVec_Acc : AIEVec_Type<"Acc", "acc"> {
  let summary = "scalar or vector 48/80 bit accumulator type";
  let parameters = (ins "int32_t":$lanes, "Type":$valueType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "int32_t":$lanes, "Type":$valueType), [{
      return $_get(valueType.getContext(), lanes, valueType);
    }]>
  ];
  let skipDefaultBuilders = 1;
}

#endif // AIEVEC_TYPES
