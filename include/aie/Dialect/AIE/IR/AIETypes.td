//===- AIETypes.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TYPES
#define AIE_TYPES

include "aie/Dialect/AIE/IR/AIE.td"
include "aie/Dialect/AIE/IR/AIEAttrs.td"

include "mlir/IR/AttrTypeBase.td"

def AIE_ObjectFifo_Depth : AnyAttrOf<[ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>, ArrayAttr]>;

def AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                          "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[Index.predicate,
                                   AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                  "scalar">;

#endif // AIE_TYPES