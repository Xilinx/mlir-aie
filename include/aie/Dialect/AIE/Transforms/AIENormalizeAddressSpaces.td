//===- AIENormalizeAddressSpaces.td ------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_NORMALIZE_ADDRESS_SPACES
#define AIE_NORMALIZE_ADDRESS_SPACES

include "aie/Dialect/AIE/IR/AIE.td"

include "mlir/Dialect/MemRef/IR/MemRefOps.td"
include "mlir/IR/PatternBase.td"

def toDefaultAddressSpace : NativeCodeCall<"TypeAttr::get(memRefToDefaultAddressSpace($0.getValue()))">;
def hasNonDefaultAddressSpace : Constraint<
    CPred<"llvm::cast<mlir::MemRefType>($0.getValue()).getMemorySpace() != 0">,
    "has non-default address space">;
def : Pat<
        /*pattern*/ (MemRef_GlobalOp $sym_name, $sym_visibility, $type, $initial_value, $constant, $attrs),
        /*result*/ (MemRef_GlobalOp $sym_name, $sym_visibility, (toDefaultAddressSpace $type), $initial_value, $constant, $attrs),
        /*preds*/ [(hasNonDefaultAddressSpace $type)],
        /*supplemental_results*/ [],
        /*benefitAdded*/ (addBenefit 20)>;

#endif // AIE_NORMALIZE_ADDRESS_SPACES
