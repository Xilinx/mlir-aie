// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#ifndef AIE_NORMALIZE_ADDRESS_SPACES
#define AIE_NORMALIZE_ADDRESS_SPACES

include "AIE.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

def toDefaultAddressSpace : NativeCodeCall<"TypeAttr::get(memRefToDefaultAddressSpace($0.getValue()))">;
def hasNonDefaultAddressSpace : Constraint<
    CPred<"$0.getValue().cast<MemRefType>().getMemorySpace() != 0">,
    "has non-default address space">;
def : Pat<(GlobalMemrefOp $sym_name, $sym_visibility, $type, $initial_value, $constant),
        (GlobalMemrefOp $sym_name, $sym_visibility, (toDefaultAddressSpace $type), $initial_value, $constant),
        [(hasNonDefaultAddressSpace $type)],
        (addBenefit 20)>;

#endif // AIE_NORMALIZE_ADDRESS_SPACES
