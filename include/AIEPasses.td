// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
//===- AIEDialect.h - Dialect definition for the AIE IR ----------------===//
//
// Copyright 2019 Xilinx
//
//===---------------------------------------------------------------------===//

#ifndef AIE_PASSES
#define AIE_PASSES

include "mlir/Pass/PassBase.td"

def AIECoreToLLVM : Pass<"aie-llvm-lowering", "ModuleOp"> {
  let summary = "Lowering operations in AIE cores' regions to LLVM";
  let description = [{

    Convert operations from the affine dialect into operations from the SCF and
    standard dialects.

    `affine.for` operations are converted to `scf.for` operations that are free
    of certain structural restrictions (on their bounds and step). `affine.if`
    is similarly converted to the `scf.if` operation. `affine.apply` operations
    are converted into sequences of primitive arithmetic operations from the
    standard dialect that have the same effect, using operands of the `index`
    type. Consequently, named maps and sets thare are no longer in use may be
    removed from the module.

    For example, `%r = affine.apply affine_map<(d0, d1)[s0] -> (d0 + 2*d1 +
    s0)>(%d0, %d1)[%s0]`
    can be converted into:

    ```mlir
    %d0 = <...>
    %d1 = <...>
    %s0 = <...>
    %0 = constant 2 : index
    %1 = muli %0, %d1
    %2 = addi %d0, %1
    %r = addi %2, %s0
    ```

    #### Input invariant

    -   no `Tensor` types;

    These restrictions may be lifted in the future.

    #### Output IR

    Functions with `affine.for` and `affine.if` operations eliminated. These
    functions may contain operations from the Standard dialect in addition to
    those already present before the pass.

    #### Invariants

    -   Functions without a body are not modified.
    -   The semantics of the other functions is preserved.
    -   Individual operations other than those mentioned above are not modified
        if they do not depend on the loop iterator value or on the result of
        `affine.apply`.
  }];
  let constructor = "xilinx::AIE::createAIECoreToLLVMPass()";
  let dependentDialects = [
    "StandardOpsDialect",
    "xilinx::AIE::AIEDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif
