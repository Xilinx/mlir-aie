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

    Outline code inside a particular AIE.core operation into the llvm dialect.
  }];
  let options = [
    Option<"tileCol", "tilecol", "unsigned",
           /*default=*/"-1", "X coordinate of tile to generate code for">,
    Option<"tileRow", "tilerow", "unsigned",
           /*default=*/"-1", "Y coordinate of tile to generate code for">
  ];

  let constructor = "xilinx::AIE::createAIECoreToLLVMPass()";
  let dependentDialects = [
    "StandardOpsDialect",
    "xilinx::AIE::AIEDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def AIECoreToStandard : Pass<"aie-standard-lowering", "ModuleOp"> {
  let summary = "Lowering operations in AIE cores' regions to Standard";
  let description = [{

    Outline code inside a particular AIE.core operation into the llvm dialect.
    BufferOp operations are converted to a GlobalMemrefOp and references to
    those buffers are converted to GetGlobalMemrefOp.  Other AIE operations
    inside the cores are generally lowered to appropriate function intrinsics.
    Other AIE operations (e.g. CoreOp, TileOp, LockOp) outside the core are removed.

  }];
  let options = [
    Option<"tileCol", "tilecol", "unsigned",
           /*default=*/"-1", "X coordinate of tile to generate code for">,
    Option<"tileRow", "tilerow", "unsigned",
           /*default=*/"-1", "Y coordinate of tile to generate code for">
  ];

  let constructor = "xilinx::AIE::createAIECoreToStandardPass()";
  let dependentDialects = [
    "StandardOpsDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIENormalizeAddressSpaces : Pass<"aie-normalize-address-spaces", "ModuleOp"> {
  let summary = "Remove non-default address spaces";
  let description = [{
    Early in the flow, it is convenient to represent multiple memories using different
    address spaces.  However, after outlining code for AIE engines, the core itself only
    has access to a single address space.  To avoid confusion, this pass normalizes
    any address spaces remaining in the code, converting them to the default address
    space.
  }];

  let constructor = "xilinx::AIE::createAIENormalizeAddressSpacesPass()";
  let dependentDialects = [
    "StandardOpsDialect",
    "xilinx::AIE::AIEDialect",
  ];
}
#endif
