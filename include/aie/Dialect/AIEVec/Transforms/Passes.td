//===-- Passes.td - AIE vector pass definition file --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file contains definitions for passes within the AIEVec/ directory.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES
#define AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AIEVectorize : Pass<"aie-vectorize", "ModuleOp"> {
  let summary = "Vectorize the output of affine "
                "supervectorizer to AIE vector abstraction";
  let constructor = "xilinx::aievec::createAIEVectorizePass()";
  let dependentDialects = ["AffineDialect", 
                           "xilinx::aievec::AIEVecDialect",
                           "arith::ArithDialect",
                           "memref::MemRefDialect",
                           "scf::SCFDialect", 
                           "vector::VectorDialect"];
  let options = [
    Option<"shiftParam", "shift", "unsigned", /*default=*/"0",
      "Shift parameter for rounding and saturation">,
    Option<"zeroOffset", "zero-offset", "unsigned", /*default=*/"0",
     "Zero offset for indicating the location of zeroes in convolution filter "
     "(useful for 16x16 scheme)">,
    Option<"dupFactor", "dup-factor", "unsigned", /*default=*/"2",
     "Duplication factor for each value in convolution filter "
     "(useful for 8x8 scheme)">,
  ];
}

def LowerVectorToAIEVec : Pass<"lower-vector-to-aievec", "func::FuncOp"> {
  let summary = "Convert standard Vector dialect operations to equivalent AIE "
                "vector ops";
  let dependentDialects = ["AffineDialect", // NOTE: Do we need this?
                           "xilinx::aievec::AIEVecDialect",
                           "arith::ArithDialect",
                           "memref::MemRefDialect",
                           "scf::SCFDialect",
                           "vector::VectorDialect"];
  let options = [
    Option<"shiftParam", "shift", "unsigned", /*default=*/"0",
      "Shift parameter for rounding and saturation">,
    Option<"zeroOffset", "zero-offset", "unsigned", /*default=*/"0",
     "Zero offset for indicating the location of zeroes in convolution filter "
     "(useful for 16x16 scheme)">,
    Option<"dupFactor", "dup-factor", "unsigned", /*default=*/"2",
     "Duplication factor for each value in convolution filter "
     "(useful for 8x8 scheme)">,
    Option<"aieTarget", "aie-target", "std::string", /*default=*/"\"aie\"",
     "Select AIE version: \\\"aie\\\" or \\\"aieml\\\". This will determine "
     "the vector size and available operations.">,
  ];
}

def CanonicalizeForAIEVec : Pass<"canonicalize-for-aievec", "func::FuncOp"> {
  let summary = "Transform common vector ops into a subset of vector ops with "
                "standard vector ops closer to AIEVec semantics.";
  let dependentDialects = ["arith::ArithDialect",
                           "vector::VectorDialect"];
  let options = [
    Option<"aieTarget", "aie-target", "std::string", /*default=*/"\"aie\"",
     "Select AIE version: \\\"aie\\\" or \\\"aieml\\\". This will determine "
     "the vector size and available operations.">,
  ];
}

def RedundantLoadStoreOptimization : Pass<"redundant-load-store-optimization", "func::FuncOp"> {
  let summary = "Set inbounds for loads and stores.";
  let dependentDialects = ["vector::VectorDialect"];
}

def TransferOpflowOpt : Pass<"transfer-op-flow-opt", "func::FuncOp"> {
  let summary = "Implement transfer op write to read forwarding and " 
                "dead transfer write optimizations.";
  let dependentDialects = ["vector::VectorDialect"];
}

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES
