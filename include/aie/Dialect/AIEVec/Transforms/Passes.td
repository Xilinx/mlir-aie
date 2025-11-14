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

def AIEVectorize : Pass<"aie-vectorize", "mlir::ModuleOp"> {
  let summary = "Vectorize the output of affine "
                "supervectorizer to AIE vector abstraction";
  let constructor = "xilinx::aievec::createAIEVectorizePass()";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "xilinx::aievec::AIEVecDialect",
    "xilinx::aievec::aie1::AIEVecAIE1Dialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect"
  ];
  let options = [
    Option<"shiftParam", "shift", "unsigned", /*default=*/"0",
      "Shift parameter for rounding and saturation">,
    Option<"zeroOffset", "zero-offset", "unsigned", /*default=*/"0",
     "Zero offset for indicating the location of zeroes in convolution filter "
     "(useful for 16x16 scheme)">,
    Option<"dupFactor", "dup-factor", "unsigned", /*default=*/"2",
     "Duplication factor for each value in convolution filter "
     "(useful for 8x8 scheme)">,
    Option<"unalignedLoadsCheck", "unaligned-loads-check", "bool", /*default=*/"true",
     "Enable the unaligned loads check.">,
    Option<"aieml", "aieml", "bool", /*default=*/"false", "">,
  ];
}

def SplitVectorLoadUpsChains : Pass<"aievec-split-load-ups-chains"> {
  let summary = "Split vector.load + aievec.ups chains to reduce shuffle operations";
  let description = [{
    This pass optimizes chains of vector.load followed by aievec.ups operations
    for AIE2p targets. Instead of loading a 1024-bit vector and then shuffling
    it into two halves for separate UPS operations (3 shuffles total), it splits
    both the load and UPS into two 512-bit halves, requiring only 1 shuffle for
    concatenation.
  }];
  let constructor = "xilinx::aievec::createSplitVectorLoadUpsChainsPass()";
  let dependentDialects = [
    "mlir::vector::VectorDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::affine::AffineDialect",
    "xilinx::aievec::AIEVecDialect"
  ];
}

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES
