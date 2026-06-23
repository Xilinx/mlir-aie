//===-- Passes.td - AIE vector pass definition file --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
// This file contains definitions for passes within the AIEVec/ directory.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES
#define AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

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
