//===-- Passes.td - AIE vector pass definition file --------*- tablegen -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains definitions for passes within the AIEVec/ directory.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES
#define AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SplitVectorLoadUpsChains : Pass<"aievec-split-load-ups-chains"> {
  let summary = "Split 1024-bit vector loads and stores whose halves are "
                "widened or narrowed separately";
  let description = [{
    On AIE2P, splits a 1024-bit vector access into two 512-bit accesses when
    the value is only used, or produced, one half at a time:
    - vector.load feeding aievec.ups (v64i16 -> v64i32);
    - aievec.srs (v64i32 -> v64i16) feeding vector.store;
    - vector.load of v64bf16 feeding only the LHS of bf16 8x8x8
      aievec.matmul_aie2p ops, whose lowering widens the LHS in two halves.
    Each half then gets its own memory access instead of being extracted from
    the full-width value with shuffles.
  }];
  let constructor = "xilinx::aievec::createSplitVectorLoadUpsChainsPass()";
  let dependentDialects = ["mlir::vector::VectorDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::affine::AffineDialect",
                           "xilinx::aievec::AIEVecDialect"];
}

#endif // AIE_DIALECT_AIEVEC_TRANSFORMS_PASSES
