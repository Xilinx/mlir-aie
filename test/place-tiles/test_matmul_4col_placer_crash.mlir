//===- test_matmul_4col_placer_crash.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Four-column matmul-style topology with all shim/memtile/compute tiles
// unpinned. The placer pass and the immediately-following
// AIEObjectFifoStatefulTransform pass must complete cleanly (no diagnostic,
// no crash) for this design.
//
// Captured from programming_examples/basic/matrix_multiplication/whole_array/
// whole_array_iron.py --n-aie-cols 4 --dev npu2 after stripping every
// `tile=Tile(...)` argument from the IRON Python source.

// RUN: aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK-NOT: PLEASE submit a bug report
// CHECK-NOT: Aborted

module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_0 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_1 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_2 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_3 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_4 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_5 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_6 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_7 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_8 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_9 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_10 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_11 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_12 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_13 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_14 = aie.logical_tile<CoreTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_15 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_16 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_17 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_18 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_19 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_20 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_21 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_22 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_23 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_24 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_25 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_26 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_27 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_28 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_29 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_30 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_31 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_32 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_33 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_34 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_35 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_36 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @A_L2L1_0(%logical_mem dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core, %logical_core_0, %logical_core_1, %logical_core_2}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @A_L3L2_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [0])
    aie.objectfifo @A_L2L1_1(%logical_mem_15 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_3, %logical_core_4, %logical_core_5, %logical_core_6}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @A_L3L2_1(%logical_shim_noc_18, {%logical_mem_15}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [0])
    aie.objectfifo @A_L2L1_2(%logical_mem_16 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_7, %logical_core_8, %logical_core_9, %logical_core_10}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @A_L3L2_2(%logical_shim_noc_19, {%logical_mem_16}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [0])
    aie.objectfifo @A_L2L1_3(%logical_mem_17 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_11, %logical_core_12, %logical_core_13, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @A_L3L2_3(%logical_shim_noc_20, {%logical_mem_17}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_0(%logical_mem_21 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core, %logical_core_3, %logical_core_7, %logical_core_11}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_0(%logical_shim_noc_25, {%logical_mem_21}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [0])
    aie.objectfifo @B_L2L1_1(%logical_mem_22 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_0, %logical_core_4, %logical_core_8, %logical_core_12}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_1(%logical_shim_noc_26, {%logical_mem_22}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [0])
    aie.objectfifo @B_L2L1_2(%logical_mem_23 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_1, %logical_core_5, %logical_core_9, %logical_core_13}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_2(%logical_shim_noc_27, {%logical_mem_23}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [0])
    aie.objectfifo @B_L2L1_3(%logical_mem_24 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_2, %logical_core_6, %logical_core_10, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_3(%logical_shim_noc_28, {%logical_mem_24}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [0])
    aie.objectfifo @C_L1L2_0_0(%logical_core, {%logical_mem_29}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_0_1(%logical_core_3, {%logical_mem_29}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_0_2(%logical_core_7, {%logical_mem_29}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_0_3(%logical_core_11, {%logical_mem_29}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_0(%logical_mem_29 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_33}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_1_0(%logical_core_0, {%logical_mem_30}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_1_1(%logical_core_4, {%logical_mem_30}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_1_2(%logical_core_8, {%logical_mem_30}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_1_3(%logical_core_12, {%logical_mem_30}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_1(%logical_mem_30 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_34}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_2_0(%logical_core_1, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_2_1(%logical_core_5, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_2_2(%logical_core_9, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_2_3(%logical_core_13, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_2(%logical_mem_31 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_35}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_3_0(%logical_core_2, {%logical_mem_32}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_3_1(%logical_core_6, {%logical_mem_32}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_3_2(%logical_core_10, {%logical_mem_32}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_3_3(%logical_core_14, {%logical_mem_32}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_3(%logical_mem_32 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_36}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 1024, 2048, 3072] [])
    %0 = aie.core(%logical_core) { aie.end } {stack_size = 3328 : i32}
    %1 = aie.core(%logical_core_0) { aie.end } {stack_size = 3328 : i32}
    %2 = aie.core(%logical_core_1) { aie.end } {stack_size = 3328 : i32}
    %3 = aie.core(%logical_core_2) { aie.end } {stack_size = 3328 : i32}
    %4 = aie.core(%logical_core_3) { aie.end } {stack_size = 3328 : i32}
    %5 = aie.core(%logical_core_4) { aie.end } {stack_size = 3328 : i32}
    %6 = aie.core(%logical_core_5) { aie.end } {stack_size = 3328 : i32}
    %7 = aie.core(%logical_core_6) { aie.end } {stack_size = 3328 : i32}
    %8 = aie.core(%logical_core_7) { aie.end } {stack_size = 3328 : i32}
    %9 = aie.core(%logical_core_8) { aie.end } {stack_size = 3328 : i32}
    %10 = aie.core(%logical_core_9) { aie.end } {stack_size = 3328 : i32}
    %11 = aie.core(%logical_core_10) { aie.end } {stack_size = 3328 : i32}
    %12 = aie.core(%logical_core_11) { aie.end } {stack_size = 3328 : i32}
    %13 = aie.core(%logical_core_12) { aie.end } {stack_size = 3328 : i32}
    %14 = aie.core(%logical_core_13) { aie.end } {stack_size = 3328 : i32}
    %15 = aie.core(%logical_core_14) { aie.end } {stack_size = 3328 : i32}
  }
}
