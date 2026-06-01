//===- test_matmul_2col_shim_overload.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Two-column matmul-style topology (4 A inputs + 2 B inputs + 2 C outputs)
// with every ShimNOCTile, MemTile, and CoreTile left unpinned. The placer
// must spread the 6 unpinned shim sources across enough physical shims that
// no single shim exceeds the NPU2 budget of 2 MM2S + 2 S2MM per direction.
//
// Captured from programming_examples/basic/matrix_multiplication/whole_array/
// whole_array_iron.py --n-aie-cols 2 --dev npu2 after stripping every
// `tile=Tile(...)` argument from the IRON Python source.

// RUN: aie-opt --aie-place-tiles --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK-NOT: DMA channel exceeded

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
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_7 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_8 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_9 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_10 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_11 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_12 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_13 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_14 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_15 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_16 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @A_L2L1_0(%logical_mem dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core, %logical_core_0}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo @A_L3L2_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<2048xi16>>
    aie.objectfifo @A_L2L1_1(%logical_mem dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_1, %logical_core_2}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0, @A_L2L1_1]([] [0, 1024])
    aie.objectfifo @A_L2L1_2(%logical_mem_7 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_3, %logical_core_4}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo @A_L3L2_1(%logical_shim_noc_8, {%logical_mem_7}, 2 : i32) : !aie.objectfifo<memref<2048xi16>>
    aie.objectfifo @A_L2L1_3(%logical_mem_7 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_5, %logical_core_6}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_2, @A_L2L1_3]([] [0, 1024])
    aie.objectfifo @B_L2L1_0(%logical_mem_9 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core, %logical_core_1, %logical_core_3, %logical_core_5}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo @B_L3L2_0(%logical_shim_noc_11, {%logical_mem_9}, 2 : i32) : !aie.objectfifo<memref<1024xi16>>
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [0])
    aie.objectfifo @B_L2L1_1(%logical_mem_10 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_0, %logical_core_2, %logical_core_4, %logical_core_6}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>>
    aie.objectfifo @B_L3L2_1(%logical_shim_noc_12, {%logical_mem_10}, 2 : i32) : !aie.objectfifo<memref<1024xi16>>
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [0])
    aie.objectfifo @C_L1L2_0_0(%logical_core, {%logical_mem_13}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    aie.objectfifo @C_L1L2_0_1(%logical_core_1, {%logical_mem_13}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    aie.objectfifo @C_L1L2_0_2(%logical_core_3, {%logical_mem_13}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    aie.objectfifo @C_L1L2_0_3(%logical_core_5, {%logical_mem_13}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    aie.objectfifo @C_L2L3_0(%logical_mem_13 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_15}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_1_0(%logical_core_0, {%logical_mem_14}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    aie.objectfifo @C_L1L2_1_1(%logical_core_2, {%logical_mem_14}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    aie.objectfifo @C_L1L2_1_2(%logical_core_4, {%logical_mem_14}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    aie.objectfifo @C_L1L2_1_3(%logical_core_6, {%logical_mem_14}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>>
    aie.objectfifo @C_L2L3_1(%logical_mem_14 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_16}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 1024, 2048, 3072] [])
    %0 = aie.core(%logical_core) { aie.end } {stack_size = 3328 : i32}
    %1 = aie.core(%logical_core_0) { aie.end } {stack_size = 3328 : i32}
    %2 = aie.core(%logical_core_1) { aie.end } {stack_size = 3328 : i32}
    %3 = aie.core(%logical_core_2) { aie.end } {stack_size = 3328 : i32}
    %4 = aie.core(%logical_core_3) { aie.end } {stack_size = 3328 : i32}
    %5 = aie.core(%logical_core_4) { aie.end } {stack_size = 3328 : i32}
    %6 = aie.core(%logical_core_5) { aie.end } {stack_size = 3328 : i32}
    %7 = aie.core(%logical_core_6) { aie.end } {stack_size = 3328 : i32}
  }
}
