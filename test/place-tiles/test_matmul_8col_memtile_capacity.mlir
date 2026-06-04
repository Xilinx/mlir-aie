//===- test_matmul_8col_memtile_capacity.mlir ------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Eight-column matmul-style topology with all shim/memtile/compute tiles
// unpinned. The placer must find a valid MemTile assignment for every
// MemTile-anchored ObjectFifo across the 8 available NPU2 memtiles.
//
// Captured from programming_examples/basic/matrix_multiplication/whole_array/
// whole_array_iron.py --n-aie-cols 8 --dev npu2 after stripping every
// `tile=Tile(...)` argument from the IRON Python source.

// RUN: aie-opt --aie-place-tiles %s 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK-NOT: no MemTile with sufficient DMA capacity

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
    %logical_core_15 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_16 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_17 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_18 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_19 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_20 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_21 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_22 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_23 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_24 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_25 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_26 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_27 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_28 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_29 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_30 = aie.logical_tile<CoreTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_31 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_32 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_33 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_34 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_35 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_36 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_37 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_38 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_39 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_40 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_41 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_42 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_43 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_44 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_45 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_46 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_47 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_48 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_49 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_50 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_51 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_52 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_53 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_54 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_55 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_56 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_57 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_58 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_59 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_60 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_61 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_62 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_63 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_64 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_65 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_66 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_67 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_68 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @A_L2L1_0(%logical_mem dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core, %logical_core_0, %logical_core_1, %logical_core_2, %logical_core_3, %logical_core_4, %logical_core_5, %logical_core_6}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @A_L3L2_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [0])
    aie.objectfifo @A_L2L1_1(%logical_mem_31 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_7, %logical_core_8, %logical_core_9, %logical_core_10, %logical_core_11, %logical_core_12, %logical_core_13, %logical_core_14}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @A_L3L2_1(%logical_shim_noc_34, {%logical_mem_31}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [0])
    aie.objectfifo @A_L2L1_2(%logical_mem_32 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_15, %logical_core_16, %logical_core_17, %logical_core_18, %logical_core_19, %logical_core_20, %logical_core_21, %logical_core_22}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @A_L3L2_2(%logical_shim_noc_35, {%logical_mem_32}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [0])
    aie.objectfifo @A_L2L1_3(%logical_mem_33 dimensionsToStream [<size = 8, stride = 128>, <size = 8, stride = 4>, <size = 4, stride = 32>, <size = 4, stride = 1>], {%logical_core_23, %logical_core_24, %logical_core_25, %logical_core_26, %logical_core_27, %logical_core_28, %logical_core_29, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @A_L3L2_3(%logical_shim_noc_36, {%logical_mem_33}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_0(%logical_mem_37 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core, %logical_core_7, %logical_core_15, %logical_core_23}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_0(%logical_shim_noc_45, {%logical_mem_37}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [0])
    aie.objectfifo @B_L2L1_1(%logical_mem_38 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_0, %logical_core_8, %logical_core_16, %logical_core_24}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_1(%logical_shim_noc_46, {%logical_mem_38}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [0])
    aie.objectfifo @B_L2L1_2(%logical_mem_39 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_1, %logical_core_9, %logical_core_17, %logical_core_25}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_2(%logical_shim_noc_47, {%logical_mem_39}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [0])
    aie.objectfifo @B_L2L1_3(%logical_mem_40 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_2, %logical_core_10, %logical_core_18, %logical_core_26}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_3(%logical_shim_noc_48, {%logical_mem_40}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [0])
    aie.objectfifo @B_L2L1_4(%logical_mem_41 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_3, %logical_core_11, %logical_core_19, %logical_core_27}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_4(%logical_shim_noc_49, {%logical_mem_41}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4]([] [0])
    aie.objectfifo @B_L2L1_5(%logical_mem_42 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_4, %logical_core_12, %logical_core_20, %logical_core_28}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_5(%logical_shim_noc_50, {%logical_mem_42}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5]([] [0])
    aie.objectfifo @B_L2L1_6(%logical_mem_43 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_5, %logical_core_13, %logical_core_21, %logical_core_29}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_6(%logical_shim_noc_51, {%logical_mem_43}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6]([] [0])
    aie.objectfifo @B_L2L1_7(%logical_mem_44 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_core_6, %logical_core_14, %logical_core_22, %logical_core_30}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @B_L3L2_7(%logical_shim_noc_52, {%logical_mem_44}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7]([] [0])
    aie.objectfifo @C_L1L2_0_0(%logical_core, {%logical_mem_53}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_0_1(%logical_core_7, {%logical_mem_53}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_0_2(%logical_core_15, {%logical_mem_53}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_0_3(%logical_core_23, {%logical_mem_53}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_0(%logical_mem_53 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_61}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_1_0(%logical_core_0, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_1_1(%logical_core_8, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_1_2(%logical_core_16, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_1_3(%logical_core_24, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_1(%logical_mem_54 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_62}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_2_0(%logical_core_1, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_2_1(%logical_core_9, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_2_2(%logical_core_17, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_2_3(%logical_core_25, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_2(%logical_mem_55 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_63}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_3_0(%logical_core_2, {%logical_mem_56}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_3_1(%logical_core_10, {%logical_mem_56}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_3_2(%logical_core_18, {%logical_mem_56}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_3_3(%logical_core_26, {%logical_mem_56}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_3(%logical_mem_56 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_64}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_4_0(%logical_core_3, {%logical_mem_57}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_4_1(%logical_core_11, {%logical_mem_57}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_4_2(%logical_core_19, {%logical_mem_57}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_4_3(%logical_core_27, {%logical_mem_57}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_4(%logical_mem_57 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_65}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_4_0, @C_L1L2_4_1, @C_L1L2_4_2, @C_L1L2_4_3] -> [@C_L2L3_4]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_5_0(%logical_core_4, {%logical_mem_58}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_5_1(%logical_core_12, {%logical_mem_58}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_5_2(%logical_core_20, {%logical_mem_58}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_5_3(%logical_core_28, {%logical_mem_58}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_5(%logical_mem_58 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_66}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_5_0, @C_L1L2_5_1, @C_L1L2_5_2, @C_L1L2_5_3] -> [@C_L2L3_5]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_6_0(%logical_core_5, {%logical_mem_59}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_6_1(%logical_core_13, {%logical_mem_59}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_6_2(%logical_core_21, {%logical_mem_59}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_6_3(%logical_core_29, {%logical_mem_59}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_6(%logical_mem_59 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_67}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_6_0, @C_L1L2_6_1, @C_L1L2_6_2, @C_L1L2_6_3] -> [@C_L2L3_6]([0, 1024, 2048, 3072] [])
    aie.objectfifo @C_L1L2_7_0(%logical_core_6, {%logical_mem_60}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_7_1(%logical_core_14, {%logical_mem_60}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_7_2(%logical_core_22, {%logical_mem_60}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L1L2_7_3(%logical_core_30, {%logical_mem_60}, 2 : i32) : !aie.objectfifo<memref<32x32xi32>> 
    aie.objectfifo @C_L2L3_7(%logical_mem_60 dimensionsToStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%logical_shim_noc_68}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    aie.objectfifo.link [@C_L1L2_7_0, @C_L1L2_7_1, @C_L1L2_7_2, @C_L1L2_7_3] -> [@C_L2L3_7]([0, 1024, 2048, 3072] [])
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
    %16 = aie.core(%logical_core_15) { aie.end } {stack_size = 3328 : i32}
    %17 = aie.core(%logical_core_16) { aie.end } {stack_size = 3328 : i32}
    %18 = aie.core(%logical_core_17) { aie.end } {stack_size = 3328 : i32}
    %19 = aie.core(%logical_core_18) { aie.end } {stack_size = 3328 : i32}
    %20 = aie.core(%logical_core_19) { aie.end } {stack_size = 3328 : i32}
    %21 = aie.core(%logical_core_20) { aie.end } {stack_size = 3328 : i32}
    %22 = aie.core(%logical_core_21) { aie.end } {stack_size = 3328 : i32}
    %23 = aie.core(%logical_core_22) { aie.end } {stack_size = 3328 : i32}
    %24 = aie.core(%logical_core_23) { aie.end } {stack_size = 3328 : i32}
    %25 = aie.core(%logical_core_24) { aie.end } {stack_size = 3328 : i32}
    %26 = aie.core(%logical_core_25) { aie.end } {stack_size = 3328 : i32}
    %27 = aie.core(%logical_core_26) { aie.end } {stack_size = 3328 : i32}
    %28 = aie.core(%logical_core_27) { aie.end } {stack_size = 3328 : i32}
    %29 = aie.core(%logical_core_28) { aie.end } {stack_size = 3328 : i32}
    %30 = aie.core(%logical_core_29) { aie.end } {stack_size = 3328 : i32}
    %31 = aie.core(%logical_core_30) { aie.end } {stack_size = 3328 : i32}
  }
}
