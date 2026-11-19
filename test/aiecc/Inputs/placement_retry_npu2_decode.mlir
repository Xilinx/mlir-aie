//===- placement_retry_npu2_decode.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// IRON's npu2 decode pack (an 8-lane MHA, two GEMVs and a copy, 17 cores),
// its core bodies reduced to their objectFIFO accesses. Four cores a column
// leave more flows between columns 0-2 than the router fits; three route.
// Input of placement_retry.test.

module {
  aie.device(npu2) @d0 {
    func.func private @never_called(memref<16xi32>) attributes {link_with = "never_called.o"}
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
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_16 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_17 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_18 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_19 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_20 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_21 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_22 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_23 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_24 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_25 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_26 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_27 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_28 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_29 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_30 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_31 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_32 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_33 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_34 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_35 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_36 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_37 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_38 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_39 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_40 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_41 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_42 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_43 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_44 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_45 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_46 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_47 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_48 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_49 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_50 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_51 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_52 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_53 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_54 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_55 = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @inKV_0(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo @memKV_0(%logical_mem dimensionsToStream [<size = 4, stride = 2048>, <size = 32, stride = 8>, <size = 8, stride = 256>, <size = 8, stride = 1>], {%logical_core}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo.link [@inKV_0] -> [@memKV_0]([] [0])
    aie.objectfifo @inKV_1(%logical_shim_noc_16, {%logical_mem_17}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo @memKV_1(%logical_mem_17 dimensionsToStream [<size = 4, stride = 2048>, <size = 32, stride = 8>, <size = 8, stride = 256>, <size = 8, stride = 1>], {%logical_core_0}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo.link [@inKV_1] -> [@memKV_1]([] [0])
    aie.objectfifo @inKV_2(%logical_shim_noc_18, {%logical_mem_19}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo @memKV_2(%logical_mem_19 dimensionsToStream [<size = 4, stride = 2048>, <size = 32, stride = 8>, <size = 8, stride = 256>, <size = 8, stride = 1>], {%logical_core_1}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo.link [@inKV_2] -> [@memKV_2]([] [0])
    aie.objectfifo @inKV_3(%logical_shim_noc_20, {%logical_mem_21}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo @memKV_3(%logical_mem_21 dimensionsToStream [<size = 4, stride = 2048>, <size = 32, stride = 8>, <size = 8, stride = 256>, <size = 8, stride = 1>], {%logical_core_2}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo.link [@inKV_3] -> [@memKV_3]([] [0])
    aie.objectfifo @inKV_4(%logical_shim_noc_22, {%logical_mem_23}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo @memKV_4(%logical_mem_23 dimensionsToStream [<size = 4, stride = 2048>, <size = 32, stride = 8>, <size = 8, stride = 256>, <size = 8, stride = 1>], {%logical_core_3}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo.link [@inKV_4] -> [@memKV_4]([] [0])
    aie.objectfifo @inKV_5(%logical_shim_noc_24, {%logical_mem_25}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo @memKV_5(%logical_mem_25 dimensionsToStream [<size = 4, stride = 2048>, <size = 32, stride = 8>, <size = 8, stride = 256>, <size = 8, stride = 1>], {%logical_core_4}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo.link [@inKV_5] -> [@memKV_5]([] [0])
    aie.objectfifo @inKV_6(%logical_shim_noc_26, {%logical_mem_27}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo @memKV_6(%logical_mem_27 dimensionsToStream [<size = 4, stride = 2048>, <size = 32, stride = 8>, <size = 8, stride = 256>, <size = 8, stride = 1>], {%logical_core_5}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo.link [@inKV_6] -> [@memKV_6]([] [0])
    aie.objectfifo @inKV_7(%logical_shim_noc_28, {%logical_mem_29}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo @memKV_7(%logical_mem_29 dimensionsToStream [<size = 4, stride = 2048>, <size = 32, stride = 8>, <size = 8, stride = 256>, <size = 8, stride = 1>], {%logical_core_6}, 2 : i32) : !aie.objectfifo<memref<32x256xbf16>>
    aie.objectfifo.link [@inKV_7] -> [@memKV_7]([] [0])
    aie.objectfifo @made_0(%logical_mem_30 dimensionsToStream [<size = 8, stride = 8>, <size = 16, stride = 64>, <size = 8, stride = 1>], {%logical_shim_noc_31}, 2 : i32) : !aie.objectfifo<memref<8x128xbf16>>
    aie.objectfifo @outO(%logical_core_7, {%logical_mem_30}, 1 : i32) : !aie.objectfifo<memref<8x128xbf16>>
    aie.objectfifo.link [@outO] -> [@made_0]([0] [])
    aie.objectfifo @made_1(%logical_shim_noc_32, {%logical_core_8}, 2 : i32) : !aie.objectfifo<memref<2x1536xbf16>>
    aie.objectfifo @made_2(%logical_shim_noc_33, {%logical_core_9}, 2 : i32) : !aie.objectfifo<memref<2x1536xbf16>>
    aie.objectfifo @made_3(%logical_shim_noc_34, {%logical_core_10}, 2 : i32) : !aie.objectfifo<memref<2x1536xbf16>>
    aie.objectfifo @made_4(%logical_shim_noc_35, {%logical_core_11}, 2 : i32) : !aie.objectfifo<memref<2x1536xbf16>>
    aie.objectfifo @made_5(%logical_shim_noc_36, {%logical_core_12}, 2 : i32) : !aie.objectfifo<memref<2x1536xbf16>>
    aie.objectfifo @made_6(%logical_shim_noc_37, {%logical_core_13}, 2 : i32) : !aie.objectfifo<memref<2x1536xbf16>>
    aie.objectfifo @made_7(%logical_shim_noc_38, {%logical_core_14}, 2 : i32) : !aie.objectfifo<memref<2x1536xbf16>>
    aie.objectfifo @made_8(%logical_shim_noc_39, {%logical_core_15}, 2 : i32) : !aie.objectfifo<memref<2x1536xbf16>>
    aie.objectfifo @made_9(%logical_core_8, {%logical_shim_noc_40}, 2 : i32) : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @made_10(%logical_core_9, {%logical_shim_noc_41}, 2 : i32) : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @made_11(%logical_core_10, {%logical_shim_noc_42}, 2 : i32) : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @made_12(%logical_core_11, {%logical_shim_noc_43}, 2 : i32) : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @made_13(%logical_core_12, {%logical_shim_noc_44}, 2 : i32) : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @made_14(%logical_core_13, {%logical_shim_noc_45}, 2 : i32) : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @made_15(%logical_core_14, {%logical_shim_noc_46}, 2 : i32) : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @made_16(%logical_core_15, {%logical_shim_noc_47}, 2 : i32) : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @made_17(%logical_shim_noc_48, {%logical_mem_49}, 1 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.objectfifo @fifo_out_0(%logical_mem_49, {%logical_shim_noc_52}, 1 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.objectfifo.link [@made_17] -> [@fifo_out_0]([] [0])
    aie.objectfifo @made_18(%logical_shim_noc_50, {%logical_mem_51}, 1 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.objectfifo @fifo_out_1(%logical_mem_51, {%logical_shim_noc_53}, 1 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.objectfifo.link [@made_18] -> [@fifo_out_1]([] [0])
    aie.objectfifo @of0(%logical_core, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<1056xf32>>
    aie.objectfifo @of1(%logical_core_0, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<1056xf32>>
    aie.objectfifo @of2(%logical_core_1, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<1056xf32>>
    aie.objectfifo @of3(%logical_core_2, {%logical_mem_54}, 2 : i32) : !aie.objectfifo<memref<1056xf32>>
    aie.objectfifo @joins_0(%logical_mem_54, {%logical_core_7}, 1 : i32) : !aie.objectfifo<memref<4224xf32>>
    aie.objectfifo.link [@of0, @of1, @of2, @of3] -> [@joins_0]([0, 1056, 2112, 3168] [])
    aie.objectfifo @of4(%logical_core_3, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<1056xf32>>
    aie.objectfifo @of5(%logical_core_4, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<1056xf32>>
    aie.objectfifo @of6(%logical_core_5, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<1056xf32>>
    aie.objectfifo @of7(%logical_core_6, {%logical_mem_55}, 2 : i32) : !aie.objectfifo<memref<1056xf32>>
    aie.objectfifo @joins_1(%logical_mem_55, {%logical_core_7}, 1 : i32) : !aie.objectfifo<memref<4224xf32>>
    aie.objectfifo.link [@of4, @of5, @of6, @of7] -> [@joins_1]([0, 1056, 2112, 3168] [])
    %buffer_0 = aie.buffer(%logical_core) {sym_name = "buffer_0"} : memref<5xi32>
    %0 = aie.buffer(%logical_core) : memref<256xbf16>
    %1 = aie.buffer(%logical_core) : memref<256xbf16>
    %2 = aie.buffer(%logical_core) : memref<3xi32> = dense<0>
    %3 = aie.buffer(%logical_core) : memref<1536xbf16>
    %4 = aie.buffer(%logical_core) : memref<8x128xbf16>
    %5 = aie.buffer(%logical_core) : memref<32xf32>
    %buffer_1 = aie.buffer(%logical_core_0) {sym_name = "buffer_1"} : memref<5xi32>
    %6 = aie.buffer(%logical_core_0) : memref<256xbf16>
    %7 = aie.buffer(%logical_core_0) : memref<256xbf16>
    %8 = aie.buffer(%logical_core_0) : memref<3xi32> = dense<0>
    %9 = aie.buffer(%logical_core_0) : memref<1536xbf16>
    %10 = aie.buffer(%logical_core_0) : memref<8x128xbf16>
    %11 = aie.buffer(%logical_core_0) : memref<32xf32>
    %buffer_2 = aie.buffer(%logical_core_1) {sym_name = "buffer_2"} : memref<5xi32>
    %12 = aie.buffer(%logical_core_1) : memref<256xbf16>
    %13 = aie.buffer(%logical_core_1) : memref<256xbf16>
    %14 = aie.buffer(%logical_core_1) : memref<3xi32> = dense<0>
    %15 = aie.buffer(%logical_core_1) : memref<1536xbf16>
    %16 = aie.buffer(%logical_core_1) : memref<8x128xbf16>
    %17 = aie.buffer(%logical_core_1) : memref<32xf32>
    %buffer_3 = aie.buffer(%logical_core_2) {sym_name = "buffer_3"} : memref<5xi32>
    %18 = aie.buffer(%logical_core_2) : memref<256xbf16>
    %19 = aie.buffer(%logical_core_2) : memref<256xbf16>
    %20 = aie.buffer(%logical_core_2) : memref<3xi32> = dense<0>
    %21 = aie.buffer(%logical_core_2) : memref<1536xbf16>
    %22 = aie.buffer(%logical_core_2) : memref<8x128xbf16>
    %23 = aie.buffer(%logical_core_2) : memref<32xf32>
    %buffer_4 = aie.buffer(%logical_core_3) {sym_name = "buffer_4"} : memref<5xi32>
    %24 = aie.buffer(%logical_core_3) : memref<256xbf16>
    %25 = aie.buffer(%logical_core_3) : memref<256xbf16>
    %26 = aie.buffer(%logical_core_3) : memref<3xi32> = dense<0>
    %27 = aie.buffer(%logical_core_3) : memref<1536xbf16>
    %28 = aie.buffer(%logical_core_3) : memref<8x128xbf16>
    %29 = aie.buffer(%logical_core_3) : memref<32xf32>
    %buffer_5 = aie.buffer(%logical_core_4) {sym_name = "buffer_5"} : memref<5xi32>
    %30 = aie.buffer(%logical_core_4) : memref<256xbf16>
    %31 = aie.buffer(%logical_core_4) : memref<256xbf16>
    %32 = aie.buffer(%logical_core_4) : memref<3xi32> = dense<0>
    %33 = aie.buffer(%logical_core_4) : memref<1536xbf16>
    %34 = aie.buffer(%logical_core_4) : memref<8x128xbf16>
    %35 = aie.buffer(%logical_core_4) : memref<32xf32>
    %buffer_6 = aie.buffer(%logical_core_5) {sym_name = "buffer_6"} : memref<5xi32>
    %36 = aie.buffer(%logical_core_5) : memref<256xbf16>
    %37 = aie.buffer(%logical_core_5) : memref<256xbf16>
    %38 = aie.buffer(%logical_core_5) : memref<3xi32> = dense<0>
    %39 = aie.buffer(%logical_core_5) : memref<1536xbf16>
    %40 = aie.buffer(%logical_core_5) : memref<8x128xbf16>
    %41 = aie.buffer(%logical_core_5) : memref<32xf32>
    %buffer_7 = aie.buffer(%logical_core_6) {sym_name = "buffer_7"} : memref<5xi32>
    %42 = aie.buffer(%logical_core_6) : memref<256xbf16>
    %43 = aie.buffer(%logical_core_6) : memref<256xbf16>
    %44 = aie.buffer(%logical_core_6) : memref<3xi32> = dense<0>
    %45 = aie.buffer(%logical_core_6) : memref<1536xbf16>
    %46 = aie.buffer(%logical_core_6) : memref<8x128xbf16>
    %47 = aie.buffer(%logical_core_6) : memref<32xf32>
    %buffer_8 = aie.buffer(%logical_core_7) {sym_name = "buffer_8"} : memref<1xi32>
    %48 = aie.buffer(%logical_core_7) : memref<1056xf32>
    %49 = aie.buffer(%logical_core_7) : memref<32xf32>
    %50 = aie.buffer(%logical_core_7) : memref<3xi32> = dense<0>
    %vectors_0 = aie.buffer(%logical_core_8) {sym_name = "vectors_0"} : memref<1536xbf16>
    %buffer_9 = aie.buffer(%logical_core_8) {sym_name = "buffer_9"} : memref<2xi32>
    %vectors_1 = aie.buffer(%logical_core_9) {sym_name = "vectors_1"} : memref<1536xbf16>
    %buffer_10 = aie.buffer(%logical_core_9) {sym_name = "buffer_10"} : memref<2xi32>
    %vectors_2 = aie.buffer(%logical_core_10) {sym_name = "vectors_2"} : memref<1536xbf16>
    %buffer_11 = aie.buffer(%logical_core_10) {sym_name = "buffer_11"} : memref<2xi32>
    %vectors_3 = aie.buffer(%logical_core_11) {sym_name = "vectors_3"} : memref<1536xbf16>
    %buffer_12 = aie.buffer(%logical_core_11) {sym_name = "buffer_12"} : memref<2xi32>
    %vectors_4 = aie.buffer(%logical_core_12) {sym_name = "vectors_4"} : memref<1536xbf16>
    %buffer_13 = aie.buffer(%logical_core_12) {sym_name = "buffer_13"} : memref<2xi32>
    %vectors_5 = aie.buffer(%logical_core_13) {sym_name = "vectors_5"} : memref<1536xbf16>
    %buffer_14 = aie.buffer(%logical_core_13) {sym_name = "buffer_14"} : memref<2xi32>
    %vectors_6 = aie.buffer(%logical_core_14) {sym_name = "vectors_6"} : memref<1536xbf16>
    %buffer_15 = aie.buffer(%logical_core_14) {sym_name = "buffer_15"} : memref<2xi32>
    %vectors_7 = aie.buffer(%logical_core_15) {sym_name = "vectors_7"} : memref<1536xbf16>
    %buffer_16 = aie.buffer(%logical_core_15) {sym_name = "buffer_16"} : memref<2xi32>
    %51 = aie.lock(%logical_core)
    aie.core(%logical_core) {
      %a0 = aie.objectfifo.acquire @memKV_0(Consume, 1) : memref<32x256xbf16>
      aie.objectfifo.release @memKV_0(Consume, 1)
      %a1 = aie.objectfifo.acquire @of0(Produce, 1) : memref<1056xf32>
      aie.objectfifo.release @of0(Produce, 1)
      aie.end
    }
    %53 = aie.lock(%logical_core_0)
    aie.core(%logical_core_0) {
      %a0 = aie.objectfifo.acquire @memKV_1(Consume, 1) : memref<32x256xbf16>
      aie.objectfifo.release @memKV_1(Consume, 1)
      %a1 = aie.objectfifo.acquire @of1(Produce, 1) : memref<1056xf32>
      aie.objectfifo.release @of1(Produce, 1)
      aie.end
    }
    %55 = aie.lock(%logical_core_1)
    aie.core(%logical_core_1) {
      %a0 = aie.objectfifo.acquire @memKV_2(Consume, 1) : memref<32x256xbf16>
      aie.objectfifo.release @memKV_2(Consume, 1)
      %a1 = aie.objectfifo.acquire @of2(Produce, 1) : memref<1056xf32>
      aie.objectfifo.release @of2(Produce, 1)
      aie.end
    }
    %57 = aie.lock(%logical_core_2)
    aie.core(%logical_core_2) {
      %a0 = aie.objectfifo.acquire @memKV_3(Consume, 1) : memref<32x256xbf16>
      aie.objectfifo.release @memKV_3(Consume, 1)
      %a1 = aie.objectfifo.acquire @of3(Produce, 1) : memref<1056xf32>
      aie.objectfifo.release @of3(Produce, 1)
      aie.end
    }
    %59 = aie.lock(%logical_core_3)
    aie.core(%logical_core_3) {
      %a0 = aie.objectfifo.acquire @memKV_4(Consume, 1) : memref<32x256xbf16>
      aie.objectfifo.release @memKV_4(Consume, 1)
      %a1 = aie.objectfifo.acquire @of4(Produce, 1) : memref<1056xf32>
      aie.objectfifo.release @of4(Produce, 1)
      aie.end
    }
    %61 = aie.lock(%logical_core_4)
    aie.core(%logical_core_4) {
      %a0 = aie.objectfifo.acquire @memKV_5(Consume, 1) : memref<32x256xbf16>
      aie.objectfifo.release @memKV_5(Consume, 1)
      %a1 = aie.objectfifo.acquire @of5(Produce, 1) : memref<1056xf32>
      aie.objectfifo.release @of5(Produce, 1)
      aie.end
    }
    %63 = aie.lock(%logical_core_5)
    aie.core(%logical_core_5) {
      %a0 = aie.objectfifo.acquire @memKV_6(Consume, 1) : memref<32x256xbf16>
      aie.objectfifo.release @memKV_6(Consume, 1)
      %a1 = aie.objectfifo.acquire @of6(Produce, 1) : memref<1056xf32>
      aie.objectfifo.release @of6(Produce, 1)
      aie.end
    }
    %65 = aie.lock(%logical_core_6)
    aie.core(%logical_core_6) {
      %a0 = aie.objectfifo.acquire @memKV_7(Consume, 1) : memref<32x256xbf16>
      aie.objectfifo.release @memKV_7(Consume, 1)
      %a1 = aie.objectfifo.acquire @of7(Produce, 1) : memref<1056xf32>
      aie.objectfifo.release @of7(Produce, 1)
      aie.end
    }
    %67 = aie.lock(%logical_core_7)
    aie.core(%logical_core_7) {
      %a0 = aie.objectfifo.acquire @joins_0(Consume, 1) : memref<4224xf32>
      aie.objectfifo.release @joins_0(Consume, 1)
      %a1 = aie.objectfifo.acquire @joins_1(Consume, 1) : memref<4224xf32>
      aie.objectfifo.release @joins_1(Consume, 1)
      %a2 = aie.objectfifo.acquire @outO(Produce, 1) : memref<8x128xbf16>
      aie.objectfifo.release @outO(Produce, 1)
      aie.end
    }
    %69 = aie.lock(%logical_core_8)
    aie.core(%logical_core_8) {
      %a0 = aie.objectfifo.acquire @made_1(Consume, 1) : memref<2x1536xbf16>
      aie.objectfifo.release @made_1(Consume, 1)
      %a1 = aie.objectfifo.acquire @made_9(Produce, 1) : memref<2xbf16>
      aie.objectfifo.release @made_9(Produce, 1)
      aie.end
    }
    %71 = aie.lock(%logical_core_9)
    aie.core(%logical_core_9) {
      %a0 = aie.objectfifo.acquire @made_10(Produce, 1) : memref<2xbf16>
      aie.objectfifo.release @made_10(Produce, 1)
      %a1 = aie.objectfifo.acquire @made_2(Consume, 1) : memref<2x1536xbf16>
      aie.objectfifo.release @made_2(Consume, 1)
      aie.end
    }
    %73 = aie.lock(%logical_core_10)
    aie.core(%logical_core_10) {
      %a0 = aie.objectfifo.acquire @made_11(Produce, 1) : memref<2xbf16>
      aie.objectfifo.release @made_11(Produce, 1)
      %a1 = aie.objectfifo.acquire @made_3(Consume, 1) : memref<2x1536xbf16>
      aie.objectfifo.release @made_3(Consume, 1)
      aie.end
    }
    %75 = aie.lock(%logical_core_11)
    aie.core(%logical_core_11) {
      %a0 = aie.objectfifo.acquire @made_12(Produce, 1) : memref<2xbf16>
      aie.objectfifo.release @made_12(Produce, 1)
      %a1 = aie.objectfifo.acquire @made_4(Consume, 1) : memref<2x1536xbf16>
      aie.objectfifo.release @made_4(Consume, 1)
      aie.end
    }
    %77 = aie.lock(%logical_core_12)
    aie.core(%logical_core_12) {
      %a0 = aie.objectfifo.acquire @made_13(Produce, 1) : memref<2xbf16>
      aie.objectfifo.release @made_13(Produce, 1)
      %a1 = aie.objectfifo.acquire @made_5(Consume, 1) : memref<2x1536xbf16>
      aie.objectfifo.release @made_5(Consume, 1)
      aie.end
    }
    %79 = aie.lock(%logical_core_13)
    aie.core(%logical_core_13) {
      %a0 = aie.objectfifo.acquire @made_14(Produce, 1) : memref<2xbf16>
      aie.objectfifo.release @made_14(Produce, 1)
      %a1 = aie.objectfifo.acquire @made_6(Consume, 1) : memref<2x1536xbf16>
      aie.objectfifo.release @made_6(Consume, 1)
      aie.end
    }
    %81 = aie.lock(%logical_core_14)
    aie.core(%logical_core_14) {
      %a0 = aie.objectfifo.acquire @made_15(Produce, 1) : memref<2xbf16>
      aie.objectfifo.release @made_15(Produce, 1)
      %a1 = aie.objectfifo.acquire @made_7(Consume, 1) : memref<2x1536xbf16>
      aie.objectfifo.release @made_7(Consume, 1)
      aie.end
    }
    %83 = aie.lock(%logical_core_15)
    aie.core(%logical_core_15) {
      %a0 = aie.objectfifo.acquire @made_16(Produce, 1) : memref<2xbf16>
      aie.objectfifo.release @made_16(Produce, 1)
      %a1 = aie.objectfifo.acquire @made_8(Consume, 1) : memref<2x1536xbf16>
      aie.objectfifo.release @made_8(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @d0(%arg0: memref<1536xbf16>, %arg1: memref<8388608xbf16>, %arg2: memref<1536xbf16>) {
      %c32_i32 = arith.constant 32 : i32
      %c32_i32_60 = arith.constant 32 : i32
      %c1_i32 = arith.constant 1 : i32
      %85 = aiex.dma_configure_task_for @inKV_0 {
        aie.dma_bd(%arg0 : memref<1536xbf16> offset = 0 len = 1536 sizes = [1, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @inKV_0 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 6656 sizes = [1, 1, 1, 6656] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @inKV_0 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 262144 sizes = [4, 32, 32, 256] strides = [256, 262144, 1024, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @inKV_1 {
        aie.dma_bd(%arg0 : memref<1536xbf16> offset = 0 len = 1536 sizes = [1, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @inKV_1 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 6656 sizes = [1, 1, 1, 6656] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @inKV_1 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 32768 len = 262144 sizes = [4, 32, 32, 256] strides = [256, 262144, 1024, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @inKV_2 {
        aie.dma_bd(%arg0 : memref<1536xbf16> offset = 0 len = 1536 sizes = [1, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @inKV_2 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 6656 sizes = [1, 1, 1, 6656] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @inKV_2 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 65536 len = 262144 sizes = [4, 32, 32, 256] strides = [256, 262144, 1024, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @inKV_3 {
        aie.dma_bd(%arg0 : memref<1536xbf16> offset = 0 len = 1536 sizes = [1, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @inKV_3 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 6656 sizes = [1, 1, 1, 6656] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @inKV_3 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 98304 len = 262144 sizes = [4, 32, 32, 256] strides = [256, 262144, 1024, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @inKV_4 {
        aie.dma_bd(%arg0 : memref<1536xbf16> offset = 0 len = 1536 sizes = [1, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @inKV_4 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 6656 sizes = [1, 1, 1, 6656] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @inKV_4 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 131072 len = 262144 sizes = [4, 32, 32, 256] strides = [256, 262144, 1024, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @inKV_5 {
        aie.dma_bd(%arg0 : memref<1536xbf16> offset = 0 len = 1536 sizes = [1, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @inKV_5 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 6656 sizes = [1, 1, 1, 6656] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @inKV_5 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 163840 len = 262144 sizes = [4, 32, 32, 256] strides = [256, 262144, 1024, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @inKV_6 {
        aie.dma_bd(%arg0 : memref<1536xbf16> offset = 0 len = 1536 sizes = [1, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @inKV_6 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 6656 sizes = [1, 1, 1, 6656] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @inKV_6 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 196608 len = 262144 sizes = [4, 32, 32, 256] strides = [256, 262144, 1024, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @inKV_7 {
        aie.dma_bd(%arg0 : memref<1536xbf16> offset = 0 len = 1536 sizes = [1, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @inKV_7 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 0 len = 6656 sizes = [1, 1, 1, 6656] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @inKV_7 {
        aie.dma_bd(%arg1 : memref<8388608xbf16> offset = 229376 len = 262144 sizes = [4, 32, 32, 256] strides = [256, 262144, 1024, 1])
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @made_0 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 0 len = 512 sizes = [2, 2, 2, 128] strides = [128, 0, 128, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @made_0 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 384 len = 512 sizes = [2, 2, 2, 128] strides = [128, 0, 128, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @made_0 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 768 len = 512 sizes = [2, 2, 2, 128] strides = [128, 0, 128, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @made_0 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 1152 len = 512 sizes = [2, 2, 2, 128] strides = [128, 0, 128, 1])
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%112)
      aiex.dma_await_task(%109)
      aiex.dma_await_task(%110)
      aiex.dma_await_task(%111)
      aiex.dma_await_task(%112)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%90)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%93)
      aiex.dma_free_task(%94)
      aiex.dma_free_task(%95)
      aiex.dma_free_task(%96)
      aiex.dma_free_task(%97)
      aiex.dma_free_task(%98)
      aiex.dma_free_task(%99)
      aiex.dma_free_task(%100)
      aiex.dma_free_task(%101)
      aiex.dma_free_task(%102)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%106)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%110)
      aiex.dma_free_task(%111)
      aiex.dma_free_task(%112)
    }
    aie.runtime_sequence @d1(%arg0: memref<2359296xbf16>, %arg1: memref<1536xbf16>, %arg2: memref<1536xbf16>) {
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_57 = arith.constant 1 : i32
      %c1_i32_59 = arith.constant 1 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c1_i32_63 = arith.constant 1 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c1_i32_67 = arith.constant 1 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %c1_i32_70 = arith.constant 1 : i32
      %c1_i32_71 = arith.constant 1 : i32
      %c1_i32_72 = arith.constant 1 : i32
      %c1_i32_73 = arith.constant 1 : i32
      %c1_i32_74 = arith.constant 1 : i32
      %c1_i32_75 = arith.constant 1 : i32
      %c1_i32_76 = arith.constant 1 : i32
      %c1_i32_77 = arith.constant 1 : i32
      %85 = aiex.dma_configure_task_for @made_9 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 0 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @made_10 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 192 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @made_11 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 384 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @made_12 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 576 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @made_13 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 768 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @made_14 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 960 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @made_15 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 1152 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @made_16 {
        aie.dma_bd(%arg2 : memref<1536xbf16> offset = 1344 len = 192 sizes = [1, 1, 1, 192] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @made_1 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @made_1 {
        aie.dma_bd(%arg0 : memref<2359296xbf16> offset = 0 len = 294912 sizes = [1, 1, 192, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @made_2 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @made_2 {
        aie.dma_bd(%arg0 : memref<2359296xbf16> offset = 294912 len = 294912 sizes = [1, 1, 192, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @made_3 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @made_3 {
        aie.dma_bd(%arg0 : memref<2359296xbf16> offset = 589824 len = 294912 sizes = [1, 1, 192, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @made_4 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @made_4 {
        aie.dma_bd(%arg0 : memref<2359296xbf16> offset = 884736 len = 294912 sizes = [1, 1, 192, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @made_5 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @made_5 {
        aie.dma_bd(%arg0 : memref<2359296xbf16> offset = 1179648 len = 294912 sizes = [1, 1, 192, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @made_6 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @made_6 {
        aie.dma_bd(%arg0 : memref<2359296xbf16> offset = 1474560 len = 294912 sizes = [1, 1, 192, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @made_7 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @made_7 {
        aie.dma_bd(%arg0 : memref<2359296xbf16> offset = 1769472 len = 294912 sizes = [1, 1, 192, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @made_8 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @made_8 {
        aie.dma_bd(%arg0 : memref<2359296xbf16> offset = 2064384 len = 294912 sizes = [1, 1, 192, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%108)
      aiex.dma_await_task(%85)
      aiex.dma_await_task(%86)
      aiex.dma_await_task(%87)
      aiex.dma_await_task(%88)
      aiex.dma_await_task(%89)
      aiex.dma_await_task(%90)
      aiex.dma_await_task(%91)
      aiex.dma_await_task(%92)
    }
    aie.runtime_sequence @d2(%arg0: memref<786432xbf16>, %arg1: memref<1536xbf16>, %arg2: memref<512xbf16>) {
      %c32_i32 = arith.constant 32 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1_i32_57 = arith.constant 1 : i32
      %c1_i32_59 = arith.constant 1 : i32
      %c32_i32_60 = arith.constant 32 : i32
      %c1_i32_61 = arith.constant 1 : i32
      %c1_i32_63 = arith.constant 1 : i32
      %c1_i32_65 = arith.constant 1 : i32
      %c1_i32_67 = arith.constant 1 : i32
      %c1_i32_69 = arith.constant 1 : i32
      %c1_i32_70 = arith.constant 1 : i32
      %c1_i32_71 = arith.constant 1 : i32
      %c1_i32_72 = arith.constant 1 : i32
      %c1_i32_73 = arith.constant 1 : i32
      %c1_i32_74 = arith.constant 1 : i32
      %c1_i32_75 = arith.constant 1 : i32
      %c1_i32_76 = arith.constant 1 : i32
      %c1_i32_77 = arith.constant 1 : i32
      %85 = aiex.dma_configure_task_for @made_9 {
        aie.dma_bd(%arg2 : memref<512xbf16> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @made_10 {
        aie.dma_bd(%arg2 : memref<512xbf16> offset = 64 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @made_11 {
        aie.dma_bd(%arg2 : memref<512xbf16> offset = 128 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @made_12 {
        aie.dma_bd(%arg2 : memref<512xbf16> offset = 192 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @made_13 {
        aie.dma_bd(%arg2 : memref<512xbf16> offset = 256 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @made_14 {
        aie.dma_bd(%arg2 : memref<512xbf16> offset = 320 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @made_15 {
        aie.dma_bd(%arg2 : memref<512xbf16> offset = 384 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @made_16 {
        aie.dma_bd(%arg2 : memref<512xbf16> offset = 448 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @made_1 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @made_1 {
        aie.dma_bd(%arg0 : memref<786432xbf16> offset = 0 len = 98304 sizes = [1, 1, 64, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @made_2 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%95)
      %96 = aiex.dma_configure_task_for @made_2 {
        aie.dma_bd(%arg0 : memref<786432xbf16> offset = 98304 len = 98304 sizes = [1, 1, 64, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @made_3 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @made_3 {
        aie.dma_bd(%arg0 : memref<786432xbf16> offset = 196608 len = 98304 sizes = [1, 1, 64, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @made_4 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @made_4 {
        aie.dma_bd(%arg0 : memref<786432xbf16> offset = 294912 len = 98304 sizes = [1, 1, 64, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @made_5 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @made_5 {
        aie.dma_bd(%arg0 : memref<786432xbf16> offset = 393216 len = 98304 sizes = [1, 1, 64, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @made_6 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @made_6 {
        aie.dma_bd(%arg0 : memref<786432xbf16> offset = 491520 len = 98304 sizes = [1, 1, 64, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @made_7 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @made_7 {
        aie.dma_bd(%arg0 : memref<786432xbf16> offset = 589824 len = 98304 sizes = [1, 1, 64, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @made_8 {
        aie.dma_bd(%arg1 : memref<1536xbf16> offset = 0 len = 1536 sizes = [2, 1, 1, 1536] strides = [0, 0, 0, 1])
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @made_8 {
        aie.dma_bd(%arg0 : memref<786432xbf16> offset = 688128 len = 98304 sizes = [1, 1, 64, 1536] strides = [0, 0, 1536, 1])
        aie.end
      }
      aiex.dma_start_task(%108)
      aiex.dma_await_task(%85)
      aiex.dma_await_task(%86)
      aiex.dma_await_task(%87)
      aiex.dma_await_task(%88)
      aiex.dma_await_task(%89)
      aiex.dma_await_task(%90)
      aiex.dma_await_task(%91)
      aiex.dma_await_task(%92)
    }
    aie.runtime_sequence @d3(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
      %85 = aiex.dma_configure_task_for @made_17 {
        aie.dma_bd(%arg0 : memref<512xbf16> offset = 0 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @fifo_out_0 {
        aie.dma_bd(%arg1 : memref<512xbf16> offset = 0 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @made_18 {
        aie.dma_bd(%arg0 : memref<512xbf16> offset = 256 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @fifo_out_1 {
        aie.dma_bd(%arg1 : memref<512xbf16> offset = 256 len = 256 sizes = [1, 1, 1, 256] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%88)
      aiex.dma_await_task(%86)
      aiex.dma_await_task(%88)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%87)
      aiex.dma_free_task(%88)
    }
  }
}
