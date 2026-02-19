//===- cpp_aie2p_target.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test AIE2P (Strix) target compilation

// RUN: aiecc --no-xchesscc --no-xbridge --verbose %s | FileCheck %s

// CHECK: Successfully parsed input file
// CHECK: Found 1 AIE device
// CHECK: Detected AIE target: AIE2p
// CHECK: Running resource allocation pipeline in-memory
// CHECK: Resource allocation pipeline completed successfully
// CHECK: Running routing pipeline in-memory
// CHECK: Compiling core (0, 2)
// CHECK: Compilation completed successfully

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index

      %subview_in = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>

      %subview_out = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>

      scf.for %i = %c0 to %c32 step %c1 {
        %val = memref.load %elem_in[%i] : memref<32xi32>
        memref.store %val, %elem_out[%i] : memref<32xi32>
      }

      aie.objectfifo.release @in(Consume, 1)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%arg_in : memref<32xi32>, %arg_out : memref<32xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c32 = arith.constant 32 : i64
      aiex.npu.dma_memcpy_nd(%arg_out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c32][%c0,%c0,%c0,%c1]) {metadata = @out, id = 1 : i64} : memref<32xi32>
      aiex.npu.dma_memcpy_nd(%arg_in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c32][%c0,%c0,%c0,%c1]) {metadata = @in, id = 0 : i64, issue_token = true} : memref<32xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}
