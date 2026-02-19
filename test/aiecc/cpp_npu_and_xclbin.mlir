//===- cpp_npu_and_xclbin.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test NPU instruction and xclbin generation

// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-npu-insts --aie-generate-xclbin --verbose %s | FileCheck %s

// CHECK: Successfully parsed input file
// CHECK: Running resource allocation pipeline in-memory
// CHECK: Resource allocation pipeline completed successfully
// CHECK: Running routing pipeline in-memory
// CHECK: Routing pipeline completed successfully
// CHECK: Compiling core (0, 2)
// CHECK: Generating NPU instructions for device
// CHECK: Generating CDO artifacts for device
// CHECK: Generated PDI
// CHECK: Generated xclbin
// CHECK: Compilation completed successfully

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @data(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index

      %subview = aie.objectfifo.acquire @data(Consume, 1) : !aie.objectfifosubview<memref<64xi32>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>

      scf.for %i = %c0 to %c64 step %c1 {
        %val = memref.load %elem[%i] : memref<64xi32>
        memref.store %val, %elem[%i] : memref<64xi32>
      }

      aie.objectfifo.release @data(Consume, 1)
      aie.end
    }

    aie.runtime_sequence(%buf : memref<64xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%buf[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @data}
    }
  }
}
