//===- buffers_xclbin.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// RUN: aie2xclbin -v --host-target=aarch64-linux-gnu --peano=%PEANO_INSTALL_DIR %s --tmpdir=%T/buffers_xclbin.mlir.prj --xclbin-name=test.xclbin 
// RUN: FileCheck %s --input-file=%T/buffers_xclbin.mlir.prj/kernels.json

// CHECK: {
// CHECK:   "ps-kernels": {
// CHECK:     "kernels": [
// CHECK:       {
// CHECK:         "arguments": [
// CHECK:           {
// CHECK:             "address-qualifier": "SCALAR",
// CHECK:             "name": "opcode",
// CHECK:             "offset": "0x00",
// CHECK:             "type": "uint64_t"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "SRAM",
// CHECK:             "name": "instr",
// CHECK:             "offset": "0x08",
// CHECK:             "type": "char *"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "SCALAR",
// CHECK:             "name": "ninstr",
// CHECK:             "offset": "0x10",
// CHECK:             "type": "uint32_t"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo0",
// CHECK:             "offset": "0x14",
// CHECK:             "type": "void*"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo1",
// CHECK:             "offset": "0x1c",
// CHECK:             "type": "void*"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo2",
// CHECK:             "offset": "0x24",
// CHECK:             "type": "void*"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo3",
// CHECK:             "offset": "0x2c",
// CHECK:             "type": "void*"
// CHECK:           },
// CHECK:           {
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "name": "bo4",
// CHECK:             "offset": "0x34",
// CHECK:             "type": "void*"
// CHECK:           }
// CHECK:         ],
// CHECK:         "extended-data": {
// CHECK:           "dpu_kernel_id": "0x901",
// CHECK:           "functional": "0",
// CHECK:           "subtype": "DPU"
// CHECK:         },
// CHECK:         "instances": [
// CHECK:           {
// CHECK:             "name": "MLIRAIEV1"
// CHECK:           }
// CHECK:         ],
// CHECK:         "name": "MLIR_AIE",
// CHECK:         "type": "dpu"
// CHECK:       }
// CHECK:     ]
// CHECK:   }
// CHECK: }

module {
  aie.device(npu1_4col) {
    memref.global "public" @in0 : memref<1024xi32>
    memref.global "public" @out0 : memref<1024xi32>
    memref.global "public" @in1 : memref<1024xi32>
    memref.global "public" @out1 : memref<1024xi32>
    memref.global "public" @in2 : memref<1024xi32>
    memref.global "public" @out2 : memref<1024xi32>
    %02 = aie.tile(0, 2)
    %12 = aie.tile(1, 2)
    %22 = aie.tile(2, 2)

    aie.core(%12) {
      aie.end
    }
    aie.shim_dma_allocation @in0(MM2S, 0, 0)
    aie.shim_dma_allocation @out0(S2MM, 0, 0)
    aie.shim_dma_allocation @in1(MM2S, 1, 0)
    aie.shim_dma_allocation @out1(S2MM, 1, 0)
    aie.shim_dma_allocation @in2(MM2S, 2, 0)
    aie.shim_dma_allocation @out2(S2MM, 2, 0)
    
    func.func @sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>, %arg4: memref<1024xi32>, %arg5: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 0 : i64, metadata = @in0} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 1 : i64, metadata = @out0} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 2 : i64, metadata = @in1} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg3[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 3 : i64, metadata = @out1} : memref<1024xi32>
      aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.dma_memcpy_nd(0, 0, %arg4[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 2 : i64, metadata = @in2} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg5[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 3 : i64, metadata = @out2} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 2 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}