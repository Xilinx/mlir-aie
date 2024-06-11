//===- buffers_xclbin.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: chess
// RUN: %PYTHON aiecc.py --xchesscc --no-link -nv --aie-generate-cdo --aie-generate-npu --no-compile-host --xclbin-name=aie.xclbin --npu-insts-name=insts.txt %s 
// RUN: FileCheck %s --input-file=buffers_xclbin.mlir.prj/kernels.json

// CHECK: {
// CHECK:   "ps-kernels": {
// CHECK:     "kernels": [
// CHECK:       {
// CHECK:         "name": "MLIR_AIE",
// CHECK:         "type": "dpu",
// CHECK:         "extended-data": {
// CHECK:           "subtype": "DPU",
// CHECK:           "functional": "0",
// CHECK:           "dpu_kernel_id": "0x901"
// CHECK:         },
// CHECK:         "arguments": [
// CHECK:           {
// CHECK:             "name": "opcode",
// CHECK:             "address-qualifier": "SCALAR",
// CHECK:             "type": "uint64_t",
// CHECK:             "offset": "0x00"
// CHECK:           },
// CHECK:           {
// CHECK:             "name": "instr",
// CHECK:             "memory-connection": "SRAM",
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "type": "char *",
// CHECK:             "offset": "0x8"
// CHECK:           },
// CHECK:           {
// CHECK:             "name": "ninstr",
// CHECK:             "address-qualifier": "SCALAR",
// CHECK:             "type": "uint32_t",
// CHECK:             "offset": "0x10"
// CHECK:           },
// CHECK:           {
// CHECK:             "name": "bo0",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "type": "void*",
// CHECK:             "offset": "0x14"
// CHECK:           },
// CHECK:           {
// CHECK:             "name": "bo1",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "type": "void*",
// CHECK:             "offset": "0x1c"
// CHECK:           },
// CHECK:           {
// CHECK:             "name": "bo2",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "type": "void*",
// CHECK:             "offset": "0x24"
// CHECK:           },
// CHECK:           {
// CHECK:             "name": "bo3",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "type": "void*",
// CHECK:             "offset": "0x2c"
// CHECK:           },
// CHECK:           {
// CHECK:             "name": "bo4",
// CHECK:             "memory-connection": "HOST",
// CHECK:             "address-qualifier": "GLOBAL",
// CHECK:             "type": "void*",
// CHECK:             "offset": "0x34"
// CHECK:           }
// CHECK:         ],
// CHECK:         "instances": [
// CHECK:           {
// CHECK:             "name": "MLIRAIE"
// CHECK:           }
// CHECK:         ]
// CHECK:       }
// CHECK:     ]
// CHECK:   }
// CHECK: }

module {
  aie.device(npu1_4col) {
    %02 = aie.tile(0, 2)
    %12 = aie.tile(1, 2)
    %22 = aie.tile(2, 2)
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