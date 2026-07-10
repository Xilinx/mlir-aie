//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 16-host-buffer end-to-end test. Exercises host buffer arguments beyond the
// first 5 (XRT group_id 3..7), which the NPU firmware does not pre-translate
// into the AIE address space -- so their DDR address patches must fold in the
// translation offset (see AIETargetNPU.cpp). Eight core-free shim->memtile->
// shim passthrough lanes (2 per column across 4 columns) pass 8 independent
// in/out pairs = 16 host buffers, all in one runtime sequence.

module {
  aie.device(NPUDEVICE) {
    %s0 = aie.tile(0, 0)
    %m0 = aie.tile(0, 1)
    %s1 = aie.tile(1, 0)
    %m1 = aie.tile(1, 1)
    %s2 = aie.tile(2, 0)
    %m2 = aie.tile(2, 1)
    %s3 = aie.tile(3, 0)
    %m3 = aie.tile(3, 1)
    aie.objectfifo @in0 (%s0, {%m0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out0(%m0, {%s0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in0] -> [@out0] ([] [])
    aie.objectfifo @in1 (%s0, {%m0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out1(%m0, {%s0}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in1] -> [@out1] ([] [])
    aie.objectfifo @in2 (%s1, {%m1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out2(%m1, {%s1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in2] -> [@out2] ([] [])
    aie.objectfifo @in3 (%s1, {%m1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out3(%m1, {%s1}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in3] -> [@out3] ([] [])
    aie.objectfifo @in4 (%s2, {%m2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out4(%m2, {%s2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in4] -> [@out4] ([] [])
    aie.objectfifo @in5 (%s2, {%m2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out5(%m2, {%s2}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in5] -> [@out5] ([] [])
    aie.objectfifo @in6 (%s3, {%m3}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out6(%m3, {%s3}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in6] -> [@out6] ([] [])
    aie.objectfifo @in7 (%s3, {%m3}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out7(%m3, {%s3}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@in7] -> [@out7] ([] [])
    aie.runtime_sequence(%in0: memref<64xi32>, %out0: memref<64xi32>, %in1: memref<64xi32>, %out1: memref<64xi32>, %in2: memref<64xi32>, %out2: memref<64xi32>, %in3: memref<64xi32>, %out3: memref<64xi32>, %in4: memref<64xi32>, %out4: memref<64xi32>, %in5: memref<64xi32>, %out5: memref<64xi32>, %in6: memref<64xi32>, %out6: memref<64xi32>, %in7: memref<64xi32>, %out7: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%in0[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 0 : i64, metadata = @in0} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out0[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 1 : i64, metadata = @out0, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in1[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 2 : i64, metadata = @in1} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out1[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 3 : i64, metadata = @out1, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in2[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 4 : i64, metadata = @in2} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out2[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 5 : i64, metadata = @out2, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in3[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 6 : i64, metadata = @in3} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out3[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 7 : i64, metadata = @out3, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in4[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 8 : i64, metadata = @in4} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out4[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 9 : i64, metadata = @out4, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in5[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 10 : i64, metadata = @in5} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out5[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 11 : i64, metadata = @out5, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in6[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 12 : i64, metadata = @in6} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out6[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 13 : i64, metadata = @out6, issue_token = true} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%in7[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 14 : i64, metadata = @in7} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%out7[0,0,0,0][1,1,1,64][0,0,0,1]) {id = 15 : i64, metadata = @out7, issue_token = true} : memref<64xi32>
      aiex.npu.dma_wait { symbol = @out0 }
      aiex.npu.dma_wait { symbol = @out1 }
      aiex.npu.dma_wait { symbol = @out2 }
      aiex.npu.dma_wait { symbol = @out3 }
      aiex.npu.dma_wait { symbol = @out4 }
      aiex.npu.dma_wait { symbol = @out5 }
      aiex.npu.dma_wait { symbol = @out6 }
      aiex.npu.dma_wait { symbol = @out7 }
    }
  }
}
