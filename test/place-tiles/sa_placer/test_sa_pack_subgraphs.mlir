//===- test_sa_pack_subgraphs.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Four independent weight-read subgraphs in one device, as a pack of
// concurrent runs places them: the SA finds a legal placement that routes,
// and their weight streams take four shim columns.

// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{placer=sa_placer sa-seed=1})' %s -o /dev/null
// RUN: aie-opt --aie-place-tiles='placer=sa_placer sa-seed=1' %s 2>&1 | FileCheck %s

// CHECK-NOT: warning
// CHECK-COUNT-4: aie.tile({{[0-7]}}, 0)

module @pack {
  aie.device(npu2) {
    %in0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %out0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %core0 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @w0(%in0, {%core0}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @y0(%core0, {%out0}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.core(%core0) {
      %w = aie.objectfifo.acquire @w0(Consume, 1) : memref<1024xbf16>
      %y = aie.objectfifo.acquire @y0(Produce, 1) : memref<64xbf16>
      aie.objectfifo.release @w0(Consume, 1)
      aie.objectfifo.release @y0(Produce, 1)
      aie.end
    }
    %in1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %out1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @w1(%in1, {%core1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @y1(%core1, {%out1}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.core(%core1) {
      %w = aie.objectfifo.acquire @w1(Consume, 1) : memref<1024xbf16>
      %y = aie.objectfifo.acquire @y1(Produce, 1) : memref<64xbf16>
      aie.objectfifo.release @w1(Consume, 1)
      aie.objectfifo.release @y1(Produce, 1)
      aie.end
    }
    %in2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %out2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @w2(%in2, {%core2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @y2(%core2, {%out2}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.core(%core2) {
      %w = aie.objectfifo.acquire @w2(Consume, 1) : memref<1024xbf16>
      %y = aie.objectfifo.acquire @y2(Produce, 1) : memref<64xbf16>
      aie.objectfifo.release @w2(Consume, 1)
      aie.objectfifo.release @y2(Produce, 1)
      aie.end
    }
    %in3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %out3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @w3(%in3, {%core3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @y3(%core3, {%out3}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.core(%core3) {
      %w = aie.objectfifo.acquire @w3(Consume, 1) : memref<1024xbf16>
      %y = aie.objectfifo.acquire @y3(Produce, 1) : memref<64xbf16>
      aie.objectfifo.release @w3(Consume, 1)
      aie.objectfifo.release @y3(Produce, 1)
      aie.end
    }
    aie.runtime_sequence @run(%a0 : memref<65536xbf16>, %b0 : memref<64xbf16>, %a1 : memref<65536xbf16>, %b1 : memref<64xbf16>, %a2 : memref<65536xbf16>, %b2 : memref<64xbf16>, %a3 : memref<65536xbf16>, %b3 : memref<64xbf16>) {
      aiex.npu.dma_memcpy_nd(%a0[0, 0, 0, 0][1, 1, 1, 65536][0, 0, 0, 1]) {metadata = @w0, id = 0 : i64} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%a1[0, 0, 0, 0][1, 1, 1, 65536][0, 0, 0, 1]) {metadata = @w1, id = 2 : i64} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%a2[0, 0, 0, 0][1, 1, 1, 65536][0, 0, 0, 1]) {metadata = @w2, id = 4 : i64} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%a3[0, 0, 0, 0][1, 1, 1, 65536][0, 0, 0, 1]) {metadata = @w3, id = 6 : i64} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%b0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @y0, id = 1 : i64, issue_token = true} : memref<64xbf16>
      aiex.npu.dma_memcpy_nd(%b1[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @y1, id = 3 : i64, issue_token = true} : memref<64xbf16>
      aiex.npu.dma_memcpy_nd(%b2[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @y2, id = 5 : i64, issue_token = true} : memref<64xbf16>
      aiex.npu.dma_memcpy_nd(%b3[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {metadata = @y3, id = 7 : i64, issue_token = true} : memref<64xbf16>
      aiex.npu.dma_wait {symbol = @y0}
      aiex.npu.dma_wait {symbol = @y1}
      aiex.npu.dma_wait {symbol = @y2}
      aiex.npu.dma_wait {symbol = @y3}
    }
  }
}
