// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu1_1col) {
    aiex.runtime_sequence @sequence(%arg0: memref<64x32xi32>, %arg1: memref<64x32xi32>, %arg2: memref<64x32xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[1, 0, 0, 0][1, 1, 1, 64][0, 0, 1, 1]) {id = 1 : i64, issue_token = true, metadata = @in} : memref<64x32xi32>
      aiex.npu.dma_memcpy_nd(%arg2[1, 0, 0, 0][1920, 1, 1, 2048][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<64x32xi32>
      aiex.npu.dma_wait {symbol = @in}
      aiex.npu.dma_wait {symbol = @out}
    }
    aie.shim_dma_allocation @objectfifo (MM2S, 0, 0)
  }
}

