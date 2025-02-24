// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu1_1col) {
    memref.global "public" @out : memref<64x32xi32>
    aiex.runtime_sequence @sequence(%arg0: memref<64x32xi32>, %arg1: memref<64x32xi32>, %arg2: memref<64x32xi32>) {
      aiex.npu.dma_memcpy_nd(%arg2[1, 0, 0, 0][1920, 1, 1, 2048][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<64x32xi32>
      aiex.npu.dma_wait {symbol = @out}
    }
    aie.shim_dma_allocation @out (MM2S, 0, 0)
  }
}

