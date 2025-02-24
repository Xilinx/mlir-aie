// RUN: aie-opt --split-input-file --aie-dma-to-npu --verify-diagnostics %s

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @out(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo.link [@in] -> [@out]([] [])
    aiex.runtime_sequence @sequence(%arg0: memref<64x32xi32>, %arg1: memref<64x32xi32>, %arg2: memref<64x32xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[1, 0, 0, 0][1, 1, 1, 64][0, 0, 1, 1]) {id = 1 : i64, issue_token = true, metadata = @in} : memref<64x32xi32>
      aiex.npu.dma_memcpy_nd(%arg2[1, 0, 0, 0][1920, 1, 1, 2048][0, 0, 0, 1]) {id = 0 : i64, metadata = @out} : memref<64x32xi32>
     aiex.npu.dma_wait {symbol = @in}
      aiex.npu.dma_wait {symbol = @out}
    }
  }
}

