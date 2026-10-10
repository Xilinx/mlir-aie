// The shim sends to memtile S2MM 0, whose program is not in the design.
module {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    aie.shim_dma_allocation @in(%s, MM2S, 0)
    aie.flow(%s, DMA : 0, %m, DMA : 0)
    aie.runtime_sequence @run(%x : memref<4xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {metadata = @in, id = 0 : i64, issue_token = true} : memref<4xi32>
      aiex.npu.dma_wait {symbol = @in}
    }
  }
}
