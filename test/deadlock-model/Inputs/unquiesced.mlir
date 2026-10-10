// The sequence sends 8 words to a tile that takes 4 and never waits, so the
// dispatch ends with the shim channel stuck halfway: the next dispatch would
// find it still running.
module {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) : memref<4xi32>
    %p = aie.lock(%t, 0) {init = 1 : i32, sym_name = "p"}
    %c = aie.lock(%t, 1) {init = 0 : i32, sym_name = "c"}
    aie.shim_dma_allocation @in(%s, MM2S, 0)
    aie.flow(%s, DMA : 0, %t, DMA : 0)
    %mem = aie.mem(%t) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^bd, ^end)
    ^bd:
      aie.use_lock(%p, AcquireGreaterEqual, %one)
      aie.dma_bd(%buf : memref<4xi32> len = 4)
      aie.use_lock(%c, Release, %one)
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<8xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {metadata = @in, id = 0 : i64} : memref<8xi32>
    }
  }
}
