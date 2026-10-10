// Two transfers share shim MM2S 0 by packet id, both into tile (0, 2). The
// core needs to_b's object before it frees to_a's buffer, so to_a's 8 words
// stop after the 4 its one buffer holds; issued first, to_a then keeps to_b
// queued behind it.
module {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    %buf0 = aie.buffer(%t) : memref<4xi32>
    %buf1 = aie.buffer(%t) : memref<4xi32>
    %prod0 = aie.lock(%t, 0) {init = 1 : i32, sym_name = "prod0"}
    %cons0 = aie.lock(%t, 1) {init = 0 : i32, sym_name = "cons0"}
    %prod1 = aie.lock(%t, 2) {init = 1 : i32, sym_name = "prod1"}
    %cons1 = aie.lock(%t, 3) {init = 0 : i32, sym_name = "cons1"}
    aie.shim_dma_allocation @to_a(%s, MM2S, 0, <pkt_id = 0>)
    aie.shim_dma_allocation @to_b(%s, MM2S, 0, <pkt_id = 1>)
    aie.packet_flow(0) { aie.packet_source<%s, DMA : 0> aie.packet_dest<%t, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%s, DMA : 0> aie.packet_dest<%t, DMA : 1> }
    %mem = aie.mem(%t) {
      %m1 = arith.constant 1 : i32
      %0 = aie.dma_start(S2MM, 0, ^a, ^next)
    ^a:
      aie.use_lock(%prod0, AcquireGreaterEqual, %m1)
      aie.dma_bd(%buf0 : memref<4xi32> len = 4)
      aie.use_lock(%cons0, Release, %m1)
      aie.next_bd ^a
    ^next:
      %1 = aie.dma_start(S2MM, 1, ^b, ^end)
    ^b:
      aie.use_lock(%prod1, AcquireGreaterEqual, %m1)
      aie.dma_bd(%buf1 : memref<4xi32> len = 4)
      aie.use_lock(%cons1, Release, %m1)
      aie.next_bd ^b
    ^end:
      aie.end
    }
    %core = aie.core(%t) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%cons1, AcquireGreaterEqual, %c1)
      aie.use_lock(%cons0, AcquireGreaterEqual, %c1)
      aie.use_lock(%prod0, Release, %c1)
      aie.use_lock(%cons0, AcquireGreaterEqual, %c1)
      aie.use_lock(%prod0, Release, %c1)
      aie.use_lock(%prod1, Release, %c1)
      aie.end
    }
    aie.runtime_sequence @first_a(%in : memref<8xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {metadata = @to_a, id = 0 : i64} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {metadata = @to_b, id = 1 : i64, issue_token = true} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @to_b}
    }
    aie.runtime_sequence @first_b(%in : memref<8xi32>) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {metadata = @to_b, id = 1 : i64} : memref<8xi32>
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {metadata = @to_a, id = 0 : i64, issue_token = true} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @to_a}
    }
  }
}
