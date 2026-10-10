// Two cores acquire one lock. If (0, 3) takes it first it releases both, and
// (0, 2) finishes and lets the result out; if (0, 2) takes it first it waits
// on a lock only (0, 3) releases, and (0, 3) waits on the lock (0, 2) holds.
module {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %out = aie.buffer(%a) : memref<4xi32>
    %l = aie.lock(%a, 0) {init = 1 : i32, sym_name = "l"}
    %m = aie.lock(%a, 1) {init = 0 : i32, sym_name = "m"}
    %r = aie.lock(%a, 2) {init = 0 : i32, sym_name = "r"}
    aie.shim_dma_allocation @result(%s, S2MM, 0)
    aie.flow(%a, DMA : 0, %s, DMA : 0)
    %mem = aie.mem(%a) {
      %one = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.use_lock(%r, AcquireGreaterEqual, %one)
      aie.dma_bd(%out : memref<4xi32> len = 4)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %core_a = aie.core(%a) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l, AcquireGreaterEqual, %one)
      aie.use_lock(%m, AcquireGreaterEqual, %one)
      aie.use_lock(%r, Release, %one)
      aie.end
    }
    %core_b = aie.core(%b) {
      %one = arith.constant 1 : i32
      aie.use_lock(%l, AcquireGreaterEqual, %one)
      aie.use_lock(%m, Release, %one)
      aie.use_lock(%l, Release, %one)
      aie.end
    }
    aie.runtime_sequence @run(%res : memref<4xi32>) {
      aiex.npu.dma_memcpy_nd(%res[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {metadata = @result, id = 0 : i64} : memref<4xi32>
      aiex.npu.dma_wait {symbol = @result}
    }
  }
}
