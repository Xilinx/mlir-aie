// UN: aie-opt --aie-create-locks %s | FileCheck %s

// Fixme: create-locks iterates over maps, so this might fail.

// CHECK-LABEL: module @test_lock5 {
// CHECK:  %0 = AIE.tile(5, 5)
// CHECK:  %1 = AIE.lock(%0, 0)
// CHECK:  %2 = AIE.tile(4, 4)
// CHECK:  %3 = AIE.lock(%2, 0)
// CHECK:  %4 = AIE.tile(3, 3)
// CHECK:  %5 = AIE.lock(%4, 1)
// CHECK:  %6 = AIE.lock(%4, 0)
// CHECK:  %7 = AIE.buffer(%4) : memref<256xi32>
// CHECK:  %8 = AIE.buffer(%2) : memref<256xi32>
// CHECK:  %9 = AIE.buffer(%0) : memref<256xi32>
// CHECK:  AIE.token(0) {sym_name = "token0"}
// CHECK:  AIE.token(0) {sym_name = "token1"}
// CHECK:  %10 = AIE.mem(%4) {
// CHECK:    AIE.useLock({{.*}}, Acquire, 1, 0)
// CHECK:    AIE.dmaBd(<%7 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock({{.*}}, Release, 0, 0)
// CHECK:    AIE.useLock({{.*}}, Acquire, 1, 0)
// CHECK:    AIE.dmaBd(<%7 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock({{.*}}, Release, 0, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %11 = AIE.mem(%2) {
// CHECK:    AIE.useLock(%3, Acquire, 0, 0)
// CHECK:    AIE.dmaBd(<%8 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%3, Release, 1, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %12 = AIE.mem(%0) {
// CHECK:    AIE.useLock(%1, Acquire, 0, 0)
// CHECK:    AIE.dmaBd(<%9 : memref<256xi32>, 0, 256>, 0)
// CHECK:    AIE.useLock(%1, Release, 1, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %13 = AIE.core(%4) {
// CHECK:    AIE.useLock(%[[Lock1:.*]], Acquire, 0, 0)
// CHECK:    AIE.useLock(%[[Lock2:.*]], Acquire, 0, 0)
// CHECK:    AIE.useLock(%[[Lock1]], Release, 1, 0)
// CHECK:    AIE.useLock(%[[Lock2]], Release, 1, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %14 = AIE.core(%2) {
// CHECK:    AIE.useLock(%3, Acquire, 1, 0)
// CHECK:    AIE.useLock(%3, Release, 0, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %15 = AIE.core(%0) {
// CHECK:    AIE.useLock(%1, Acquire, 1, 0)
// CHECK:    AIE.useLock(%1, Release, 0, 0)
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  AIE.flow(%4, DMA : 0, %2, DMA : 0)
// CHECK:  AIE.flow(%4, DMA : 1, %0, DMA : 0)
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---> [Core-Mem] (non-neighboring tiles)
//     |---------> [Core-Mem]
// single producer, multipler consumers
module @test_lock5 {
  %t55 = AIE.tile(5, 5)
  %t44 = AIE.tile(4, 4)
  %t33 = AIE.tile(3, 3)

  %buf33 = AIE.buffer(%t33) : memref<256xi32>
  %buf44 = AIE.buffer(%t44) : memref<256xi32>
  %buf55 = AIE.buffer(%t55) : memref<256xi32>

  AIE.token(0) {sym_name = "token0"}
  AIE.token(0) {sym_name = "token1"}

  %m33 = AIE.mem(%t33) {
      %dmaSt0 = AIE.dmaStart(MM2S0, ^bd0, ^dma0)
    ^dma0:
      %dmaSt1 = AIE.dmaStart("MM2S1", ^bd1, ^end)
    ^bd0:
      AIE.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0(Release, 2)
      br ^end
    ^bd1:
      AIE.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0(Release, 2)
      br ^end
    ^end:
      AIE.end
  }

  %m44 = AIE.mem(%t44) {
      %dmaSt = AIE.dmaStart(S2MM0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf44 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0(Release, 2)
      br ^end
    ^end:
      AIE.end
  }

  %m55 = AIE.mem(%t55) {
      %dmaSt = AIE.dmaStart(S2MM0, ^bd0, ^end)
    ^bd0:
      AIE.useToken @token0(Acquire, 1)
      AIE.dmaBd(<%buf55 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0(Release, 2)
      br ^end
    ^end:
      AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token0(Acquire, 0)
    AIE.useToken @token0(Release, 1)
    AIE.end
  }

  %c44 = AIE.core(%t44) {
    AIE.useToken @token0(Acquire, 2)
    AIE.useToken @token0(Release, 3)
    AIE.end
  }

  %c55 = AIE.core(%t55) {
    AIE.useToken @token0(Acquire, 2)
    AIE.useToken @token0(Release, 3)
    AIE.end
  }

  AIE.flow(%t33, DMA : 0, %t44, DMA : 0)
  AIE.flow(%t33, DMA : 1, %t55, DMA : 0)
}
