// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock5 {
// CHECK-NEXT:  %0 = AIE.tile(5, 5)
// CHECK-NEXT:  %1 = AIE.lock(%0, 0)
// CHECK-NEXT:  %2 = AIE.tile(4, 4)
// CHECK-NEXT:  %3 = AIE.lock(%2, 0)
// CHECK-NEXT:  %4 = AIE.tile(3, 3)
// CHECK-NEXT:  %5 = AIE.lock(%4, 1)
// CHECK-NEXT:  %6 = AIE.lock(%4, 0)
// CHECK-NEXT:  %7 = AIE.buffer(%4) : memref<256xi32>
// CHECK-NEXT:  %8 = AIE.buffer(%2) : memref<256xi32>
// CHECK-NEXT:  %9 = AIE.buffer(%0) : memref<256xi32>
// CHECK-NEXT:  AIE.token(0) {sym_name = "token0"}
// CHECK-NEXT:  AIE.token(0) {sym_name = "token1"}
// CHECK-NEXT:  %10 = AIE.mem(%4) {
// CHECK-NEXT:    %16 = AIE.dmaStart("MM2S0")
// CHECK-NEXT:    %17 = AIE.dmaStart("MM2S1")
// CHECK-NEXT:    AIE.terminator(^bb5, ^bb1, ^bb2)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    cond_br %16, ^bb3, ^bb5
// CHECK-NEXT:  ^bb2:  // pred: ^bb0
// CHECK-NEXT:    cond_br %17, ^bb4, ^bb5
// CHECK-NEXT:  ^bb3:  // pred: ^bb1
// CHECK-NEXT:    AIE.useLock(%5, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.dmaBd(<%7 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useLock(%5, "Release", 0, 0)
// CHECK-NEXT:    br ^bb5
// CHECK-NEXT:  ^bb4:  // pred: ^bb2
// CHECK-NEXT:    AIE.useLock(%6, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.dmaBd(<%7 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useLock(%6, "Release", 0, 0)
// CHECK-NEXT:    br ^bb5
// CHECK-NEXT:  ^bb5:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %11 = AIE.mem(%2) {
// CHECK-NEXT:    %16 = AIE.dmaStart("S2MM0")
// CHECK-NEXT:    AIE.terminator(^bb3, ^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    cond_br %16, ^bb2, ^bb3
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    AIE.useLock(%3, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.dmaBd(<%8 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useLock(%3, "Release", 1, 0)
// CHECK-NEXT:    br ^bb3
// CHECK-NEXT:  ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %12 = AIE.mem(%0) {
// CHECK-NEXT:    %16 = AIE.dmaStart("S2MM0")
// CHECK-NEXT:    AIE.terminator(^bb3, ^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    cond_br %16, ^bb2, ^bb3
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    AIE.useLock(%1, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.dmaBd(<%9 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useLock(%1, "Release", 1, 0)
// CHECK-NEXT:    br ^bb3
// CHECK-NEXT:  ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %13 = AIE.core(%4) {
// CHECK-NEXT:    AIE.useLock(%6, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.useLock(%5, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.useLock(%5, "Release", 1, 0)
// CHECK-NEXT:    AIE.useLock(%6, "Release", 1, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %14 = AIE.core(%2) {
// CHECK-NEXT:    AIE.useLock(%3, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.useLock(%3, "Release", 0, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = AIE.core(%0) {
// CHECK-NEXT:    AIE.useLock(%1, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.useLock(%1, "Release", 0, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  AIE.flow(%4, "DMA" : 0, %2, "DMA" : 0)
// CHECK-NEXT:  AIE.flow(%4, "DMA" : 1, %0, "DMA" : 0)
// CHECK-NEXT:}

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
    %dmaSt0 = AIE.dmaStart("MM2S0")
    %dmaSt1 = AIE.dmaStart("MM2S1")
    AIE.terminator(^end, ^dma0, ^dma1)
    ^dma0:
      cond_br %dmaSt0, ^bd0, ^end
    ^dma1:
      cond_br %dmaSt1, ^bd1, ^end
    ^bd0:
      AIE.useToken @token0("Acquire", 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0("Release", 2)
      br ^end
    ^bd1:
      AIE.useToken @token1("Acquire", 1)
      AIE.dmaBd(<%buf33 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token1("Release", 2)
      br ^end
    ^end:
      AIE.end
  }

  %m44 = AIE.mem(%t44) {
    %dmaSt = AIE.dmaStart("S2MM0")
    AIE.terminator(^end, ^dma0)
    ^dma0:
      cond_br %dmaSt, ^bd0, ^end
    ^bd0:
      AIE.useToken @token0("Acquire", 1)
      AIE.dmaBd(<%buf44 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token0("Release", 2)
      br ^end
    ^end:
      AIE.end
  }

  %m55 = AIE.mem(%t55) {
    %dmaSt = AIE.dmaStart("S2MM0")
    AIE.terminator(^end, ^dma0)
    ^dma0:
      cond_br %dmaSt, ^bd0, ^end
    ^bd0:
      AIE.useToken @token1("Acquire", 1)
      AIE.dmaBd(<%buf55 : memref<256xi32>, 0, 256>, 0)
      AIE.useToken @token1("Release", 2)
      br ^end
    ^end:
      AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token1("Acquire", 0)
    AIE.useToken @token0("Acquire", 0)
    AIE.useToken @token0("Release", 1)
    AIE.useToken @token1("Release", 1)
    AIE.end
  }

  %c44 = AIE.core(%t44) {
    AIE.useToken @token0("Acquire", 2)
    AIE.useToken @token0("Release", 3)
    AIE.end
  }

  %c55 = AIE.core(%t55) {
    AIE.useToken @token1("Acquire", 2)
    AIE.useToken @token1("Release", 3)
    AIE.end
  }

  AIE.flow(%t33, "DMA" : 0, %t44, "DMA" : 0)
  AIE.flow(%t33, "DMA" : 1, %t55, "DMA" : 0)
}
