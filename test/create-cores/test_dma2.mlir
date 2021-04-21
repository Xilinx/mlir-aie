// REQUIRES: unknown
// RUN: aie-opt --aie-create-cores %s | FileCheck %s

// CHECK-LABEL: module @test_dma2 {
// CHECK-NEXT:  %0 = AIE.tile(1, 1)
// CHECK-NEXT:  %1 = AIE.buffer(%0) : memref<256xi32>
// CHECK-NEXT:  %2 = AIE.mem(%0) {
// CHECK-NEXT:    %17 = AIE.dmaStart(MM2S0)
// CHECK-NEXT:    AIE.terminator(^bb3, ^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    cond_br %17, ^bb2, ^bb3
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    AIE.useToken @token0(Acquire, 1)
// CHECK-NEXT:    AIE.dmaBd(<%1 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useToken @token0(Release, 2)
// CHECK-NEXT:    br ^bb3
// CHECK-NEXT:  ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = AIE.tile(2, 2)
// CHECK-NEXT:  %4 = AIE.buffer(%3) : memref<256xi32>
// CHECK-NEXT:  %5 = AIE.mem(%3) {
// CHECK-NEXT:    %17 = AIE.dmaStart(MM2S0)
// CHECK-NEXT:    AIE.terminator(^bb3, ^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    cond_br %17, ^bb2, ^bb3
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    AIE.useToken @token1(Acquire, 1)
// CHECK-NEXT:    AIE.dmaBd(<%4 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useToken @token1(Release, 2)
// CHECK-NEXT:    br ^bb3
// CHECK-NEXT:  ^bb3:  // 3 preds: ^bb0, ^bb1, ^bb2
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %6 = AIE.tile(3, 3)
// CHECK-NEXT:  %7 = AIE.buffer(%6) : memref<256xi32>
// CHECK-NEXT:  %8 = AIE.buffer(%6) : memref<256xi32>
// CHECK-NEXT:  %9 = AIE.mem(%6) {
// CHECK-NEXT:    %17 = AIE.dmaStart(S2MM0)
// CHECK-NEXT:    %18 = AIE.dmaStart(S2MM0)
// CHECK-NEXT:    AIE.terminator(^bb5, ^bb1, ^bb3)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    cond_br %17, ^bb2, ^bb5
// CHECK-NEXT:  ^bb2:  // pred: ^bb1
// CHECK-NEXT:    AIE.useToken @token0(Acquire, 1)
// CHECK-NEXT:    AIE.dmaBd(<%7 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useToken @token0(Release, 2)
// CHECK-NEXT:    br ^bb5
// CHECK-NEXT:  ^bb3:  // pred: ^bb0
// CHECK-NEXT:    cond_br %18, ^bb4, ^bb5
// CHECK-NEXT:  ^bb4:  // pred: ^bb3
// CHECK-NEXT:    AIE.useToken @token1(Acquire, 1)
// CHECK-NEXT:    AIE.dmaBd(<%8 : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:    AIE.useToken @token1(Release, 2)
// CHECK-NEXT:    br ^bb5
// CHECK-NEXT:  ^bb5:  // 5 preds: ^bb0, ^bb1, ^bb2, ^bb3, ^bb4
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %10 = alloc() : memref<256xi32>
// CHECK-NEXT:  %11 = alloc() : memref<256xi32>
// CHECK-NEXT:  %12 = alloc() : memref<256xi32>
// CHECK-NEXT:  %13 = alloc() : memref<256xi32>
// CHECK-NEXT:  AIE.token(0) {sym_name = "token0"}
// CHECK-NEXT:  AIE.token(0) {sym_name = "token1"}
// CHECK-NEXT:  %14 = AIE.core(%0) {
// CHECK-NEXT:    AIE.useToken @token0(Acquire, 0)
// CHECK-NEXT:    AIE.useToken @token0(Release, 1)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = AIE.core(%3) {
// CHECK-NEXT:    AIE.useToken @token1(Acquire, 0)
// CHECK-NEXT:    AIE.useToken @token1(Release, 1)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  AIE.flow(%0, DMA : 0, %6, DMA : 0)
// CHECK-NEXT:  AIE.flow(%3, DMA : 0, %6, DMA : 1)
// CHECK-NEXT:  %16 = AIE.core(%6) {
// CHECK-NEXT:    AIE.useToken @token0(Acquire, 2)
// CHECK-NEXT:    AIE.useToken @token1(Acquire, 2)
// CHECK-NEXT:    AIE.useToken @token0(Release, 3)
// CHECK-NEXT:    AIE.useToken @token1(Release, 3)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:}

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreOp,
// AIE::MemOp, and AIE::TileOp
// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreModuleOp
// Lowering AIE::memcpy to AIE::DMAStartOp and AIE::DMABDOp
// multiple producers, single consumer
module @test_dma2 {
  %t11 = AIE.tile(1, 1) // producer
  %t22 = AIE.tile(2, 2) // producer
  %t33 = AIE.tile(3, 3) // consumer

  %buf0 = alloc() : memref<256xi32>
  %buf1 = alloc() : memref<256xi32>
  %buf2 = alloc() : memref<256xi32>
  %buf3 = alloc() : memref<256xi32>

  AIE.token(0) { sym_name="token0" }
  AIE.token(0) { sym_name="token1" }

  func @task0(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token0(Acquire, 0)
    // code
    AIE.useToken @token0(Release, 1)
    return
  }

  func @task1(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token1(Acquire, 0)
    // code
    AIE.useToken @token1(Release, 1)
    return
  }

  func @task2(%arg0: memref<256xi32>, %arg1: memref<256xi32>) -> () {
    AIE.useToken @token0(Acquire, 2)
    AIE.useToken @token1(Acquire, 2)
    // code
    AIE.useToken @token0(Release, 3)
    AIE.useToken @token1(Release, 3)
    return
  }

  call @task0(%buf0) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> ()
  call @task1(%buf1) { aie.x = 2, aie.y = 2 } : (memref<256xi32>) -> ()
  AIE.memcpy @token0(1, 2) (%t11 : <%buf0, 0, 256>, %t33 : <%buf2, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  AIE.memcpy @token1(1, 2) (%t22 : <%buf1, 0, 256>, %t33 : <%buf3, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  call @task2(%buf2, %buf3) { aie.x = 3, aie.y = 3 } : (memref<256xi32>, memref<256xi32>) -> ()
}
