// RUN: aie-opt --aie-create-coremodule %s | FileCheck %s

// CHECK-LABEL: module @test_dma2 {
// CHECK-NEXT:         %[[m33:.*]] = AIE.mem(3, 3) {
// CHECK-NEXT:           %[[buf0:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK-NEXT:           %[[buf1:.*]] = alloc() {id = 1 : i32} : memref<256xi32>
// CHECK-NEXT:           %[[dmaSt0:.*]] = AIE.dmaStart("S2MM0")
// CHECK-NEXT:           %[[dmaSt1:.*]] = AIE.dmaStart("S2MM0")
// CHECK-NEXT:           AIE.terminator(^[[end:.*]], ^[[dma0:.*]], ^[[dma1:.*]])
// CHECK-NEXT:           ^[[dma0]]:  // pred: ^bb0
// CHECK-NEXT:             cond_br %[[dmaSt0]], ^[[bd0:.*]], ^[[end]]
// CHECK-NEXT:           ^[[bd0]]:  // pred: ^[[dma0]]
// CHECK-NEXT:             AIE.useToken @token0("Acquire", 1)
// CHECK-NEXT:             AIE.dmaBd(<%[[buf0]] : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:             AIE.useToken @token0("Release", 2)
// CHECK-NEXT:             br ^[[end]]
// CHECK-NEXT:           ^[[dma1]]:  // pred: ^bb0
// CHECK-NEXT:             cond_br %[[dmaSt1]], ^[[bd1:.*]], ^[[end]]
// CHECK-NEXT:           ^[[bd1]]:  // pred: ^[[dma1]]
// CHECK-NEXT:             AIE.useToken @token1("Acquire", 1)
// CHECK-NEXT:             AIE.dmaBd(<%[[buf1]] : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:             AIE.useToken @token1("Release", 2)
// CHECK-NEXT:             br ^[[end]]
// CHECK-NEXT:           ^[[end]]:  // 5 preds: ^bb0, ^[[dma0]], ^[[bd0]], ^[[dma1]], ^[[bd1]]
// CHECK-NEXT:             AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[m22:.*]] = AIE.mem(2, 2) {
// CHECK-NEXT:           %[[buf:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK-NEXT:           %[[dmaSt:.*]] = AIE.dmaStart("MM2S0")
// CHECK-NEXT:           AIE.terminator(^[[end:.*]], ^[[dma0:.*]])
// CHECK-NEXT:           ^[[dma0]]:  // pred: ^bb0
// CHECK-NEXT:             cond_br %[[dmaSt]], ^[[bd0:.*]], ^[[end]]
// CHECK-NEXT:           ^[[bd0]]:  // pred: ^[[dma0]]
// CHECK-NEXT:             AIE.useToken @token1("Acquire", 1)
// CHECK-NEXT:             AIE.dmaBd(<%[[buf]] : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:             AIE.useToken @token1("Release", 2)
// CHECK-NEXT:             br ^[[end]]
// CHECK-NEXT:           ^[[end]]:  // 3 preds: ^bb0, ^[[dma0]], ^[[bd0]]
// CHECK-NEXT:             AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[m11:.*]] = AIE.mem(1, 1) {
// CHECK-NEXT:           %[[buf:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK-NEXT:           %[[dmaSt:.*]] = AIE.dmaStart("MM2S0")
// CHECK-NEXT:           AIE.terminator(^[[end:.*]], ^[[dma0:.*]])
// CHECK-NEXT:           ^[[dma0]]:  // pred: ^bb0
// CHECK-NEXT:             cond_br %[[dmaSt]], ^[[bd0:.*]], ^[[end]]
// CHECK-NEXT:           ^[[bd0]]:  // pred: ^[[dma0]]
// CHECK-NEXT:             AIE.useToken @token0("Acquire", 1)
// CHECK-NEXT:             AIE.dmaBd(<%[[buf]] : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:             AIE.useToken @token0("Release", 2)
// CHECK-NEXT:             br ^[[end]]
// CHECK-NEXT:           ^[[end]]:  // 3 preds: ^bb0, ^[[dma0]], ^[[bd0]]
// CHECK-NEXT:             AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[c11:.*]] = AIE.core(1, 1)
// CHECK-NEXT:         %[[c22:.*]] = AIE.core(2, 2)
// CHECK-NEXT:         %[[c33:.*]] = AIE.core(3, 3)
// CHECK-NEXT:         %[[buf0:.*]] = alloc() : memref<256xi32>
// CHECK-NEXT:         %[[buf1:.*]] = alloc() : memref<256xi32>
// CHECK-NEXT:         %[[buf2:.*]] = alloc() : memref<256xi32>
// CHECK-NEXT:         %[[buf3:.*]] = alloc() : memref<256xi32>
// CHECK-NEXT:         AIE.token(0) {sym_name = "token0"}
// CHECK-NEXT:         AIE.token(0) {sym_name = "token1"}
// CHECK-NEXT:         AIE.coreModule<%[[c11]], %[[m11]]> {
// CHECK-NEXT:         ^bb0(%[[core:.*]]: index, %[[mem_w:.*]]: index):  // no predecessors
// CHECK-NEXT:           %[[buf:.*]] = AIE.buffer(%[[mem_w]], 0) : memref<256xi32>
// CHECK-NEXT:           AIE.useToken @token0("Acquire", 0)
// CHECK-NEXT:           AIE.useToken @token0("Release", 1)
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         AIE.coreModule<%[[c22]], %[[m22]]> {
// CHECK-NEXT:         ^bb0(%[[core:.*]]: index, %[[mem_w:.*]]: index):  // no predecessors
// CHECK-NEXT:           %[[buf:.*]] = AIE.buffer(%[[mem_w:.*]], 0) : memref<256xi32>
// CHECK-NEXT:           AIE.useToken @token1("Acquire", 0)
// CHECK-NEXT:           AIE.useToken @token1("Release", 1)
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         AIE.flow(%[[c11]], "DMA" : 0, %[[c33]], "DMA" : 0)
// CHECK-NEXT:         AIE.flow(%[[c22]], "DMA" : 0, %[[c33]], "DMA" : 1)
// CHECK-NEXT:         AIE.coreModule<%[[c33]], %[[m33]]> {
// CHECK-NEXT:         ^bb0(%[[core:.*]]: index, %[[mem_w:.*]]: index):  // no predecessors
// CHECK-NEXT:           %[[buf0:.*]] = AIE.buffer(%[[mem_w]], 0) : memref<256xi32>
// CHECK-NEXT:           %[[buf1:.*]] = AIE.buffer(%[[mem_w]], 1) : memref<256xi32>
// CHECK-NEXT:           AIE.useToken @token0("Acquire", 2)
// CHECK-NEXT:           AIE.useToken @token1("Acquire", 2)
// CHECK-NEXT:           AIE.useToken @token0("Release", 3)
// CHECK-NEXT:           AIE.useToken @token1("Release", 3)
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreModuleOp
// Lowering AIE::memcpy to AIE::DMAStartOp and AIE::DMABDOp
// multiple producers, single consumer
module @test_dma2 {
  %c11 = AIE.core(1, 1) // producer
  %c22 = AIE.core(2, 2) // producer
  %c33 = AIE.core(3, 3) // consumer

  %buf0 = alloc() : memref<256xi32>
  %buf1 = alloc() : memref<256xi32>
  %buf2 = alloc() : memref<256xi32>
  %buf3 = alloc() : memref<256xi32>

  AIE.token(0) { sym_name="token0" }
  AIE.token(0) { sym_name="token1" }

  func @task0(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token0("Acquire", 0)
    // code
    AIE.useToken @token0("Release", 1)
    return
  }

  func @task1(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token1("Acquire", 0)
    // code
    AIE.useToken @token1("Release", 1)
    return
  }

  func @task2(%arg0: memref<256xi32>, %arg1: memref<256xi32>) -> () {
    AIE.useToken @token0("Acquire", 2)
    AIE.useToken @token1("Acquire", 2)
    // code
    AIE.useToken @token0("Release", 3)
    AIE.useToken @token1("Release", 3)
    return
  }

  call @task0(%buf0) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> ()
  call @task1(%buf1) { aie.x = 2, aie.y = 2 } : (memref<256xi32>) -> ()
  AIE.memcpy @token0(1, 2) (%c11 : <%buf0, 0, 256>, %c33 : <%buf2, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  AIE.memcpy @token1(1, 2) (%c22 : <%buf1, 0, 256>, %c33 : <%buf3, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  call @task2(%buf2, %buf3) { aie.x = 3, aie.y = 3 } : (memref<256xi32>, memref<256xi32>) -> ()
}
