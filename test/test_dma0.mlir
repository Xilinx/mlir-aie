// RUN: aie-opt --aie-create-coremodule %s | FileCheck %s

// CHECK-LABEL: module @test_dma0 {
// CHECK:         %[[m22:.*]] = AIE.mem(2, 2) {
// CHECK:           %[[buf:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK:           %[[dmaSt:.*]] = AIE.dmaStart("S2MM0")
// CHECK:           ^[[dma0:.*]]:  // no predecessors
// CHECK:             cond_br %[[dmaSt]], ^[[bd0:.*]], ^[[end:.*]]
// CHECK:           ^[[bd0]]:  // pred: ^bb1
// CHECK:             AIE.useToken @token0("Acquire", 1)
// CHECK:             AIE.dmaBd(<%[[buf]] : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.useToken @token0("Release", 2)
// CHECK:             br ^[[end]]
// CHECK:           ^[[end]]:  // 3 preds: ^bb0, ^[[dma0]], ^[[bd0]]
// CHECK:             AIE.end
// CHECK:         }
// CHECK:         %[[m11:.*]] = AIE.mem(1, 1) {
// CHECK:           %[[buf:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK:           %[[dmaSt:.*]] = AIE.dmaStart("MM2S0")
// CHECK:           ^[[dma0]]:  // no predecessors
// CHECK:             cond_br %[[dmaSt]], ^[[bd0:.*]], ^[[end:.*]]
// CHECK:           ^[[bd0:.*]]:  // pred: ^[[dma0]]
// CHECK:             AIE.useToken @token0("Acquire", 1)
// CHECK:             AIE.dmaBd(<%[[buf]] : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.useToken @token0("Release", 2)
// CHECK:             br ^[[end]]
// CHECK:           ^[[end]]:  // 3 preds: ^bb0, ^[[dma0]], ^[[bd0]]
// CHECK:             AIE.end
// CHECK:         }
// CHECK:         %[[c11:.*]] = AIE.core(1, 1)
// CHECK:         %[[c22:.*]] = AIE.core(2, 2)
// CHECK:         %[[c33:.*]] = alloc() : memref<256xi32>
// CHECK:         %[[buf:.*]] = alloc() : memref<256xi32>
// CHECK:         AIE.token(0) {sym_name = "token0"}
// CHECK:         %[[cm11:.*]] = AIE.coreModule(%[[c11]], %[[m11]]) {
// CHECK:           %[[buf:.*]] = AIE.buffer(%[[m11]], 0) : memref<256xi32>
// CHECK:           AIE.useToken @token0("Acquire", 0)
// CHECK:           AIE.useToken @token0("Release", 1)
// CHECK:         }
// CHECK:         %[[cm22:.*]] = AIE.coreModule(%[[c22]], %[[m22]]) {
// CHECK:           %[[buf:.*]] = AIE.buffer(%[[m22]], 0) : memref<256xi32>
// CHECK:           AIE.useToken @token0("Acquire", 2)
// CHECK:           AIE.useToken @token0("Release", 3)
// CHECK:         }
// CHECK:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreModuleOp
// Lowering AIE::memcpy to AIE::DMAStartOp and AIE::DMABDOp
// memcpy is a logical op represents memory transfer of buffers from one core's memory region to
// a different core's memory region
// When lowering, we need to insert the DMA ops on both the sender and receiver
// In addition, we want to model the block descriptions for DMA transfers.
// In the AIE array device, each DMA (of a Core Tile) has 16 Block Descriptions (BD) that are shared
// among four DMA channels (MM2S0, MM2S1, S2MM0, S2MM1). The BDs can be chained together so that
// a DMA channel can process one transfer after another
//
// For now, an MLIR Block is used to denote one BD, and branch op is used to denote the relationship
// of the BDs (what would be the next BD, etc.)
// Using Blocks also allows us to inject lock/token Ops more naturally (instead of having to create
// a new op with locking mechanism -- which is clunky and makes it harder to do locking analysis ...)

module @test_dma0 {
  %c11 = AIE.core(1, 1)
  %c22 = AIE.core(2, 2)

  %buf0 = alloc() : memref<256xi32>
  %buf1 = alloc() : memref<256xi32>

  AIE.token(0) { sym_name="token0" }

  func @task0(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token0("Acquire", 0)
    // code
    AIE.useToken @token0("Release", 1)
    return
  }

  func @task1(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token0("Acquire", 2)
    // code
    AIE.useToken @token0("Release", 3)
    return
  }

  call @task0(%buf0) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> ()
  AIE.memcpy @token0(1, 2) (%c11 : <%buf0, 0, 256>, %c22 : <%buf1, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  call @task1(%buf1) { aie.x = 2, aie.y = 2 } : (memref<256xi32>) -> ()
}
