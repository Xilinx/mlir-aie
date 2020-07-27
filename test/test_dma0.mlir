// RUN: aie-opt --aie-create-coremodule %s | FileCheck %s

// CHECK-LABEL: module @test_dma0 {
// CHECK-NEXT:         %[[m22:.*]] = AIE.mem(2, 2) {
// CHECK-NEXT:           %[[buf:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK-NEXT:           %[[dmaSt:.*]] = AIE.dmaStart("S2MM0")
// CHECK-NEXT:           AIE.terminator(^[[end:.*]], ^[[dma0:.*]])
// CHECK-NEXT:           ^[[dma0]]:  // pred: ^bb0
// CHECK-NEXT:             cond_br %[[dmaSt]], ^[[bd0:.*]], ^[[end]]
// CHECK-NEXT:           ^[[bd0]]:  // pred: ^bb1
// CHECK-NEXT:             AIE.useToken @token0("Acquire", 1)
// CHECK-NEXT:             AIE.dmaBd(<%[[buf]] : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:             AIE.useToken @token0("Release", 2)
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
// CHECK-NEXT:           ^[[bd0:.*]]:  // pred: ^[[dma0]]
// CHECK-NEXT:             AIE.useToken @token0("Acquire", 1)
// CHECK-NEXT:             AIE.dmaBd(<%[[buf]] : memref<256xi32>, 0, 256>, 0)
// CHECK-NEXT:             AIE.useToken @token0("Release", 2)
// CHECK-NEXT:             br ^[[end]]
// CHECK-NEXT:           ^[[end]]:  // 3 preds: ^bb0, ^[[dma0]], ^[[bd0]]
// CHECK-NEXT:             AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[c11:.*]] = AIE.core(1, 1)
// CHECK-NEXT:         %[[c22:.*]] = AIE.core(2, 2)
// CHECK-NEXT:         %[[buf0:.*]] = alloc() : memref<256xi32>
// CHECK-NEXT:         %[[buf1:.*]] = alloc() : memref<256xi32>
// CHECK-NEXT:         AIE.token(0) {sym_name = "token0"}
// CHECK-NEXT:         AIE.coreModule<%[[c11]], %[[m11]]> {
// CHECK-NEXT:         ^bb0(%[[core:.*]]: index, %[[mem_w:.*]]: index):  // no predecessors
// CHECK-NEXT:           %[[buf:.*]] = AIE.buffer(%[[mem_w]], 0) : memref<256xi32>
// CHECK-NEXT:           AIE.useToken @token0("Acquire", 0)
// CHECK-NEXT:           AIE.useToken @token0("Release", 1)
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         AIE.flow(%[[c11]], "DMA" : 0, %[[c22]], "DMA" : 0)
// CHECK-NEXT:         AIE.coreModule<%[[c22]], %[[m22]]> {
// CHECK-NEXT:         ^bb0(%[[core:.*]]: index, %[[mem_w:.*]]: index):  // no predecessors
// CHECK-NEXT:           %[[buf:.*]] = AIE.buffer(%[[mem_w]], 0) : memref<256xi32>
// CHECK-NEXT:           AIE.useToken @token0("Acquire", 2)
// CHECK-NEXT:           AIE.useToken @token0("Release", 3)
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:       }

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
