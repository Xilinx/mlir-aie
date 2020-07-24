// RUN: aie-opt --aie-create-coremodule %s | FileCheck %s

// CHECK-LABEL: module @test_dma0 {
// CHECK:         %0 = AIE.mem(2, 2) {
// CHECK:           %8 = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK:           %9 = AIE.dmaStart("S2MM0")
// CHECK:           ^bb1:  // pred: ^bb0
// CHECK:             cond_br %9, ^bb2, ^bb3
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useToken @token0("Acquire", 1)
// CHECK:             AIE.dmaBd(<%8 : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.useToken @token0("Release", 2)
// CHECK:             br ^bb3
// CHECK:           ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK:             AIE.end
// CHECK:         }
// CHECK:         %1 = AIE.mem(1, 1) {
// CHECK:           %8 = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK:           %9 = AIE.dmaStart("MM2S0")
// CHECK:           ^bb1:  // pred: ^bb0
// CHECK:             cond_br %9, ^bb2, ^bb3
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useToken @token0("Acquire", 1)
// CHECK:             AIE.dmaBd(<%8 : memref<256xi32>, 0, 256>, 0)
// CHECK:             AIE.useToken @token0("Release", 2)
// CHECK:             br ^bb3
// CHECK:           ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK:             AIE.end
// CHECK:         }
// CHECK:         %2 = AIE.core(1, 1)
// CHECK:         %3 = AIE.core(2, 2)
// CHECK:         %4 = alloc() : memref<256xi32>
// CHECK:         %5 = alloc() : memref<256xi32>
// CHECK:         AIE.token(0) {sym_name = "token0"}
// CHECK:         %6 = AIE.coreModule(%2, %1) {
// CHECK:           %8 = AIE.buffer(%1, 0) : memref<256xi32>
// CHECK:           AIE.useToken @token0("Acquire", 0)
// CHECK:           AIE.useToken @token0("Release", 1)
// CHECK:         }
// CHECK:         %7 = AIE.coreModule(%3, %0) {
// CHECK:           %8 = AIE.buffer(%0, 0) : memref<256xi32>
// CHECK:           AIE.useToken @token0("Acquire", 2)
// CHECK:           AIE.useToken @token0("Release", 3)
// CHECK:         }
// CHECK:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreModuleOp
// Lowering AIE::memcpy to AIE::DMAStartOp and AIE::DMABDOp
// memcpy is a logical op represents memory tranfer of buffers from one core's memory region to
// a different core's memory region
// When lowering, we need to insert the DMA ops on both the sender and receiver
// In addition, we want to model the block descriptions for DMA transfers.
// In the AIE array device, each DMA (of a Core Tile) has 16 Block Descriptions (BD) that are shared
// among four DMA channels (MM2S0, MM2S1, S2MM0, S2MM1). The BDs can be chained together so that
// a DMA channel can process one transfer after another
//
// For now, a MLIR Block is used to denote one BD, and branch op is used to denote the relationship
// of the BD (what would be the next BD, etc.)
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
