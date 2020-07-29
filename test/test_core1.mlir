// RUN: aie-opt --aie-create-cores %s | FileCheck %s

// CHECK-LABEL: module @test_core1 {
// CHECK-NEXT:    %0 = AIE.tile(1, 1)
// CHECK-NEXT:    %1 = AIE.buffer(%0) : memref<256xi32>
// CHECK-NEXT:    %2 = AIE.buffer(%0) : memref<1xi32>
// CHECK-NEXT:    %3 = AIE.mem(%0) {
// CHECK-NEXT:      AIE.terminator(^bb1)
// CHECK-NEXT:      ^bb1:  // pred: ^bb0
// CHECK-NEXT:        AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = alloc() : memref<256xi32>
// CHECK-NEXT:    func @host_task() {
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    %c0_i32 = constant 0 : i32
// CHECK-NEXT:    %5 = AIE.core(%0) {
// CHECK-NEXT:      %c0 = constant 0 : index
// CHECK-NEXT:      %6 = load %2[%c0] : memref<1xi32>
// CHECK-NEXT:      %c10 = constant 10 : index
// CHECK-NEXT:      store %6, %1[%c10] : memref<256xi32>
// CHECK-NEXT:      AIE.end
// CHECK-NEXT:    }
// CHECK-NEXT:    call @host_task() : () -> ()
// CHECK-NEXT:  }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreOp,
// AIE::MemOp, and AIE::TileOp
// In this test, the aie func have both memref argument and scalar argument
// We promote the scalar argument to memref kind (single-element)
// For now, we only support scalar type of int type or float type
module @test_core1 {
  %buf = alloc() : memref<256xi32>

  func @aie_task(%arg0: memref<256xi32>, %arg1: i32) -> () {
    %i = constant 10 : index
    store %arg1, %arg0[%i] : memref<256xi32>
    return
  }

  func @host_task() -> () {
    return
  }

  %a = constant 0 : i32
  call @aie_task(%buf, %a) { aie.x = 1, aie.y = 1 } : (memref<256xi32>, i32) -> ()
  call @host_task() : () -> ()
}
