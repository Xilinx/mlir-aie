// RUN: aie-opt --aie-create-coremodule %s | FileCheck %s

// CHECK-LABEL: module @test_coremodule1 {
// CHECK-NEXT:         %[[mem11:.*]] = AIE.mem(1, 1) {
// CHECK-NEXT:           %[[buf0:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK-NEXT:           %[[buf1:.*]] = alloc() {id = 1 : i32} : memref<1xi32>
// CHECK-NEXT:           AIE.terminator(^bb1)
// CHECK-NEXT:         ^bb1:
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[core11:.*]] = AIE.core(1, 1)
// CHECK-NEXT:         %[[buf:.*]] = alloc() : memref<256xi32>
// CHECK-NEXT:         func @host_task() {
// CHECK-NEXT:           return
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[c0:.*]] = constant 0 : i32
// CHECK-NEXT:         AIE.coreModule<%[[core11]], %[[mem11]]> {
// CHECK-NEXT:         ^bb0(%[[core:.*]]: index, %[[mem_w:.*]]: index):  // no predecessors
// CHECK-NEXT:           %[[buf0:.*]] = AIE.buffer(%[[mem_w]], 0) : memref<256xi32>
// CHECK-NEXT:           %[[buf1:.*]] = AIE.buffer(%[[mem_w]], 1) : memref<1xi32>
// CHECK-NEXT:           %[[val0:.*]] = constant 0 : index
// CHECK-NEXT:           %[[val1:.*]] = load %[[buf1]][%[[val0]]] : memref<1xi32>
// CHECK-NEXT:           %[[val2:.*]] = constant 10 : index
// CHECK-NEXT:           store %[[val1]], %[[buf0]][%[[val2]]] : memref<256xi32>
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         call @host_task() : () -> ()
// CHECK-NEXT:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreModuleOp
// In this test, the aie func have both memref argument and scalar argument
// We promote the scalar argument to memref kind (single-element)
// For now, we only support scalar type of int type or float type
module @test_coremodule1 {
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
