// RUN: aie-opt --aie-create-coremodule %s | FileCheck %s

// CHECK-LABEL: module @test_coremodule0 {
// CHECK-NEXT:         %[[mem11:.*]] = AIE.mem(1, 1) {
// CHECK-NEXT:           %[[buf:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK-NEXT:           AIE.terminator(^bb1)
// CHECK-NEXT:         ^bb1:
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[core11:.*]] = AIE.core(1, 1)
// CHECK-NEXT:         %[[buf:.*]] = alloc() : memref<256xi32>
// CHECK-NEXT:         func @host_task() {
// CHECK-NEXT:           return
// CHECK-NEXT:         }
// CHECK-NEXT:         AIE.coreModule<%[[core11]], %[[mem11]]> {
// CHECK-NEXT:         ^bb0(%[[core:.*]]: index, %[[mem_w:.*]]: index):  // no predecessors
// CHECK-NEXT:           %[[buf:.*]] = AIE.buffer(%[[mem_w]], 0) : memref<256xi32>
// CHECK-NEXT:           AIE.end
// CHECK-NEXT:         }
// CHECK-NEXT:         call @host_task() : () -> ()
// CHECK-NEXT:       }

// Lowering Std::FuncOp and Std::CallOp with (aie.x, aie.y) attributes to AIE::CoreModuleOp
// Basic test
// Things to do when lowering:
//   - create core instance and mem instance if they do not exist already
//   - allocate memory buffers corresponding to the function arguments inside mem's region
//   - convert function arguments to AIE::BufferOp inside CoreModuleOp's region
//   - clone function body (ops) into CoreModuleOp's region
module @test_coremodule0 {
  %buf = alloc() : memref<256xi32>

  func @aie_task(%arg0: memref<256xi32>) -> () {
    return
  }

  func @host_task() -> () {
    return
  }

  call @aie_task(%buf) { aie.x = 1, aie.y = 1 } : (memref<256xi32>) -> ()
  call @host_task() : () -> ()
}
