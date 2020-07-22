// RUN: aie-opt --aie-create-coremodule %s | FileCheck %s

// CHECK-LABEL: module @test_coremodule0 {
// CHECK:         %[[mem11:.*]] = AIE.mem(1, 1) {
// CHECK:           %[[buf:.*]] = alloc() {id = 0 : i32} : memref<256xi32>
// CHECK:         }
// CHECK:         %[[core11:.*]] = AIE.core(1, 1)
// CHECK:         %[[buf:.*]] = alloc() : memref<256xi32>
// CHECK:         func @host_task() {
// CHECK:           return
// CHECK:         }
// CHECK:         %[[cm11:.*]] = AIE.coreModule(%[[core11]], %[[mem11]]) {
// CHECK:           %[[buf:.*]] = AIE.buffer(%[[mem11]], 0) : memref<256xi32>
// CHECK:         }
// CHECK:         call @host_task() : () -> ()
// CHECK:       }

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
