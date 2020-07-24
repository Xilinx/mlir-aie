// RUN: aie-opt %s | FileCheck %s

// This is the logical netlist of mapping code to AIE array. This is closer to the application-level
// Tokens (logical locks) are used to denote the execution order of the cores
// Attributes are used to specify binding code to a core
// Memcpy op is used to perform logical memory transfer from one core to another
//
// In the example below, core(3, 3) is a producer, while core(4, 2) and core(4, 4) are consumers.
// Since these cores cannot share memory physically (they are not abut), a memory transfer op is used
// to explicitly move buffers from the producer to the consumers

// CHECK-LABEL: module @example1 {
// CHECK:       }

module @example1 {
  %c33 = AIE.core(3, 3)
  %c42 = AIE.core(4, 2)
  %c44 = AIE.core(4, 4)

  %buf0 = alloc() : memref<256xi32>
  %buf1 = alloc() : memref<256xi32>
  %buf2 = alloc() : memref<256xi32>

  AIE.token(0) { sym_name="token0" }
  AIE.token(0) { sym_name="token1" }

  func @task0(%arg0: memref<256xi32>, %arg1: i32) -> () {
    AIE.useToken @token0("Acquire", 0)
    AIE.useToken @token1("Acquire", 0)

    // code
    %i = constant 8: index
    //%k = constant 1: i32
    store %arg1, %arg0[%i] : memref<256xi32>
    //store %k, %arg0[%i] : memref<256xi32>

    AIE.useToken @token0("Release", 1)
    AIE.useToken @token1("Release", 1)
    return
  }

  func @task1(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token0("Acquire", 2)

    // code

    AIE.useToken @token0("Release", 3)
    return
  }

  func @task2(%arg0: memref<256xi32>) -> () {
    AIE.useToken @token1("Acquire", 2)

    // code

    AIE.useToken @token1("Release", 3)
    return
  }

  func @task3() -> () {
    return
  }

  %t0 = constant 19 : i32
  call @task0(%buf0, %t0) { aie.x = 3, aie.y = 3 } : (memref<256xi32>, i32) -> ()
  call @task1(%buf1) { aie.x = 4, aie.y = 2 } : (memref<256xi32>) -> ()
  call @task2(%buf2) { aie.x = 4, aie.y = 4 } : (memref<256xi32>) -> ()
  call @task3() : () -> ()

  AIE.memcpy @token0(1, 2) (%c33 : <%buf0, 0, 256>, %c42 : <%buf1, 0, 256>) : (memref<256xi32>, memref<256xi32>)
  AIE.memcpy @token1(1, 2) (%c33 : <%buf0, 0, 256>, %c44 : <%buf2, 0, 256>) : (memref<256xi32>, memref<256xi32>)
}
