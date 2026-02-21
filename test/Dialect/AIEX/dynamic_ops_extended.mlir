// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

// Test parsing and printing of additional dynamic AIEX operations

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    // CHECK-LABEL: @test_dynamic_blockwrite
    aie.runtime_sequence @test_dynamic_blockwrite(%size: index) {
      %c1000 = arith.constant 1000 : i32
      %c10 = arith.constant 10 : i32
      %c20 = arith.constant 20 : i32
      %c30 = arith.constant 30 : i32

      // CHECK: aiex.npu.dyn_blockwrite
      aiex.npu.dyn_blockwrite(%c1000, %c10, %c20, %c30) : i32, i32, i32, i32
    }

    // CHECK-LABEL: @test_dynamic_address_patch
    aie.runtime_sequence @test_dynamic_address_patch(%addr: index) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %addr_i32 = arith.index_cast %addr : index to i32

      // CHECK: aiex.npu.dyn_address_patch
      aiex.npu.dyn_address_patch(%addr_i32, %c0, %c1) : i32, i32, i32
    }

    // CHECK-LABEL: @test_dynamic_push_queue
    aie.runtime_sequence @test_dynamic_push_queue(%bd_id: index) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %bd = arith.index_cast %bd_id : index to i32

      // CHECK: aiex.npu.dyn_push_queue
      aiex.npu.dyn_push_queue(%c0, %c0, %c0, %c0, %c1, %c1, %bd)
        : i32, i32, i32, i32, i32, i32, i32
    }

    // CHECK-LABEL: @test_dynamic_writebd
    aie.runtime_sequence @test_dynamic_writebd(%length: index, %offset: index) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %len = arith.index_cast %length : index to i32
      %off = arith.index_cast %offset : index to i32

      // CHECK: aiex.npu.dyn_writebd
      aiex.npu.dyn_writebd(%c0, %c0, %c0, %len, %off,
                           %c1, %c1, %c1, %c1, %c1, %c1, %c1, %c1)
        : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
    }
  }
}
