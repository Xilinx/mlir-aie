// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt %s | aie-opt | FileCheck %s

// Test parsing and printing of dynamic AIEX operations

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %buf = aie.buffer(%tile_0_2) : memref<1024xi32>

    // CHECK-LABEL: @test_dynamic_ops
    aie.runtime_sequence @test_dynamic_ops(%arg0: memref<1024xi32>, %arg1: index, %arg2: index) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c100 = arith.constant 100 : i32
      %c200 = arith.constant 200 : i32

      // CHECK: aiex.npu.dyn_write32
      // CHECK-SAME: ({{.*}}, {{.*}}) : i32, i32
      aiex.npu.dyn_write32(%c100, %c200) : i32, i32

      // CHECK: aiex.npu.dyn_maskwrite32
      // CHECK-SAME: ({{.*}}, {{.*}}, {{.*}}) : i32, i32, i32
      aiex.npu.dyn_maskwrite32(%c100, %c200, %c1) : i32, i32, i32

      // CHECK: aiex.npu.dyn_sync
      // CHECK-SAME: ({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : i32, i32, i32, i32, i32, i32
      aiex.npu.dyn_sync(%c0, %c0, %c0, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}
