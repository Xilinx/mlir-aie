// RUN: aie-opt --convert-aiex-to-emitc %s | FileCheck %s

// Test conversion of dynamic AIEX operations to EmitC

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // CHECK-LABEL: @test_dyn_write32_conversion
    aie.runtime_sequence @test_dyn_write32_conversion(%arg0: memref<1024xi32>) {
      %c100 = arith.constant 100 : i32
      %c200 = arith.constant 200 : i32

      // CHECK: emitc.call "append_npu_write32"
      // CHECK-SAME: (%{{.*}}, %{{.*}}) : (i32, i32) -> ()
      aiex.npu.dyn_write32(%c100, %c200) : i32, i32
    }

    // CHECK-LABEL: @test_dyn_maskwrite32_conversion
    aie.runtime_sequence @test_dyn_maskwrite32_conversion(%arg0: memref<1024xi32>) {
      %c100 = arith.constant 100 : i32
      %c200 = arith.constant 200 : i32
      %c1 = arith.constant 1 : i32

      // CHECK: emitc.call "append_npu_maskwrite32"
      // CHECK-SAME: (%{{.*}}, %{{.*}}, %{{.*}}) : (i32, i32, i32) -> ()
      aiex.npu.dyn_maskwrite32(%c100, %c200, %c1) : i32, i32, i32
    }

    // CHECK-LABEL: @test_dyn_sync_conversion
    aie.runtime_sequence @test_dyn_sync_conversion(%arg0: memref<1024xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32

      // CHECK: emitc.call "append_npu_sync"
      // CHECK-SAME: (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (i32, i32, i32, i32, i32, i32) -> ()
      aiex.npu.dyn_sync(%c0, %c0, %c0, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}
