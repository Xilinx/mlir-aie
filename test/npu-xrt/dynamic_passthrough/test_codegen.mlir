// RUN: aie-translate --aie-generate-txn-cpp %s -o %t.cpp
// RUN: cat %t.cpp | FileCheck %s

// Test C++ code generation for dynamic runtime sequences

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    // Simple dynamic sequence for testing
    // CHECK-LABEL: generate_txn_simple_dynamic
    aie.runtime_sequence @simple_dynamic(%size: index) {
      %c100 = arith.constant 100 : i32
      %c200 = arith.constant 200 : i32

      // CHECK: append_npu_write32
      aiex.npu.dyn_write32(%c100, %c200) : i32, i32
    }

    // Dynamic sequence with loop
    // CHECK-LABEL: generate_txn_loop_dynamic
    aie.runtime_sequence @loop_dynamic(%iterations: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c100 = arith.constant 100 : i32

      // CHECK: for (auto {{.*}} = 0; {{.*}} < {{.*}}; {{.*}} += 1)
      scf.for %i = %c0 to %iterations step %c1 {
        %val = arith.index_cast %i : index to i32
        %addr = arith.addi %val, %c100 : i32
        // CHECK: append_npu_write32
        aiex.npu.dyn_write32(%addr, %val) : i32, i32
      }
    }

    // Dynamic sequence with sync
    // CHECK-LABEL: generate_txn_sync_dynamic
    aie.runtime_sequence @sync_dynamic(%num_syncs: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32

      scf.for %i = %c0 to %num_syncs step %c1 {
        // CHECK: append_npu_sync
        aiex.npu.dyn_sync(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c1_i32, %c1_i32)
          : i32, i32, i32, i32, i32, i32
      }
    }
  }
}
