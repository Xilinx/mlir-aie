// RUN: aie-opt %s | aie-translate --aie-generate-txn-cpp -o %t_generated.cpp
// RUN: c++ -std=c++17 -I%S/../../../runtime_lib %t_generated.cpp %S/test_generated_wrapper.cpp -o %t_test && %t_test

// Full workflow test: MLIR → C++ generation → compile → execute
// Demonstrates end-to-end runtime transaction generation

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    // Runtime sequence with dynamic parameters
    aie.runtime_sequence @parameterized_sequence(%num_writes: index, %base_addr: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // Generate num_writes write operations
      scf.for %i = %c0 to %num_writes step %c1 {
        // Compute address: base_addr + (i * 4)
        %offset = arith.muli %i, %c4 : index
        %addr_idx = arith.addi %base_addr, %offset : index
        %addr = arith.index_cast %addr_idx : index to i32

        // Compute value: i
        %val = arith.index_cast %i : index to i32

        // Dynamic write
        aiex.npu.dyn_write32(%addr, %val) : i32, i32
      }
    }
  }
}
