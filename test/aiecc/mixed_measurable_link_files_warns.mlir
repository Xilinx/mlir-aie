//===- mixed_measurable_link_files_warns.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// stack_size_unmeasurable_warns.mlir covers a core whose ONLY link_files
// object is unmeasurable, where computeStackRequirement itself fails
// (nullopt) because a reached symbol has no frame data at all. This test
// covers the other unmeasurable path, for both reserved_data_size and
// stack_size: a core with a MIX of a measurable object (entry_a, actually
// called and reachable) and an unmeasurable one (an archive, never
// referenced by anything the core calls). Both auto-measurements still
// SUCCEED (the archive's contents are never reached), and each warns purely
// because not every link_files artifact could be inspected, not because
// either computed number is missing. This is the "if (!skipped.empty())"
// branch that exists in both populateReservedDataSize and
// checkStackSizeRequirements (IRTransforms.h) -- reserved_data_size's copy
// had no test of this branch at all, and stack_size's copy only had the
// all-unmeasurable case above.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -fstack-size-section -c %S/stack_size_max_not_sum_kernel.cc -o %t.d/stack_size_max_not_sum_kernel.o
// RUN: llvm-ar rcs %t.d/unrelated.a %t.d/stack_size_max_not_sum_kernel.o
// RUN: cd %t.d && %aiecc %s 2>&1 | FileCheck %s

// CHECK: warning: reserved_data_size auto-measured as {{[0-9]+}} bytes from link_files, but could not inspect 1 artifact(s) (archive, bitcode, or unreadable), so this estimate may be incomplete: {{.*}}unrelated.a
// CHECK: warning: stack requirement computed as {{[0-9]+}} bytes, but 1 link_files artifact(s) could not be inspected (archive, bitcode, or unreadable), so this may be incomplete: {{.*}}unrelated.a

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    // link_files is set directly on the core below (not inferred from
    // link_with here) so both the measurable object and the archive reach
    // this core without the archive ever being treated as reachable.
    func.func private @entry_a(memref<512xi8>)

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @entry_a(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {
      stack_size = 8192 : i32,
      link_files = ["stack_size_max_not_sum_kernel.o", "unrelated.a"]
    }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
