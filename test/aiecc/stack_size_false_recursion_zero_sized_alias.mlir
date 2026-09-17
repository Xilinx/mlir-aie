//===- stack_size_false_recursion_zero_sized_alias.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A zero-sized function alias can share another function's address after link.
// A plain address literal to that alias in `.text` must not become a call edge.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang --target=aie2p-none-unknown-elf -c -x assembler %S/stack_size_false_recursion_zero_sized_alias.s -o %t.d/stack_size_false_recursion_zero_sized_alias.o
// RUN: cd %t.d && %aiecc --output-dir=%t.d/prj %s 2>&1 | FileCheck --check-prefix=BUILD --allow-empty %s
// RUN: llvm-readelf -r -s %t.d/prj/elfs_main_core_0_2/elfs_main_core_0_2.elf | FileCheck --check-prefix=ELF %s

// BUILD-NOT: cannot determine this core's stack requirement
// BUILD-NOT: recursion detected

// ELF-DAG: [[ADDR:[0-9A-Fa-f]+]]{{ +}}{{[1-9][0-9]*}}{{ +}}FUNC{{ +}}GLOBAL{{.*}}helper_cycle
// ELF-DAG: [[ADDR]]{{ +}}0{{ +}}FUNC{{ +}}GLOBAL{{.*}}inlined_entry
// ELF: R_AIE_62{{.*}}entry_real
// ELF: R_AIE_62{{.*}}inlined_entry

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @helper_cycle(memref<512xi8>) attributes {link_with = "stack_size_false_recursion_zero_sized_alias.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @helper_cycle(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 1024 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
