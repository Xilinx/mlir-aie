//===- prebaked_elf_ranges.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core with an `elf_file` is never compiled here, so it is never probed
// either, and the allocator used to have no idea the prebaked image held any
// of the tile's data memory. A buffer could be placed straight on top of it,
// silently: the addresses in the ELF are final and nothing downstream compares
// them against the placement.
//
// aiecc now reads the ELF and records what it occupies in `measured_data_ranges`
// as tile-relative address/size pairs. The pass turns each into a pinned
// extent, which is all placement needs -- an address the design fixed itself is
// something it already knows how to avoid.
//
// The values here come from a real npu2 core ELF: `.data` at 0x74000 and an
// `.aie.bank2` table at 0x7bc00, tile-relative 16384 and 48128 against the
// 0x70000 base.

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s

// CHECK-DAG: aie.buffer({{.*}}) {address = 16384 : i32, {{.*}}sym_name = "prebaked_0_2_16384"} : memref<2048xi8>
// CHECK-DAG: aie.buffer({{.*}}) {address = 48128 : i32, {{.*}}sym_name = "prebaked_0_2_48128"} : memref<1024xi8>

// Without the reservation this lands at 16384, exactly on the prebaked .data.
// CHECK-DAG: aie.buffer({{.*}}) {address = 49152 : i32, {{.*}}sym_name = "scratch"}

module @prebaked_elf_ranges {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %scratch = aie.buffer(%t) { sym_name = "scratch" } : memref<16384xi8>
    aie.core(%t) {
      aie.end
    } { elf_file = "prebaked.elf", stack_size = 1024 : i32,
        measured_data_ranges = array<i32: 48128, 1024, 16384, 2048> }
  }
}

// Idempotent, like the other materializers: the pass runs standalone and
// inside larger pipelines.

// RUN: aie-opt --aie-assign-buffer-addresses %s | aie-opt --aie-assign-buffer-addresses | FileCheck %s --check-prefix=TWICE

// TWICE-COUNT-1: sym_name = "prebaked_0_2_16384"
// TWICE-NOT: sym_name = "prebaked_0_2_16384"
