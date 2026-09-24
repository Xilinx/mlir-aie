//===- device_cache.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// --device-cache reuses each aie.device's compiled cores, and its outputs match
// a build that compiles them. Changing one device, a file it links, or the
// command line misses exactly the entries that depend on it.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d/cold %t.d/pop %t.d/hit %t.d/cold_b %t.d/b %t.d/kernel %t.d/attr %t.d/flag %t.d/incomplete %t.d/refill %t.d/core_elfs
// RUN: cp %s %t.d/design.mlir
// RUN: sed 's/arith.constant 7 : i32/arith.constant 8 : i32/' %s > %t.d/changed_b.mlir
// RUN: sed 's/module {/module attributes {cache_test = 1 : i32} {/' %s > %t.d/changed_attr.mlir
// RUN: clang++ --target=aie2p-none-unknown-elf -O2 -DBUMP=1 -c %S/device_cache_kernel.cc -o %t.d/device_cache_kernel.o

// RUN: cd %t.d/cold && aiecc --get-pdi %t.d/design.mlir
// RUN: cd %t.d/pop && aiecc -v --get-pdi --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=POP
// RUN: cd %t.d/hit && aiecc -v --get-pdi --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=HIT --implicit-check-not="device cache miss" --implicit-check-not="device cache store" --implicit-check-not=elfs_
// RUN: cmp %t.d/cold/a.pdi %t.d/pop/a.pdi
// RUN: cmp %t.d/cold/b.pdi %t.d/pop/b.pdi
// RUN: cmp %t.d/cold/a.pdi %t.d/hit/a.pdi
// RUN: cmp %t.d/cold/b.pdi %t.d/hit/b.pdi

// POP-DAG: aiecc: device cache miss: a
// POP-DAG: aiecc: device cache miss: b
// POP-DAG: aiecc: device cache store: a
// POP-DAG: aiecc: device cache store: b

// HIT-DAG: aiecc: device cache hit: a
// HIT-DAG: aiecc: device cache hit: b

// RUN: cd %t.d/cold_b && aiecc --get-pdi %t.d/changed_b.mlir
// RUN: cd %t.d/b && aiecc -v --get-pdi --device-cache=%t.d/cache %t.d/changed_b.mlir 2>&1 | FileCheck %s --check-prefix=B --implicit-check-not="device cache store: a"
// RUN: cmp %t.d/cold_b/a.pdi %t.d/b/a.pdi
// RUN: cmp %t.d/cold_b/b.pdi %t.d/b/b.pdi
// RUN: not cmp -s %t.d/cold/b.pdi %t.d/b/b.pdi

// B-DAG: aiecc: device cache hit: a
// B-DAG: aiecc: device cache miss: b
// B-DAG: aiecc: device cache store: b

// RUN: clang++ --target=aie2p-none-unknown-elf -O2 -DBUMP=2 -c %S/device_cache_kernel.cc -o %t.d/device_cache_kernel.o
// RUN: cd %t.d/kernel && aiecc -v --get-pdi --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=KERNEL --implicit-check-not="device cache store: b"
// RUN: not cmp -s %t.d/cold/a.pdi %t.d/kernel/a.pdi
// RUN: cmp %t.d/cold/b.pdi %t.d/kernel/b.pdi

// KERNEL-DAG: aiecc: device cache miss: a
// KERNEL-DAG: aiecc: device cache hit: b
// KERNEL-DAG: aiecc: device cache store: a

// Module attributes are shared inputs to every device.
// RUN: cd %t.d/attr && aiecc -v --get-pdi --device-cache=%t.d/cache %t.d/changed_attr.mlir 2>&1 | FileCheck %s --check-prefix=POP

// RUN: cd %t.d/flag && aiecc -v --get-pdi --no-measure-data-size --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=POP

// An entry missing any of its files is a miss, and the build replaces it.
// RUN: find %t.d/cache -name "*.elf" -delete
// RUN: cd %t.d/incomplete && aiecc -v --get-pdi --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=POP
// RUN: cd %t.d/refill && aiecc -v --get-pdi --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=HIT --implicit-check-not="device cache miss" --implicit-check-not="device cache store"
// RUN: cmp %t.d/incomplete/a.pdi %t.d/refill/a.pdi
// RUN: cmp %t.d/incomplete/b.pdi %t.d/refill/b.pdi

// Requested placed-core artifacts must not disappear when the cache is warm.
// RUN: mkdir -p %t.d/scripts_cold %t.d/scripts %t.d/scripts_cut
// RUN: cd %t.d/scripts_cold && aiecc --get='ldScripts_{0}.ld.script' --get='{0}.bcf' %t.d/design.mlir
// RUN: cd %t.d/scripts && aiecc -v --get='ldScripts_{0}.ld.script' --get='{0}.bcf' --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=CORE_ELFS --implicit-check-not="device cache hit"
// RUN: cmp %t.d/scripts_cold/ldScripts_a_core_0_2.ld.script %t.d/scripts/ldScripts_a_core_0_2.ld.script
// RUN: cmp %t.d/scripts_cold/ldScripts_b_core_0_2.ld.script %t.d/scripts/ldScripts_b_core_0_2.ld.script
// RUN: cmp %t.d/scripts_cold/ldScripts_b_core_1_2.ld.script %t.d/scripts/ldScripts_b_core_1_2.ld.script
// RUN: cmp %t.d/scripts_cold/a_core_0_2.bcf %t.d/scripts/a_core_0_2.bcf
// RUN: cmp %t.d/scripts_cold/b_core_0_2.bcf %t.d/scripts/b_core_0_2.bcf
// RUN: cmp %t.d/scripts_cold/b_core_1_2.bcf %t.d/scripts/b_core_1_2.bcf
// RUN: cd %t.d/scripts_cut && aiecc -v --get-pdi --cut='ldScripts_{0}.ld.script' --checkpoint=%t.d/scripts_cut/checkpoint --tmpdir=%t.d/scripts_cut/work --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=CORE_ELFS --implicit-check-not="device cache hit"
// RUN: cmp %t.d/scripts_cold/ldScripts_a_core_0_2.ld.script %t.d/scripts_cut/work/ldScripts_a_core_0_2.ld.script
// RUN: cmp %t.d/scripts_cold/ldScripts_b_core_0_2.ld.script %t.d/scripts_cut/work/ldScripts_b_core_0_2.ld.script
// RUN: cmp %t.d/scripts_cold/ldScripts_b_core_1_2.ld.script %t.d/scripts_cut/work/ldScripts_b_core_1_2.ld.script

// A build that emits a core's own ELF compiles every core.
// RUN: cd %t.d/core_elfs && aiecc -v --get-pdi --get-core-elfs --device-cache=%t.d/cache %t.d/design.mlir 2>&1 | FileCheck %s --check-prefix=CORE_ELFS --implicit-check-not="device cache hit"

// CORE_ELFS: aiecc: device cache: not used by this build

module {
  aie.device(npu2) @a {
    %t02 = aie.tile(0, 2)
    %buf = aie.buffer(%t02) {sym_name = "buf_a"} : memref<16xi32>
    func.func private @bump(memref<16xi32>) attributes {link_with = "device_cache_kernel.o"}
    %core = aie.core(%t02) {
      func.call @bump(%buf) : (memref<16xi32>) -> ()
      aie.end
    }
  }
  aie.device(npu2) @b {
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)
    %buf = aie.buffer(%t02) {sym_name = "buf_b"} : memref<16xi32>
    %core_0_2 = aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      memref.store %v, %buf[%c0] : memref<16xi32>
      aie.end
    }
    %core_1_2 = aie.core(%t12) {
      aie.end
    }
  }
}
