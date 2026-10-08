//===- stack_size_stale_measured.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The input carries an 8192-byte measured_stack_size from some earlier build,
// such as a physical MLIR fed back to aiecc. The core needs less than the
// 1024-byte default, so the reservation stays at the default whether aiecc
// measures again or not.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_unmeasurable_kernel.cc -o %t.d/stack_size_unmeasurable_kernel.o
// RUN: cd %t.d && %aiecc --get=input_with_addresses.mlir --get=measured_stack_sizes.mlir --output-dir=%t.out --tmpdir=%t.prj %s
// RUN: FileCheck %s < %t.out/input_with_addresses.mlir
// RUN: FileCheck %s --check-prefix=LDSCRIPT < %t.prj/ldScripts_main_core_0_2.ld.script

// CHECK: aie.buffer(%{{.*}}tile_0_2) {address = 1024 : i32
// CHECK-NOT: measured_stack_size = 8192
// LDSCRIPT: . += 0x400; /* stack */

// RUN: rm -rf %t.noauto.d && mkdir -p %t.noauto.d
// RUN: cp %t.d/stack_size_unmeasurable_kernel.o %t.noauto.d/
// RUN: cd %t.noauto.d && %aiecc --no-measure-stack-size --get=input_with_addresses.mlir --get-core-elfs --output-dir=%t.noauto.out --tmpdir=%t.noauto.prj %s
// RUN: FileCheck %s --check-prefix=NOAUTO < %t.noauto.out/input_with_addresses.mlir
// RUN: FileCheck %s --check-prefix=LDSCRIPT < %t.noauto.prj/ldScripts_main_core_0_2.ld.script

// NOAUTO: aie.buffer(%{{.*}}tile_0_2) {address = 1024 : i32
// NOAUTO-NOT: measured_stack_size

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @touch_scratch(memref<512xi8>) attributes {link_with = "stack_size_unmeasurable_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @touch_scratch(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {measured_stack_size = 8192 : i32}

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
