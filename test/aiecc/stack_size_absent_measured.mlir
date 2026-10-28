//===- stack_size_absent_measured.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// stack_size is absent, and the frame of entry_a exceeds the 1024-byte device
// default. aiecc measures the requirement from the probe link before placement,
// and the stack reservation grows to cover it.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_max_not_sum_kernel.cc -o %t.d/stack_size_max_not_sum_kernel.o
// RUN: cd %t.d && %aiecc --get=input_with_addresses.mlir --get=measured_stack_sizes.mlir --output-dir=%t.out --tmpdir=%t.prj %s 2>&1 | FileCheck --check-prefix=BUILD --allow-empty %s
// RUN: FileCheck --check-prefix=PLACED %s < %t.out/input_with_addresses.mlir
// RUN: FileCheck --check-prefix=LDSCRIPT %s < %t.prj/ldScripts_main_core_0_2.ld.script

// BUILD-NOT: error
// BUILD-NOT: warning

// The 4096-byte buffer plus the call chain, aligned to aie2p's 64 bytes. The
// first buffer starts where the stack ends.
// PLACED: aie.buffer(%{{.*}}tile_0_2) {address = {{4[0-9][0-9][0-9]}} : i32
// PLACED: measured_stack_size = {{4[0-9][0-9][0-9]}} : i32
// LDSCRIPT: . += 0x10{{[0-9A-F]}}0; /* stack */

// --no-measure-stack-size keeps the device default.
// RUN: rm -rf %t.noauto.d && mkdir -p %t.noauto.d
// RUN: cp %t.d/stack_size_max_not_sum_kernel.o %t.noauto.d/
// RUN: cd %t.noauto.d && %aiecc --no-measure-stack-size --get=input_with_addresses.mlir --get-core-elfs --output-dir=%t.noauto.out --tmpdir=%t.noauto.prj %s
// RUN: FileCheck --check-prefix=NOAUTO %s < %t.noauto.prj/ldScripts_main_core_0_2.ld.script
// RUN: FileCheck --check-prefix=NOAUTO-ATTR %s < %t.noauto.out/input_with_addresses.mlir
// NOAUTO: . += 0x400; /* stack */
// NOAUTO-ATTR-NOT: measured_stack_size

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_a(memref<512xi8>) attributes {link_with = "stack_size_max_not_sum_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @entry_a(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
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
