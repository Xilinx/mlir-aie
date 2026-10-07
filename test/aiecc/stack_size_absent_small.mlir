//===- stack_size_absent_small.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// stack_size is absent and the core needs less than the 1024-byte device
// default. aiecc records the measurement, and the reservation stays at the
// default, so the layout matches a build that never measured.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_unmeasurable_kernel.cc -o %t.d/stack_size_unmeasurable_kernel.o
// RUN: cd %t.d && %aiecc --get=input_with_addresses.mlir --get=measured_stack_sizes.mlir --output-dir=%t.out --tmpdir=%t.prj %s
// RUN: FileCheck %s < %t.out/input_with_addresses.mlir
// RUN: FileCheck %s --check-prefix=LDSCRIPT < %t.prj/ldScripts_main_core_0_2.ld.script

// CHECK: aie.buffer(%{{.*}}tile_0_2) {address = 1024 : i32
// CHECK: measured_stack_size = {{[0-9]+}} : i32
// LDSCRIPT: . += 0x400; /* stack */

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
