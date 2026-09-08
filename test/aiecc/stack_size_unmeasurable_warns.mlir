//===- stack_size_unmeasurable_warns.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A kernel object without a `.stack_sizes` section raises a warning, and the
// build continues, see StackSizeAnalysis.h. Its frame counts as 0, so the
// requirement stays a lower bound and the core keeps no measured_stack_size.
// The RUN line compiles the object without -fstack-size-section, which stands
// in for a Chess-compiled or a pre-compiled object.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c %S/stack_size_unmeasurable_kernel.cc -o %t.d/stack_size_unmeasurable_kernel.o
// RUN: cd %t.d && %aiecc --get=measured_stack_sizes.mlir --output-dir=%t.out %s 2>&1 | FileCheck %s
// RUN: FileCheck %s --check-prefix=ATTR --input-file %t.out/measured_stack_sizes.mlir --implicit-check-not=measured_stack_size

// CHECK: warning: no stack size information for 1 function(s) this core reaches
// CHECK-SAME: touch_scratch

// ATTR: aie.core

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
