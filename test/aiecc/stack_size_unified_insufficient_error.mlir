//===- stack_size_unified_insufficient_error.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The scenario of stack_size_absent_insufficient_error.mlir, built --unified.
// The per-core strategy and the unified strategy both compile a core to
// "objects_<key>.o" (see "keys match coreKey" in splitLoweredCores), so the
// post-build check measures that object under either strategy. Under another
// object path, measureFunctionFrameSize finds nothing and the check passes the
// build.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_max_not_sum_kernel.cc -o %t.d/stack_size_max_not_sum_kernel.o
// RUN: cd %t.d && not %aiecc --unified --get-xclbin --xclbin-name=final.xclbin --output-dir=%t.out %s 2>&1 | FileCheck %s

// CHECK: error: stack_size is absent (this core's buffers were placed assuming the device default of 1024 bytes), but this core's real requirement is {{[0-9]+}} bytes; set stack_size = {{[0-9]+}} explicitly on this aie.core (Worker(stack_size=...) in IRON) and rebuild, or pass --no-auto-stack-size to skip this check

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
