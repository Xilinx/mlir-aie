//===- stack_size_absent_lower_bound_error.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// stack_size is absent, but entry_cross is compiled without
// -fstack-size-section, so the measurement is only a lower bound and cannot
// size the stack. The core keeps the 1024-byte device default, which the
// 4096-byte frame of helper_cross already exceeds. The build fails and names
// the value to declare.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -c %S/stack_size_cross_object_caller.cc -o %t.d/stack_size_cross_object_caller.o
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O0 -DNDEBUG -ffunction-sections -fdata-sections -fstack-size-section -c %S/stack_size_cross_object_callee.cc -o %t.d/stack_size_cross_object_callee.o
// RUN: cd %t.d && not %aiecc --get-xclbin --xclbin-name=final.xclbin --output-dir=%t.out %s 2>&1 | FileCheck %s

// CHECK: warning: no stack size information for 1 function(s) this core reaches{{.*}}: entry_cross
// CHECK: error: stack_size is absent and this core's stack could not be measured exactly before placement, so it uses the device default of 1024 bytes, but it needs at least {{[0-9]+}} bytes; set stack_size = {{[0-9]+}} (Worker(stack_size=...) in IRON), or pass --no-measure-stack-size to skip this check

// The check fails ahead of the xclbin edge, so a caller that ignores the exit
// code finds no artifact.
// RUN: not ls %t.out/final.xclbin

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @entry_cross(memref<512xi8>)

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<512xi8>
      func.call @entry_cross(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {
      link_files = ["stack_size_cross_object_caller.o", "stack_size_cross_object_callee.o"]
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
