//===- cpp_host_compile.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test host compilation flags with aiecc. Host source files and any extra host
// compiler arguments are passed after the `--` separator; everything before
// `--` is parsed strictly by aiecc (unknown options are rejected, not silently
// forwarded). Runs use dry-run (-n) so the observable aie_inc generation and
// host clang++ invocation are checked without depending on a real host link.

// REQUIRES: peano

// RUN: echo "int main(){return 0;}" > %t.cpp

// aie_inc.cpp generation and host clang++ (C++17) are triggered by --compile-host.
// RUN: aiecc --no-xchesscc --no-xbridge --compile-host -n --verbose %s -- %t.cpp 2>&1 | FileCheck %s

// --host-target propagates to the host compiler target triple.
// RUN: aiecc --no-xchesscc --no-xbridge --compile-host --host-target=aarch64-linux-gnu -n --verbose %s -- %t.cpp 2>&1 | FileCheck %s --check-prefix=AARCH64

// Host source files and extra flags after `--` are forwarded verbatim, in
// order, to the host clang++ driver. This works for AIE2 host compilation
// (device is npu1_1col below), not just AIE1.
// RUN: aiecc --no-xchesscc --no-xbridge --compile-host -n --verbose %s -- -I/some/include -L/some/lib -lsomelib %t.cpp 2>&1 | FileCheck %s --check-prefix=HOSTFULL

// CHECK: ({{[0-9]+}}/{{[0-9]+}}) aie_inc.cpp
// CHECK: exec:{{.*}}clang++{{.*}}-std=c++17

// AARCH64: exec:{{.*}}clang++{{.*}}--target=aarch64-linux-gnu

// HOSTFULL: ({{[0-9]+}}/{{[0-9]+}}) aie_inc.cpp
// HOSTFULL: exec:{{.*}}clang++{{.*}}-I/some/include{{.*}}-L/some/lib{{.*}}-lsomelib{{.*}}.cpp

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32

      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        %result = arith.addi %val, %c1_i32 : i32
        memref.store %result, %elem_out[%i] : memref<16xi32>
      }

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
