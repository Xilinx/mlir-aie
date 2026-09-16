//===- lut_banks_stack.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An aie::lut<4> whose two tables are function locals.
//
// The gather reads the pair at once and so needs them in separate banks. The
// stack is one contiguous run, so locals cannot be separated at all -- no
// stack_size and no allocator choice can fix it. The pairing is recovered from
// the kernel's own IR, so nothing here carries an annotation.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I%S/../../third_party/aie_api/include -fembed-bitcode -c %S/lut_banks_stack_kernel.cc -o %t.d/lut_banks_stack_kernel.o
// RUN: cd %t.d && not aiecc --get-core-elfs --check-lut-banks %s 2>&1 | FileCheck %s

// CHECK: error: core (0, 2): the aie::lut tables in 'classify'
// CHECK-SAME: are on the stack
// CHECK-SAME: Make them static or pass them in as aie.buffers pinned to different banks
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    func.func private @classify(memref<64xi8>) attributes {link_with = "lut_banks_stack_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<64xi8>
      func.call @classify(%e) : (memref<64xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 8192 : i32 }

    aie.runtime_sequence(%out : memref<64xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
