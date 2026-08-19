//===- reserved_data_size_explicit_wins.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An explicit reserved_data_size -- including 0 -- must never be overwritten
// by auto-measurement, even though the linked kernel object here has a real
// 8192-byte .bss that would otherwise measure to 8448 (see
// reserved_data_size_measured.mlir). "= 0" is a legal way to say "reserve
// nothing, I know what I'm doing" and must stay exactly that.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c %S/reserved_data_size_measured_kernel.cc -o %t.d/reserved_data_size_measured_kernel.o
// RUN: cd %t.d && aiecc --cut='reserved_data.mlir' --checkpoint=%t.d/ckpt %s
// RUN: cat %t.d/ckpt/*/reserved_data.mlir | FileCheck %s

// CHECK: reserved_data_size = 0 : i32

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @touch_scratch(memref<512xi8>) attributes {link_with = "reserved_data_size_measured_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @touch_scratch(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { reserved_data_size = 0 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
