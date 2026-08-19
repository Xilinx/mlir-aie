//===- reserved_data_size_measured.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// reserved_data_size is auto-measured from a core's link_files objects: the
// kernel object below has a single 8192-byte zero-initialised static and
// nothing else in .data/.rodata/.bss, so the auto-measured value is exactly
// that plus the driver's fixed margin (kReservedDataMargin, see
// IRTransforms.h) -- 8192 + 256 = 8448.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c %S/reserved_data_size_measured_kernel.cc -o %t.d/reserved_data_size_measured_kernel.o

// The measurement runs at the `reserved_data.mlir` graph edge, before buffer
// address assignment; --cut/--checkpoint captures that intermediate IR.
// RUN: cd %t.d && aiecc --cut='reserved_data.mlir' --checkpoint=%t.d/ckpt %s
// RUN: cat %t.d/ckpt/*/reserved_data.mlir | FileCheck %s

// --no-auto-reserved-data disables the measurement entirely.
// RUN: rm -rf %t.d/noauto.ckpt
// RUN: cd %t.d && aiecc --no-auto-reserved-data --cut='reserved_data.mlir' --checkpoint=%t.d/noauto.ckpt %s
// RUN: cat %t.d/noauto.ckpt/*/reserved_data.mlir | FileCheck --check-prefix=NOAUTO %s

// CHECK: reserved_data_size = 8448 : i32

// NOAUTO: aie.core
// NOAUTO-NOT: reserved_data_size =

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
