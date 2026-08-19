//===- reserved_data_size_undersized.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A tile too small to give an auto-measured reservation the contiguous room
// it needs must fail at buffer address assignment with the allocator's usual
// clean diagnostic (see bank_aware_prealloc_error.mlir's @test6, which
// exercises the same check with a hand-written value) -- never downstream as
// an opaque linker region-overflow error. This is what keeps a wrong
// auto-measurement from turning a loud, actionable error into a confusing
// one: however the reservation number was produced, too-small always fails
// here, before a linker ever runs.
//
// tile(0, 2) on npu2 has 65536 bytes of local data memory. Two 28672-byte
// buffers, the objectfifo's own 1024-byte buffer, and a 1024-byte stack
// leave 6144 contiguous bytes, less than the 8448 the linked kernel's .bss
// auto-measures to (see reserved_data_size_measured.mlir).

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c %S/reserved_data_size_measured_kernel.cc -o %t.d/reserved_data_size_measured_kernel.o
// RUN: cd %t.d && not %aiecc %s 2>&1 | FileCheck %s

// CHECK: warning: buffers leave only 6144 contiguous bytes for the core's data sections, which need 8448 bytes
// CHECK: warning: Bank-aware allocation failed, trying basic sequential allocation.
// CHECK: warning: buffers leave only 6144 contiguous bytes for the core's data sections, which need 8448 bytes.
// CHECK: error: 'aie.tile' op Basic sequential allocation also failed.
// CHECK-NOT: ld.lld
// CHECK-NOT: region 'data' overflowed

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<28672xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<28672xi8>

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @touch_scratch(memref<512xi8>) attributes {link_with = "reserved_data_size_measured_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @touch_scratch(%e) : (memref<512xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } { stack_size = 1024 : i32 }

    aie.runtime_sequence(%out : memref<512xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c512 = arith.constant 512 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c512][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<512xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
