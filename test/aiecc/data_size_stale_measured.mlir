//===- data_size_stale_measured.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The input carries data measurements from some earlier build, such as a
// physical MLIR fed back to aiecc: 8192 bytes of static data, 4096 bytes
// pinned to bank 1, and a prebaked range at 30000. The core's linked sections
// take 128 bytes, none pinned, and it has no elf_file, so none of them stands.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -c %S/data_size_measured_after_link_kernel.cc -o %t.d/data_size_measured_after_link_kernel.o
// RUN: cd %t.d && %aiecc --get-core-elfs --get-input-with-addresses --output-dir=%t.out %s
// RUN: FileCheck %s --implicit-check-not=bank_reserved --implicit-check-not=prebaked --implicit-check-not=measured_bank --implicit-check-not=measured_data_ranges --input-file=%t.out/input_with_addresses.mlir
// RUN: cd %t.d && %aiecc --no-measure-data-size --get-core-elfs --get-input-with-addresses --output-dir=%t.disabled.out %s
// RUN: FileCheck %s --check-prefix=DISABLED --implicit-check-not=measured_data --implicit-check-not=measured_bank --implicit-check-not=core_data --implicit-check-not=bank_reserved --implicit-check-not=prebaked --input-file=%t.disabled.out/input_with_addresses.mlir

// CHECK: aie.buffer({{.*}}) {{.*}}core_data{{.*}} : memref<128xi8>
// CHECK: measured_data_size = 128 : i32
// DISABLED: aie.core(

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    func.func private @classify(memref<64xi8>) attributes {link_with = "data_size_measured_after_link_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %e = aie.objectfifo.acquire @of_out(Produce, 1) : memref<64xi8>
      func.call @classify(%e) : (memref<64xi8>) -> ()
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    } {measured_data_size = 8192 : i32, measured_data_alignment = 16 : i32, measured_bank_sizes = array<i32: 0, 4096, 0, 0>, measured_bank_alignments = array<i32: 1, 16, 1, 1>, measured_data_ranges = array<i32: 30000, 2048>}

    aie.runtime_sequence(%out : memref<64xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
