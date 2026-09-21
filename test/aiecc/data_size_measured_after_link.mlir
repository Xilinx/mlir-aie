//===- data_size_measured_after_link.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aiecc counts a core's static data in its linked ELF, after --gc-sections has
// dropped the sections nothing reads. The kernel carries a 300000-byte table
// that no code reads, which alone exceeds the tile. Counting the object file
// would reject the design; counting the ELF leaves 128 bytes.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -ffunction-sections -fdata-sections -c %S/data_size_measured_after_link_kernel.cc -o %t.d/data_size_measured_after_link_kernel.o
// RUN: cd %t.d && %aiecc --get=measured_data_sizes.mlir --output-dir=%t.out %s
// RUN: FileCheck %s --input-file %t.out/measured_data_sizes.mlir
// RUN: sed 's/data_size = 1024 : i32//' %s > %t.d/implicit.mlir
// RUN: cd %t.d && %aiecc --get-core-elfs --get-input-with-addresses --output-dir=%t.implicit.out implicit.mlir
// RUN: FileCheck %s --check-prefix=AUTO --input-file=%t.implicit.out/input_with_addresses.mlir
// RUN: cd %t.d && %aiecc --no-measure-data-size --get-core-elfs --get-input-with-addresses --output-dir=%t.disabled.out implicit.mlir
// RUN: FileCheck %s --check-prefix=DISABLED --implicit-check-not=measured_data_ --implicit-check-not=core_data --input-file=%t.disabled.out/input_with_addresses.mlir
// RUN: sed 's/data_size = 1024 : i32/data_size = 64 : i32/' %s > %t.d/small.mlir
// RUN: cd %t.d && not %aiecc --get-input-with-addresses --output-dir=%t.small.out small.mlir 2>&1 | FileCheck %s --check-prefix=SMALL
// RUN: cd %t.d && %aiecc --no-measure-data-size --get-input-with-addresses --output-dir=%t.small.disabled.out small.mlir
// RUN: FileCheck %s --check-prefix=EXPLICIT --implicit-check-not=measured_data_ --input-file=%t.small.disabled.out/input_with_addresses.mlir

// CHECK: aie.core(%tile_0_2)
// CHECK: measured_data_size = 128 : i32
// AUTO: aie.buffer({{.*}}) {{.*}}core_data{{.*}} : memref<128xi8>
// AUTO: measured_data_alignment =
// AUTO-SAME: measured_data_size = 128 : i32
// DISABLED: aie.core(
// A disabled measurement does not waive linker overflow for an undersized
// explicit reservation; only request placement for this case.
// SMALL: data_size 64 is smaller than the 128 bytes this core's linked sections occupy
// EXPLICIT: aie.buffer({{.*}}) {{.*}}core_data{{.*}} : memref<64xi8>
// EXPLICIT: data_size = 64 : i32

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
    } { data_size = 1024 : i32 }

    aie.runtime_sequence(%out : memref<64xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c64 = arith.constant 64 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c64][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<64xi8>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
