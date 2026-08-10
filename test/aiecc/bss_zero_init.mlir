//===- bss_zero_init.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for the aie-rt ELF loader out-of-bounds read fixed by
// third_party/patches/aie-rt/0002-elfloader-zero-bss-gap.patch.
//
// A kernel with both .data and .bss links to one PT_LOAD with
// 0 < p_filesz < p_memsz -- the ELF-spec representation of .bss.
// _XAie_LoadDataMemSection used to write p_memsz bytes from a p_filesz-sized
// buffer, so the ELF's own .comment/.symtab bytes were DMA'd into the tile in
// place of zero-initialised statics. The corruption is visible in the generated
// CDO, so this needs no hardware.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c %S/bss_zero_init_kernel.cc -o %t.d/bss_zero_init_kernel.o
// RUN: cd %t.d && aiecc --get-xclbin --xclbin-name=%t.d/test.xclbin --tmpdir=%t.d/prj %s

// Precondition: the segment really is mixed, otherwise the test is vacuous and
// would pass even with the bug present (a p_filesz == 0 segment takes aie-rt's
// separate, correct calloc path).
// RUN: llvm-readelf -lW %t.d/prj/elfs_main_core_0_2/elfs_main_core_0_2.elf | awk '/^  LOAD/ && $7=="RW" {if (strtonum($6) > strtonum($5)) f=1} END {exit !f}'

// The configuration image must not carry ELF metadata: those bytes would be
// written over .bss on the device.
// RUN: not grep -a -q "Linker: LLD" %t.d/prj/cdo_main/main_aie_cdo_elfs.bin
// RUN: not grep -a -q "clang version" %t.d/prj/cdo_main/main_aie_cdo_elfs.bin

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    func.func private @bss_probe(memref<512xi8>) attributes {link_with = "bss_zero_init_kernel.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<512xi8>>
      %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<512xi8>> -> memref<512xi8>
      func.call @bss_probe(%e) : (memref<512xi8>) -> ()
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
