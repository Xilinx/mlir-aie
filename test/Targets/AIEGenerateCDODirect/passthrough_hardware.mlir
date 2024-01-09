// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $BASENAME.prj %s
// RUN: mkdir $BASENAME.cdo_direct && cp $BASENAME.prj/*.elf $BASENAME.cdo_direct
// RUN: aie-translate --aie-generate-cdo-direct $BASENAME.prj/input_physical.mlir --work-dir-path=$BASENAME.cdo_direct
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_elfs.bin $BASENAME.prj/aie_cdo_elfs.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_enable.bin $BASENAME.prj/aie_cdo_enable.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_error_handling.bin $BASENAME.prj/aie_cdo_error_handling.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_init.bin $BASENAME.prj/aie_cdo_init.bin

module {
  aie.device(ipu) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in] -> [@out]()
    %core_0_2 = aie.core(%tile_0_2) {
      %alloc = memref.alloc() : memref<1xi32>
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      memref.store %c0_i32, %alloc[%c0] : memref<1xi32>
      aie.end
    }
    func.func @sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<4096xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 4096][0, 0, 0]) {id = 1 : i64, metadata = @in} : memref<4096xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}