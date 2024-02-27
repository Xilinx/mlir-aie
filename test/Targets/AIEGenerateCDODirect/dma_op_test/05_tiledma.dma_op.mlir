// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename -s .dma_op.mlir %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.dma_op.prj $BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_start.prj && pushd $BASENAME.dma_start.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %S/$BASENAME.dma_start && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_start.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_op.prj && pushd $BASENAME.dma_op.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %s && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_op.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_op.prj

// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_error_handling.bin $BASENAME.dma_start.prj/aie_cdo_error_handling.bin
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_init.bin $BASENAME.dma_start.prj/aie_cdo_init.bin


module @test05_tiledma {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %a13 = aie.buffer(%tile_1_3) {sym_name = "a13"} : memref<256xi32>
    %b13 = aie.buffer(%tile_1_3) {sym_name = "b13"} : memref<256xi32>
    %a33 = aie.buffer(%tile_3_3) {sym_name = "a33"} : memref<256xi32>
    %b33 = aie.buffer(%tile_3_3) {sym_name = "b33"} : memref<256xi32>
    %input_lock = aie.lock(%tile_1_3, 3) {sym_name = "input_lock"}
    %interlock1 = aie.lock(%tile_1_3, 5) {sym_name = "interlock1"}
    %interlock2 = aie.lock(%tile_3_3, 6) {sym_name = "interlock2"}
    %output_lock = aie.lock(%tile_3_3, 7) {sym_name = "output_lock"}
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<DMA : 0, East : 1>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<West : 1, East : 3>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<West : 3, DMA : 1>
    }
    %core_1_3 = aie.core(%tile_1_3) {
      aie.use_lock(%input_lock, Acquire, 1)
      aie.use_lock(%interlock1, Acquire, 0)
      %c3 = arith.constant 3 : index
      %0 = memref.load %a13[%c3] : memref<256xi32>
      %1 = arith.addi %0, %0 : i32
      %2 = arith.addi %1, %0 : i32
      %3 = arith.addi %2, %0 : i32
      %4 = arith.addi %3, %0 : i32
      %c5 = arith.constant 5 : index
      memref.store %4, %b13[%c5] : memref<256xi32>
      aie.use_lock(%input_lock, Release, 0)
      aie.use_lock(%interlock1, Release, 1)
      aie.end
    }
    %core_3_3 = aie.core(%tile_3_3) {
      aie.use_lock(%interlock2, Acquire, 1)
      aie.use_lock(%output_lock, Acquire, 0)
      %c5 = arith.constant 5 : index
      %0 = memref.load %a33[%c5] : memref<256xi32>
      %1 = arith.addi %0, %0 : i32
      %2 = arith.addi %1, %0 : i32
      %3 = arith.addi %2, %0 : i32
      %4 = arith.addi %3, %0 : i32
      %c5_0 = arith.constant 5 : index
      memref.store %4, %b33[%c5_0] : memref<256xi32>
      aie.use_lock(%interlock2, Release, 0)
      aie.use_lock(%output_lock, Release, 1)
      aie.end
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      aie.dma(MM2S, 0) {loop = false} [{
        aie.use_lock(%interlock1, Acquire, 1)
        aie.dma_bd(%b13 : memref<256xi32>, 0, 256)
        aie.use_lock(%interlock1, Release, 0)
      }]
      aie.end
    }
    %mem_3_3 = aie.mem(%tile_3_3) {
      aie.dma(S2MM, 1) {loop = false} [{
        aie.use_lock(%interlock2, Acquire, 0)
        aie.dma_bd(%a33 : memref<256xi32>, 0, 256)
        aie.use_lock(%interlock2, Release, 1)
      }]
      aie.end
    }
  }
}

