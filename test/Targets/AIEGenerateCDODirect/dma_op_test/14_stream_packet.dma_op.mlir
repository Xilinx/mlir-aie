// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename -s .dma_op.mlir %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.dma_op.prj $BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_start.prj && pushd $BASENAME.dma_start.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %S/$BASENAME.dma_start && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_start.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_op.prj && pushd $BASENAME.dma_op.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %s && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_op.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_op.prj

// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_error_handling.bin $BASENAME.dma_start.prj/aie_cdo_error_handling.bin
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_init.bin $BASENAME.dma_start.prj/aie_cdo_init.bin

module @test14_stream_packet {
  aie.device(ipu) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %switchbox_1_3 = aie.switchbox(%tile_1_4) {
      aie.connect<DMA : 0, South : 3>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<DMA : 0, North : 1>
    }
    %switchbox_1_2_1 = aie.switchbox(%tile_1_3) {
      %0 = aie.amsel<1> (0)
      %1 = aie.masterset(West : 3, %0)
      aie.packet_rules(North : 3) {
        aie.rule(31, 13, %0)
      }
      aie.packet_rules(South : 1) {
        aie.rule(31, 12, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<East : 3, DMA : 0>
    }
    %buf73 = aie.buffer(%tile_1_4) {sym_name = "buf73"} : memref<256xi32>
    %buf71 = aie.buffer(%tile_1_2) {sym_name = "buf71"} : memref<256xi32>
    %lock73 = aie.lock(%tile_1_4, 0) {sym_name = "lock73"}
    %lock71 = aie.lock(%tile_1_2, 0) {sym_name = "lock71"}
    %mem_1_3 = aie.mem(%tile_1_4) {
      aie.dma(MM2S, 0) {loop = false} [{
        aie.use_lock(%lock73, Acquire, 0)
        aie.dma_bd_packet(5, 13)
        aie.dma_bd(%buf73 : memref<256xi32>, 0, 256)
        aie.use_lock(%lock73, Release, 1)
      }]
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      aie.dma(MM2S, 0) {loop = false} [{
        aie.use_lock(%lock71, Acquire, 0)
        aie.dma_bd_packet(4, 12)
        aie.dma_bd(%buf71 : memref<256xi32>, 0, 256)
        aie.use_lock(%lock71, Release, 1)
      }]
      aie.end
    }
    %buf62 = aie.buffer(%tile_0_2) {sym_name = "buf62"} : memref<512xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.dma(S2MM, 0) {loop = false} [{
        aie.use_lock(%lock_0_2, Acquire, 0)
        aie.dma_bd(%buf62 : memref<512xi32>, 0, 512)
        aie.use_lock(%lock_0_2, Release, 1)
      }]
      aie.end
    }
  }
}

