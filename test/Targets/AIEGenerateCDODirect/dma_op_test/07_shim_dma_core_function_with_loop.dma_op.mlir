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
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_elfs.bin $BASENAME.dma_start.prj/aie_cdo_elfs.bin
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_enable.bin $BASENAME.dma_start.prj/aie_cdo_enable.bin

module @test_chess_04_deprecated_shim_dma_precompiled_kernel {
  aie.device(ipu) {
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_0 = aie.tile(0, 0)
    %a_ping = aie.buffer(%tile_0_3) {sym_name = "a_ping"} : memref<64xi32>
    %a_pong = aie.buffer(%tile_0_3) {sym_name = "a_pong"} : memref<64xi32>
    %b_ping = aie.buffer(%tile_0_3) {sym_name = "b_ping"} : memref<64xi32>
    %b_pong = aie.buffer(%tile_0_3) {sym_name = "b_pong"} : memref<64xi32>
    %lock_0_3 = aie.lock(%tile_0_3, 3)
    %lock_0_3_0 = aie.lock(%tile_0_3, 4)
    %lock_0_3_1 = aie.lock(%tile_0_3, 5)
    %lock_0_3_2 = aie.lock(%tile_0_3, 6)
    // func.func private @func(memref<64xi32>, memref<64xi32>, i32)
    %core_0_3 = aie.core(%tile_0_3) {
      %c64_i32 = arith.constant 64 : i32
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_3 = arith.constant 0 : index
      %c1_4 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        aie.use_lock(%lock_0_3, Acquire, 1)
        aie.use_lock(%lock_0_3_1, Acquire, 0)
        // func.call @func(%a_ping, %b_ping, %c64_i32) : (memref<64xi32>, memref<64xi32>, i32) -> ()
        aie.use_lock(%lock_0_3, Release, 0)
        aie.use_lock(%lock_0_3_1, Release, 1)
        aie.use_lock(%lock_0_3_0, Acquire, 1)
        aie.use_lock(%lock_0_3_2, Acquire, 0)
        // func.call @func(%a_pong, %b_pong, %c64_i32) : (memref<64xi32>, memref<64xi32>, i32) -> ()
        aie.use_lock(%lock_0_3_0, Release, 0)
        aie.use_lock(%lock_0_3_2, Release, 1)
      }
      aie.end
    } // {link_with = "kernel.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_3, Acquire, 0)
        aie.dma_bd(%a_ping : memref<64xi32>, 0, 64)
        aie.use_lock(%lock_0_3, Release, 1)
      }, {
        aie.use_lock(%lock_0_3_0, Acquire, 0)
        aie.dma_bd(%a_pong : memref<64xi32>, 0, 64)
        aie.use_lock(%lock_0_3_0, Release, 1)
      }]
      aie.dma(MM2S, 1) [{
        aie.use_lock(%lock_0_3_1, Acquire, 1)
        aie.dma_bd(%b_ping : memref<64xi32>, 0, 64)
        aie.use_lock(%lock_0_3_1, Release, 0)
      }, {
        aie.use_lock(%lock_0_3_2, Acquire, 1)
        aie.dma_bd(%b_pong : memref<64xi32>, 0, 64)
        aie.use_lock(%lock_0_3_2, Release, 0)
      }]
      aie.end
    }
    %input_buffer = aie.external_buffer {sym_name = "input_buffer"} : memref<512xi32>
    %output_buffer = aie.external_buffer {sym_name = "output_buffer"} : memref<512xi32>
    %input_lock = aie.lock(%tile_0_0, 1) {sym_name = "input_lock"}
    %output_lock = aie.lock(%tile_0_0, 2) {sym_name = "output_lock"}
    aie.flow(%tile_0_1, South : 0, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_3, DMA : 1, %tile_0_1, South : 0)
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<North : 0, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
  }
}

