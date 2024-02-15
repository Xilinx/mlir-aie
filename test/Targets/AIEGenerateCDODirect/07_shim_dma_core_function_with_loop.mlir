// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

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
      %0 = aie.dma_start(S2MM, 0, ^bb2, ^bb1)
    ^bb1:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb2:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%lock_0_3, Acquire, 0)
      aie.dma_bd(%a_ping : memref<64xi32>, 0, 64)
      aie.use_lock(%lock_0_3, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%lock_0_3_0, Acquire, 0)
      aie.dma_bd(%a_pong : memref<64xi32>, 0, 64)
      aie.use_lock(%lock_0_3_0, Release, 1)
      aie.next_bd ^bb2
    ^bb4:  // 2 preds: ^bb1, ^bb5
      aie.use_lock(%lock_0_3_1, Acquire, 1)
      aie.dma_bd(%b_ping : memref<64xi32>, 0, 64)
      aie.use_lock(%lock_0_3_1, Release, 0)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_3_2, Acquire, 1)
      aie.dma_bd(%b_pong : memref<64xi32>, 0, 64)
      aie.use_lock(%lock_0_3_2, Release, 0)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb1
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
    %shim_dma_0_0 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb2, ^bb1)
    ^bb1:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb2:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%input_lock, Acquire, 1)
      aie.dma_bd(%input_buffer : memref<512xi32>, 0, 512)
      aie.use_lock(%input_lock, Release, 0)
      aie.next_bd ^bb2
    ^bb3:  // 2 preds: ^bb1, ^bb3
      aie.use_lock(%output_lock, Acquire, 1)
      aie.dma_bd(%output_buffer : memref<512xi32>, 0, 512)
      aie.use_lock(%output_lock, Release, 0)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb1
      aie.end
    }
  }
}

