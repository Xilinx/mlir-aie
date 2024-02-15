// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @test_chess_05_shim_dma_core_function {
  aie.device(ipu) {
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_0 = aie.tile(0, 0)
    %a_ping = aie.buffer(%tile_0_3) {sym_name = "a_ping"} : memref<16xi32>
    %a_pong = aie.buffer(%tile_0_3) {sym_name = "a_pong"} : memref<16xi32>
    %b_ping = aie.buffer(%tile_0_3) {sym_name = "b_ping"} : memref<16xi32>
    %b_pong = aie.buffer(%tile_0_3) {sym_name = "b_pong"} : memref<16xi32>
    %lock_0_3 = aie.lock(%tile_0_3, 3) {init = 2 : i32}
    %lock_0_3_0 = aie.lock(%tile_0_3, 4)
    %lock_0_3_1 = aie.lock(%tile_0_3, 5) {init = 2 : i32}
    %lock_0_3_2 = aie.lock(%tile_0_3, 6)
    %lock_0_3_3 = aie.lock(%tile_0_3, 7)
    // func.func private @func(memref<16xi32>, memref<16xi32>)
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_4 {
        aie.use_lock(%lock_0_3_0, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_1, AcquireGreaterEqual, 1)
        // func.call @func(%a_ping, %b_ping) : (memref<16xi32>, memref<16xi32>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
        aie.use_lock(%lock_0_3_2, Release, 1)
        aie.use_lock(%lock_0_3_0, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_1, AcquireGreaterEqual, 1)
        // func.call @func(%a_pong, %b_pong) : (memref<16xi32>, memref<16xi32>) -> ()
        aie.use_lock(%lock_0_3, Release, 1)
        aie.use_lock(%lock_0_3_2, Release, 1)
      }
      aie.end
    } // {link_with = "kernel.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb2, ^bb1)
    ^bb1:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
    ^bb2:  // 2 preds: ^bb0, ^bb3
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_ping : memref<16xi32>, 0, 16)
      aie.use_lock(%lock_0_3_0, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%a_pong : memref<16xi32>, 0, 16)
      aie.use_lock(%lock_0_3_0, Release, 1)
      aie.next_bd ^bb2
    ^bb4:  // 2 preds: ^bb1, ^bb5
      aie.use_lock(%lock_0_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_ping : memref<16xi32>, 0, 16)
      aie.use_lock(%lock_0_3_1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%b_pong : memref<16xi32>, 0, 16)
      aie.use_lock(%lock_0_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb1
      aie.end
    }
    %input_buffer = aie.external_buffer {sym_name = "input_buffer"} : memref<32xi32>
    %output_buffer = aie.external_buffer {sym_name = "output_buffer"} : memref<32xi32>
    %input_lock_write = aie.lock(%tile_0_0, 1) {init = 1 : i32, sym_name = "input_lock_write"}
    %input_lock_read = aie.lock(%tile_0_0, 2) {sym_name = "input_lock_read"}
    %output_lock_write = aie.lock(%tile_0_0, 3) {init = 1 : i32, sym_name = "output_lock_write"}
    %output_lock_read = aie.lock(%tile_0_0, 4) {sym_name = "output_lock_read"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_3, DMA : 1, %tile_0_0, DMA : 0)
    %shim_dma_0_0 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb2, ^bb1)
    ^bb1:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb3, ^bb4)
    ^bb2:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%input_lock_read, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_buffer : memref<32xi32>, 0, 32)
      aie.use_lock(%input_lock_write, Release, 1)
      aie.next_bd ^bb2
    ^bb3:  // 2 preds: ^bb1, ^bb3
      aie.use_lock(%output_lock_write, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_buffer : memref<32xi32>, 0, 32)
      aie.use_lock(%output_lock_read, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb1
      aie.end
    }
  }
}

