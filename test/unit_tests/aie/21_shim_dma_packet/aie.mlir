//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// A single kernel 32x32x32 GEMM using GMIO.
// This design illustrates a common technique using the packet switched
// interconnect, where 3 input tiles are needed, from 3 separate DMA
// sources.  In this case, we can use packet switching to share 2 sources
// on a single DMA.

module @kernel_gemm  {
  %3 = aie.tile(27, 0)
  %4 = aie.switchbox(%3)  {
    %43 = aie.amsel<0> (0)
    %44 = aie.masterset(West : 0, %43)
    aie.packet_rules(South : 0)  {
      aie.rule(31, 2, %43)
    }
  }
  %5 = aie.lock(%3, 0) { sym_name = "input_lock_0" }
  %6 = aie.external_buffer {sym_name = "buf0"} : memref<32x32xi32>

  %8 = aie.shim_dma(%3)  {
    %43 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%5, Acquire, 1)
    aie.dma_bd_packet(0, 2)
    aie.dma_bd(%6 : memref<32x32xi32>, 0, 1024)
    aie.use_lock(%5, Release, 0)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb0
    aie.end
  }
  %9 = aie.tile(26, 0)
  %10 = aie.switchbox(%9)  {
    %43 = aie.amsel<0> (0)
    %44 = aie.masterset(West : 0, %43)
    aie.packet_rules(South : 1)  {
      aie.rule(31, 3, %43)
    }
    aie.packet_rules(East : 0)  {
      aie.rule(31, 2, %43)
    }
    aie.connect<South : 3, North : 0>
    aie.connect<West : 0, South : 2>
  }
  %11 = aie.lock(%9, 2) { sym_name = "input_lock_2" }
  %12 = aie.lock(%9, 1) { sym_name = "input_lock_1" }
  %13 = aie.lock(%9, 0) { sym_name = "output_lock" }
  %14 = aie.external_buffer {sym_name = "buf1"} : memref<32x32xi32>
  %15 = aie.external_buffer {sym_name = "buf2"} : memref<32x32xi32>
  %16 = aie.external_buffer {sym_name = "buf3"} : memref<32x32xi32>

  %18 = aie.shim_dma(%9)  {
    %43 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%13, Acquire, 0)
    aie.dma_bd(%16 : memref<32x32xi32>, 0, 1024)
    aie.use_lock(%13, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb0
    %44 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    aie.use_lock(%11, Acquire, 1)
    aie.dma_bd(%15 : memref<32x32xi32>, 0, 1024)
    aie.use_lock(%11, Release, 0)
    aie.next_bd ^bb3
  ^bb4:  // pred: ^bb2
    %45 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    aie.use_lock(%12, Acquire, 1)
    aie.dma_bd_packet(0, 3)
    aie.dma_bd(%14 : memref<32x32xi32>, 0, 1024)
    aie.use_lock(%12, Release, 0)
    aie.next_bd ^bb5
  ^bb6:  // pred: ^bb4
    aie.end
  }
  %19 = aie.tile(25, 2) {polyaie.leaf}
  %20 = aie.lock(%19, 15)
  %21 = aie.switchbox(%19)  {
    %43 = aie.amsel<0> (0)
    %44 = aie.masterset(DMA : 1, %43)
    aie.packet_rules(South : 0)  {
      aie.rule(30, 2, %43)
    }
    aie.connect<East : 0, DMA : 0>
    aie.connect<DMA : 0, South : 0>
  }
  %22 = aie.lock(%19, 3)
  %23 = aie.lock(%19, 2)
  %24 = aie.lock(%19, 1)
  %25 = aie.lock(%19, 0)
  %26 = aie.buffer(%19) {sym_name = "C_out"} : memref<32x32xi32>
  %29 = aie.buffer(%19) {sym_name = "C"} : memref<32x32xi32>
  %30 = aie.buffer(%19) {sym_name = "A"} : memref<32x32xi32>
  %31 = aie.buffer(%19) {sym_name = "B"} : memref<32x32xi32>
  %32 = aie.core(%19)  {
    aie.use_lock(%23, Acquire, 1)
    aie.use_lock(%24, Acquire, 1)
    aie.use_lock(%25, Acquire, 0)
    aie.use_lock(%22, Acquire, 1)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %43 = affine.load %29[%arg0, %arg1] : memref<32x32xi32>
        affine.store %43, %26[%arg0, %arg1] : memref<32x32xi32>
        affine.for %arg2 = 0 to 32 {
          %44 = affine.load %26[%arg0, %arg1] : memref<32x32xi32>
          %45 = affine.load %31[%arg0, %arg2] : memref<32x32xi32>
          %46 = affine.load %30[%arg2, %arg1] : memref<32x32xi32>
          %47 = arith.muli %45, %46 : i32
          %48 = arith.addi %44, %47 : i32
          affine.store %48, %26[%arg0, %arg1] : memref<32x32xi32>
        }
      }
    }
    aie.use_lock(%22, Release, 0)
    aie.use_lock(%25, Release, 1)
    aie.use_lock(%24, Release, 0)
    aie.use_lock(%23, Release, 0)
    aie.use_lock(%20, Release, 1)
    aie.end
  }
  %33 = aie.mem(%19)  {
    %43 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    aie.use_lock(%22, Acquire, 0)
    aie.dma_bd(%31 : memref<32x32xi32>, 0, 1024)
    aie.use_lock(%22, Release, 1)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb0
    %44 = aie.dma_start(S2MM, 1, ^bb3, ^bb5)
  ^bb3:  // 2 preds: ^bb2, ^bb4
    aie.use_lock(%24, Acquire, 0)
    aie.dma_bd_packet(0, 2)
    aie.dma_bd(%29 : memref<32x32xi32>, 0, 1024)
    aie.use_lock(%24, Release, 1)
    aie.next_bd ^bb4
  ^bb4:  // pred: ^bb3
    aie.use_lock(%23, Acquire, 0)
    aie.dma_bd_packet(0, 3)
    aie.dma_bd(%30 : memref<32x32xi32>, 0, 1024)
    aie.use_lock(%23, Release, 1)
    aie.next_bd ^bb3
  ^bb5:  // pred: ^bb2
    %45 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    aie.use_lock(%25, Acquire, 1)
    aie.dma_bd(%26 : memref<32x32xi32>, 0, 1024)
    aie.use_lock(%25, Release, 0)
    aie.next_bd ^bb6
  ^bb7:  // pred: ^bb5
    aie.end
  }
  %34 = aie.tile(25, 0)
  %35 = aie.switchbox(%34)  {
    %43 = aie.amsel<0> (0)
    %44 = aie.masterset(North : 0, %43)
    aie.packet_rules(East : 0)  {
      aie.rule(30, 2, %43)
    }
    aie.connect<North : 0, East : 0>
  }
  %36 = aie.tile(25, 1)
  %37 = aie.switchbox(%36)  {
    %43 = aie.amsel<0> (0)
    %44 = aie.masterset(North : 0, %43)
    aie.packet_rules(South : 0)  {
      aie.rule(30, 2, %43)
    }
    aie.connect<North : 0, South : 0>
  }
  %38 = aie.tile(26, 1)
  %39 = aie.tile(26, 2)
  %40 = aie.switchbox(%38)  {
    aie.connect<South : 0, North : 0>
  }
  %41 = aie.switchbox(%39)  {
    aie.connect<South : 0, West : 0>
  }
  %42 = aie.shim_mux(%9)  {
    aie.connect<DMA : 0, North : 3>
    aie.connect<North : 2, DMA : 0>
  }
}
