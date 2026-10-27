//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

// A single kernel 32x32x32 GEMM using GMIO.
// This design illustrates a common technique using the packet switched
// interconnect, where 3 input tiles are needed, from 3 separate DMA
// sources.  In this case, we can use packet switching to share 2 sources
// on a single DMA.

module @kernel_gemm  {
aie.device(xcvc1902) {
  %3 = aie.tile(27, 0)
  %4 = aie.switchbox(%3)  {
    %43 = aie.amsel<0> (0)
    %44 = aie.masterset(West : 0, %43)
    aie.packet_rules(South : 0)  {
      aie.rule(31, 2, %43)
    }
  }
  aie.lock @input_lock_0(%3, 0)
  aie.external_buffer @buf0 : memref<32x32xi32>

  %8 = aie.shim_dma(%3)  {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %43 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(@input_lock_0, Acquire, %c1_ul1)
    aie.dma_bd_packet(0, 2)
    %buffer_6_0 = aie.get_buffer @buf0 : memref<32x32xi32>
    aie.dma_bd(%buffer_6_0 : memref<32x32xi32> offset = 0 len = 1024)
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(@input_lock_0, Release, %c0_ul2)
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
  aie.lock @input_lock_2(%9, 2)
  aie.lock @input_lock_1(%9, 1)
  aie.lock @output_lock(%9, 0)
  aie.external_buffer @buf1 : memref<32x32xi32>
  aie.external_buffer @buf2 : memref<32x32xi32>
  aie.external_buffer @buf3 : memref<32x32xi32>

  %18 = aie.shim_dma(%9)  {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %43 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %c0_ul3 = arith.constant 0 : i32
    aie.use_lock(@output_lock, Acquire, %c0_ul3)
    %buffer_16_0 = aie.get_buffer @buf3 : memref<32x32xi32>
    aie.dma_bd(%buffer_16_0 : memref<32x32xi32> offset = 0 len = 1024)
    %c1_ul4 = arith.constant 1 : i32
    aie.use_lock(@output_lock, Release, %c1_ul4)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb0
    %44 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    %c1_ul5 = arith.constant 1 : i32
    aie.use_lock(@input_lock_2, Acquire, %c1_ul5)
    %buffer_15_0 = aie.get_buffer @buf2 : memref<32x32xi32>
    aie.dma_bd(%buffer_15_0 : memref<32x32xi32> offset = 0 len = 1024)
    %c0_ul6 = arith.constant 0 : i32
    aie.use_lock(@input_lock_2, Release, %c0_ul6)
    aie.next_bd ^bb3
  ^bb4:  // pred: ^bb2
    %45 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    %c1_ul7 = arith.constant 1 : i32
    aie.use_lock(@input_lock_1, Acquire, %c1_ul7)
    aie.dma_bd_packet(0, 3)
    %buffer_14_0 = aie.get_buffer @buf1 : memref<32x32xi32>
    aie.dma_bd(%buffer_14_0 : memref<32x32xi32> offset = 0 len = 1024)
    %c0_ul8 = arith.constant 0 : i32
    aie.use_lock(@input_lock_1, Release, %c0_ul8)
    aie.next_bd ^bb5
  ^bb6:  // pred: ^bb4
    aie.end
  }
  %19 = aie.tile(25, 2) {polyaie.leaf}
  aie.lock @lock_20(%19, 15)
  %21 = aie.switchbox(%19)  {
    %43 = aie.amsel<0> (0)
    %44 = aie.masterset(DMA : 1, %43)
    aie.packet_rules(South : 0)  {
      aie.rule(30, 2, %43)
    }
    aie.connect<East : 0, DMA : 0>
    aie.connect<DMA : 0, South : 0>
  }
  aie.lock @lock_22(%19, 3)
  aie.lock @lock_23(%19, 2)
  aie.lock @lock_24(%19, 1)
  aie.lock @lock_25(%19, 0)
  aie.buffer @C_out(%19) : memref<32x32xi32>
  aie.buffer @C(%19) : memref<32x32xi32>
  aie.buffer @A(%19) : memref<32x32xi32>
  aie.buffer @B(%19) : memref<32x32xi32>
  %32 = aie.core(%19)  {
    %c1_ul9 = arith.constant 1 : i32
    aie.use_lock(@lock_23, Acquire, %c1_ul9)
    %c1_ul10 = arith.constant 1 : i32
    aie.use_lock(@lock_24, Acquire, %c1_ul10)
    %c0_ul11 = arith.constant 0 : i32
    aie.use_lock(@lock_25, Acquire, %c0_ul11)
    %c1_ul12 = arith.constant 1 : i32
    aie.use_lock(@lock_22, Acquire, %c1_ul12)
    affine.for %arg0 = 0 to 32 {
      affine.for %arg1 = 0 to 32 {
        %buffer_29_0 = aie.get_buffer @C : memref<32x32xi32>
        %43 = affine.load %buffer_29_0[%arg0, %arg1] : memref<32x32xi32>
        %buffer_26_0 = aie.get_buffer @C_out : memref<32x32xi32>
        affine.store %43, %buffer_26_0[%arg0, %arg1] : memref<32x32xi32>
        affine.for %arg2 = 0 to 32 {
          %buffer_26_1 = aie.get_buffer @C_out : memref<32x32xi32>
          %44 = affine.load %buffer_26_1[%arg0, %arg1] : memref<32x32xi32>
          %buffer_31_0 = aie.get_buffer @B : memref<32x32xi32>
          %45 = affine.load %buffer_31_0[%arg0, %arg2] : memref<32x32xi32>
          %buffer_30_0 = aie.get_buffer @A : memref<32x32xi32>
          %46 = affine.load %buffer_30_0[%arg2, %arg1] : memref<32x32xi32>
          %47 = arith.muli %45, %46 : i32
          %48 = arith.addi %44, %47 : i32
          affine.store %48, %buffer_26_1[%arg0, %arg1] : memref<32x32xi32>
        }
      }
    }
    %c0_ul13 = arith.constant 0 : i32
    aie.use_lock(@lock_22, Release, %c0_ul13)
    %c1_ul14 = arith.constant 1 : i32
    aie.use_lock(@lock_25, Release, %c1_ul14)
    %c0_ul15 = arith.constant 0 : i32
    aie.use_lock(@lock_24, Release, %c0_ul15)
    %c0_ul16 = arith.constant 0 : i32
    aie.use_lock(@lock_23, Release, %c0_ul16)
    %c1_ul17 = arith.constant 1 : i32
    aie.use_lock(@lock_20, Release, %c1_ul17)
    aie.end
  }
  %33 = aie.mem(%19)  {
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %43 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %c0_ul18 = arith.constant 0 : i32
    aie.use_lock(@lock_22, Acquire, %c0_ul18)
    %buffer_31_1 = aie.get_buffer @B : memref<32x32xi32>
    aie.dma_bd(%buffer_31_1 : memref<32x32xi32> offset = 0 len = 1024)
    %c1_ul19 = arith.constant 1 : i32
    aie.use_lock(@lock_22, Release, %c1_ul19)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb0
    %44 = aie.dma_start(S2MM, 1, ^bb3, ^bb5)
  ^bb3:  // 2 preds: ^bb2, ^bb4
    %c0_ul20 = arith.constant 0 : i32
    aie.use_lock(@lock_24, Acquire, %c0_ul20)
    aie.dma_bd_packet(0, 2)
    %buffer_29_1 = aie.get_buffer @C : memref<32x32xi32>
    aie.dma_bd(%buffer_29_1 : memref<32x32xi32> offset = 0 len = 1024)
    %c1_ul21 = arith.constant 1 : i32
    aie.use_lock(@lock_24, Release, %c1_ul21)
    aie.next_bd ^bb4
  ^bb4:  // pred: ^bb3
    %c0_ul22 = arith.constant 0 : i32
    aie.use_lock(@lock_23, Acquire, %c0_ul22)
    aie.dma_bd_packet(0, 3)
    %buffer_30_1 = aie.get_buffer @A : memref<32x32xi32>
    aie.dma_bd(%buffer_30_1 : memref<32x32xi32> offset = 0 len = 1024)
    %c1_ul23 = arith.constant 1 : i32
    aie.use_lock(@lock_23, Release, %c1_ul23)
    aie.next_bd ^bb3
  ^bb5:  // pred: ^bb2
    %45 = aie.dma_start(MM2S, 0, ^bb6, ^bb7)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    %c1_ul24 = arith.constant 1 : i32
    aie.use_lock(@lock_25, Acquire, %c1_ul24)
    %buffer_26_2 = aie.get_buffer @C_out : memref<32x32xi32>
    aie.dma_bd(%buffer_26_2 : memref<32x32xi32> offset = 0 len = 1024)
    %c0_ul25 = arith.constant 0 : i32
    aie.use_lock(@lock_25, Release, %c0_ul25)
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
}
