//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// A single kernel 32x32x32 GEMM using GMIO.
// This design illustrates a common technique using the packet switched
// interconnect, where 3 input tiles are needed, from 3 separate DMA
// sources.  In this case, we can use packet switching to share 2 sources
// on a single DMA.

module @kernel_gemm  {
  %3 = AIE.tile(27, 0)
  %4 = AIE.switchbox(%3)  {
    %43 = AIE.amsel<0> (0)
    %44 = AIE.masterset(West : 0, %43)
    AIE.packetrules(DMA : 0)  {
      AIE.rule(31, 2, %43)
    }
  }
  %5 = AIE.lock(%3, 0) { sym_name = "input_lock_0" }
  %6 = AIE.external_buffer {sym_name = "buf0"} : memref<32x32xi32>

  %8 = AIE.shimDMA(%3)  {
    %43 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%5, Acquire, 1)
    AIE.dmaBdPacket(0, 2)
    AIE.dmaBd(<%6 : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%5, Release, 0)
    AIE.nextBd ^bb1
  ^bb2:  // pred: ^bb0
    AIE.end
  }
  %9 = AIE.tile(26, 0)
  %10 = AIE.switchbox(%9)  {
    %43 = AIE.amsel<0> (0)
    %44 = AIE.masterset(West : 0, %43)
    AIE.packetrules(DMA : 1)  {
      AIE.rule(31, 3, %43)
    }
    AIE.packetrules(East : 0)  {
      AIE.rule(31, 2, %43)
    }
    AIE.connect<South : 3, North : 0>
    AIE.connect<West : 0, South : 2>
  }
  %11 = AIE.lock(%9, 2) { sym_name = "input_lock_2" }
  %12 = AIE.lock(%9, 1) { sym_name = "input_lock_1" }
  %13 = AIE.lock(%9, 0) { sym_name = "output_lock" }
  %14 = AIE.external_buffer {sym_name = "buf1"} : memref<32x32xi32>
  %15 = AIE.external_buffer {sym_name = "buf2"} : memref<32x32xi32>
  %16 = AIE.external_buffer {sym_name = "buf3"} : memref<32x32xi32>

  %18 = AIE.shimDMA(%9)  {
    %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%13, Acquire, 0)
    AIE.dmaBd(<%16 : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%13, Release, 1)
    AIE.nextBd ^bb1
  ^bb2:  // pred: ^bb0
    %44 = AIE.dmaStart(MM2S, 0, ^bb3, ^bb4)
  ^bb3:  // 2 preds: ^bb2, ^bb3
    AIE.useLock(%11, Acquire, 1)
    AIE.dmaBd(<%15 : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%11, Release, 0)
    AIE.nextBd ^bb3
  ^bb4:  // pred: ^bb2
    %45 = AIE.dmaStart(MM2S, 1, ^bb5, ^bb6)
  ^bb5:  // 2 preds: ^bb4, ^bb5
    AIE.useLock(%12, Acquire, 1)
    AIE.dmaBdPacket(0, 3)
    AIE.dmaBd(<%14 : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%12, Release, 0)
    AIE.nextBd ^bb5
  ^bb6:  // pred: ^bb4
    AIE.end
  }
  %19 = AIE.tile(25, 2) {polyaie.leaf}
  %20 = AIE.lock(%19, 15)
  %21 = AIE.switchbox(%19)  {
    %43 = AIE.amsel<0> (0)
    %44 = AIE.masterset(DMA : 1, %43)
    AIE.packetrules(South : 0)  {
      AIE.rule(30, 2, %43)
    }
    AIE.connect<East : 0, DMA : 0>
    AIE.connect<DMA : 0, South : 0>
  }
  %22 = AIE.lock(%19, 3)
  %23 = AIE.lock(%19, 2)
  %24 = AIE.lock(%19, 1)
  %25 = AIE.lock(%19, 0)
  %26 = AIE.buffer(%19) {sym_name = "C_out"} : memref<32x32xi32>
  %29 = AIE.buffer(%19) {sym_name = "C"} : memref<32x32xi32>
  %30 = AIE.buffer(%19) {sym_name = "A"} : memref<32x32xi32>
  %31 = AIE.buffer(%19) {sym_name = "B"} : memref<32x32xi32>
  %32 = AIE.core(%19)  {
    AIE.useLock(%23, Acquire, 1)
    AIE.useLock(%24, Acquire, 1)
    AIE.useLock(%25, Acquire, 0)
    AIE.useLock(%22, Acquire, 1)
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
    AIE.useLock(%22, Release, 0)
    AIE.useLock(%25, Release, 1)
    AIE.useLock(%24, Release, 0)
    AIE.useLock(%23, Release, 0)
    AIE.useLock(%20, Release, 1)
    AIE.end
  }
  %33 = AIE.mem(%19)  {
    %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%22, Acquire, 0)
    AIE.dmaBd(<%31 : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%22, Release, 1)
    AIE.nextBd ^bb1
  ^bb2:  // pred: ^bb0
    %44 = AIE.dmaStart(S2MM, 1, ^bb3, ^bb5)
  ^bb3:  // 2 preds: ^bb2, ^bb4
    AIE.useLock(%24, Acquire, 0)
    AIE.dmaBdPacket(0, 2)
    AIE.dmaBd(<%29 : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%24, Release, 1)
    AIE.nextBd ^bb4
  ^bb4:  // pred: ^bb3
    AIE.useLock(%23, Acquire, 0)
    AIE.dmaBdPacket(0, 3)
    AIE.dmaBd(<%30 : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%23, Release, 1)
    AIE.nextBd ^bb3
  ^bb5:  // pred: ^bb2
    %45 = AIE.dmaStart(MM2S, 0, ^bb6, ^bb7)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    AIE.useLock(%25, Acquire, 1)
    AIE.dmaBd(<%26 : memref<32x32xi32>, 0, 1024>, 0)
    AIE.useLock(%25, Release, 0)
    AIE.nextBd ^bb6
  ^bb7:  // pred: ^bb5
    AIE.end
  }
  %34 = AIE.tile(25, 0)
  %35 = AIE.switchbox(%34)  {
    %43 = AIE.amsel<0> (0)
    %44 = AIE.masterset(North : 0, %43)
    AIE.packetrules(East : 0)  {
      AIE.rule(30, 2, %43)
    }
    AIE.connect<North : 0, East : 0>
  }
  %36 = AIE.tile(25, 1)
  %37 = AIE.switchbox(%36)  {
    %43 = AIE.amsel<0> (0)
    %44 = AIE.masterset(North : 0, %43)
    AIE.packetrules(South : 0)  {
      AIE.rule(30, 2, %43)
    }
    AIE.connect<North : 0, South : 0>
  }
  %38 = AIE.tile(26, 1)
  %39 = AIE.tile(26, 2)
  %40 = AIE.switchbox(%38)  {
    AIE.connect<South : 0, North : 0>
  }
  %41 = AIE.switchbox(%39)  {
    AIE.connect<South : 0, West : 0>
  }
  %42 = AIE.shimmux(%9)  {
    AIE.connect<DMA : 0, North : 3>
    AIE.connect<North : 2, DMA : 0>
  }
}
