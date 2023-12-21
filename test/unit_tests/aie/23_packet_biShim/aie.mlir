//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf

module @aie_module  {
  %t70 = AIE.tile(7, 0)
  %t72 = AIE.tile(7, 2)

  %10 = AIE.lock(%t72, 1) {sym_name = "inter_lock"}
  %lock1 = AIE.lock(%t70, 1) {sym_name = "input_lock"}
  %lock2 = AIE.lock(%t70, 2) {sym_name = "output_lock"}

  %11 = AIE.buffer(%t72) {sym_name = "buf1"} : memref<256xi32>
  %buf_i = AIE.external_buffer {sym_name = "input"} : memref<256xi32>
  %buf_o = AIE.external_buffer {sym_name = "output"} : memref<257xi32>

  %12 = AIE.mem(%t72)  {
    %srcDma = AIE.dma_start("S2MM", 0, ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dma_start("MM2S", 0, ^bb3, ^end)
  ^bb2:
    AIE.use_lock(%10, Acquire, 0)
    AIE.dma_bd(%11 : memref<256xi32>, 0, 256)
    AIE.use_lock(%10, Release, 1)
    AIE.next_bd ^bb2
  ^bb3:
    AIE.use_lock(%10, Acquire, 1)
    AIE.dma_bd_packet(0x6, 10)
    AIE.dma_bd(%11 : memref<256xi32>, 0, 256)
    AIE.next_bd ^bb3
  ^end:
    AIE.end
  }

  %dma = AIE.shim_dma(%t70)  {
    AIE.dma_start("MM2S", 0, ^bb0, ^dma0)
  ^dma0:
    AIE.dma_start("S2MM", 0, ^bb1, ^end)
  ^bb0:
    AIE.use_lock(%lock1, Acquire, 1)
    AIE.dma_bd_packet(0x2, 3)
    AIE.dma_bd(%buf_i : memref<256xi32>, 0, 256)
    AIE.use_lock(%lock1, Release, 0)
    AIE.next_bd ^bb0
  ^bb1:
    AIE.use_lock(%lock2, Acquire, 0)
    AIE.dma_bd(%buf_o : memref<257xi32>, 0, 257)
    AIE.use_lock(%lock2, Release, 1)
    AIE.next_bd ^bb1
  ^end:
    AIE.end
  }

  AIE.packet_flow(0x3) {
    AIE.packet_source<%t70, DMA : 0>
    AIE.packet_dest<%t72, DMA : 0>
  }

  AIE.packet_flow(0xA) {
    AIE.packet_source<%t72, DMA : 0>
    AIE.packet_dest<%t70, DMA : 0>
  }
}
