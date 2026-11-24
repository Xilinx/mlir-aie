//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The core waits on lock 15 until a control packet sets it, so the shim's
// input task is still stalled mid-transfer (four objects into a fifo two
// deep) when the control packet's DMA programs a shim BD. That BD must not
// be the one the input task runs on. The core adds 1 to each element.
module {
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @sequence(%in : memref<64xi32>, %out : memref<64xi32>) {
      aiex.configure @design {
        aiex.run @run(%in, %out) : (memref<64xi32>, memref<64xi32>)
      }
    }
  }
  aie.device(npu2_1col) @design {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %gate = aie.lock(%t02, 15) {init = 0 : i32, sym_name = "gate"}
    aie.objectfifo @in(%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%t02, {%t00}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %one = arith.constant 1 : i32
      aie.use_lock(%gate, AcquireGreaterEqual, %one)
      scf.for %k = %c0 to %c4 step %c1 {
        %bi = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %bo = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        scf.for %i = %c0 to %c16 step %c1 {
          %x = memref.load %bi[%i] : memref<16xi32>
          %y = arith.addi %x, %one : i32
          memref.store %y, %bo[%i] : memref<16xi32>
        }
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence @run(%in : memref<64xi32>, %out : memref<64xi32>) {
      %ti = aiex.dma_configure_task_for @in {
        aie.dma_bd(%in : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      %to = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out : memref<64xi32> offset = 0 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti)
      aiex.dma_start_task(%to)
      // Tile (0, 2), lock 15: set to 1.
      aiex.control_packet {address = 0x21f0f0 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.dma_await_task(%to)
      aiex.dma_await_task(%ti)
    }
  }
}
