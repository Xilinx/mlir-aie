// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each core waits for input before reading its bias. Both biases start at zero,
// so the host output depends on delivery of the control-packet update.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%in : memref<256xi32>, %out : memref<256xi32>) {
      aiex.configure @cfg {
        aiex.run @run(%in, %out) : (memref<256xi32>, memref<256xi32>)
      }
    }
  }

  aie.device(npu2) @cfg {
    %shim = aie.tile(0, 0)
    %c2 = aie.tile(0, 2)
    %c4 = aie.tile(0, 4)
    %bias2 = aie.buffer(%c2) {address = 16384 : i32, sym_name = "bias2"} : memref<1xi32> = dense<0>
    %bias4 = aie.buffer(%c4) {address = 16384 : i32, sym_name = "bias4"} : memref<1xi32> = dense<0>
    aie.objectfifo @in(%shim, {%c2}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @mid(%c2, {%c4}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out(%c4, {%shim}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.core(%c2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %bin = aie.objectfifo.acquire @in(Consume, 1) : memref<256xi32>
        %bout = aie.objectfifo.acquire @mid(Produce, 1) : memref<256xi32>
        %one = memref.load %bias2[%c0] : memref<1xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %bin[%k] : memref<256xi32>
          %r = arith.addi %v, %one : i32
          memref.store %r, %bout[%k] : memref<256xi32>
        }
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @mid(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%c4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %bin = aie.objectfifo.acquire @mid(Consume, 1) : memref<256xi32>
        %bout = aie.objectfifo.acquire @out(Produce, 1) : memref<256xi32>
        %one = memref.load %bias4[%c0] : memref<1xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %bin[%k] : memref<256xi32>
          %r = arith.addi %v, %one : i32
          memref.store %r, %bout[%k] : memref<256xi32>
        }
        aie.objectfifo.release @mid(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @run(%a : memref<256xi32>, %b : memref<256xi32>) {
      aiex.control_packet_barrier
      aiex.control_packet {address = 2113536 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 4210688 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%b : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in)
      aiex.dma_free_task(%t_out)
    }
  }
}
