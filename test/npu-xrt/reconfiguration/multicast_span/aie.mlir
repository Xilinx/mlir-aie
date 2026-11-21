// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each core waits for input before reading its bias. The host output depends
// on delivery of the shared control-packet bias update to all four cores.

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%in : memref<256xi32>, %out : memref<256xi32>) {
      aiex.configure @cfg {
        aiex.run @run(%in, %out) : (memref<256xi32>, memref<256xi32>)
      }
    }
  }

  aie.device(npu2) @cfg {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 4)
    %c = aie.tile(1, 2)
    %d = aie.tile(1, 4)
    %bias_a = aie.buffer(%a) {address = 16384 : i32, sym_name = "bias_a"} : memref<1xi32> = dense<0>
    %bias_b = aie.buffer(%b) {address = 16384 : i32, sym_name = "bias_b"} : memref<1xi32> = dense<0>
    %bias_c = aie.buffer(%c) {address = 16384 : i32, sym_name = "bias_c"} : memref<1xi32> = dense<0>
    %bias_d = aie.buffer(%d) {address = 16384 : i32, sym_name = "bias_d"} : memref<1xi32> = dense<0>
    aie.objectfifo @in(%s0, {%a}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @ab(%a, {%b}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @bc(%b, {%c}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @cd(%c, {%d}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @out(%d, {%s1}, 1 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.core(%a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %bin = aie.objectfifo.acquire @in(Consume, 1) : memref<256xi32>
        %bout = aie.objectfifo.acquire @ab(Produce, 1) : memref<256xi32>
        %one = memref.load %bias_a[%c0] : memref<1xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %bin[%k] : memref<256xi32>
          %r = arith.addi %v, %one : i32
          memref.store %r, %bout[%k] : memref<256xi32>
        }
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @ab(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %bin = aie.objectfifo.acquire @ab(Consume, 1) : memref<256xi32>
        %bout = aie.objectfifo.acquire @bc(Produce, 1) : memref<256xi32>
        %one = memref.load %bias_b[%c0] : memref<1xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %bin[%k] : memref<256xi32>
          %r = arith.addi %v, %one : i32
          memref.store %r, %bout[%k] : memref<256xi32>
        }
        aie.objectfifo.release @ab(Consume, 1)
        aie.objectfifo.release @bc(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%c) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %bin = aie.objectfifo.acquire @bc(Consume, 1) : memref<256xi32>
        %bout = aie.objectfifo.acquire @cd(Produce, 1) : memref<256xi32>
        %one = memref.load %bias_c[%c0] : memref<1xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %bin[%k] : memref<256xi32>
          %r = arith.addi %v, %one : i32
          memref.store %r, %bout[%k] : memref<256xi32>
        }
        aie.objectfifo.release @bc(Consume, 1)
        aie.objectfifo.release @cd(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%d) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cn = arith.constant 256 : index
      %true = arith.constant true
      scf.while : () -> () {
        scf.condition(%true)
      } do {
        %bin = aie.objectfifo.acquire @cd(Consume, 1) : memref<256xi32>
        %bout = aie.objectfifo.acquire @out(Produce, 1) : memref<256xi32>
        %one = memref.load %bias_d[%c0] : memref<1xi32>
        scf.for %k = %c0 to %cn step %c1 {
          %v = memref.load %bin[%k] : memref<256xi32>
          %r = arith.addi %v, %one : i32
          memref.store %r, %bout[%k] : memref<256xi32>
        }
        aie.objectfifo.release @cd(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<256xi32>, %y : memref<256xi32>) {
      aiex.control_packet_barrier
      aiex.control_packet {address = 2113536 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 4210688 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 35667968 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet {address = 37765120 : ui32, data = array<i32: 1>, opcode = 0 : i32, stream_id = 0 : i32}
      aiex.control_packet_barrier
      %t_in = aiex.dma_configure_task_for @in {
        aie.dma_bd(%x : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @out {
        aie.dma_bd(%y : memref<256xi32> offset = 0 len = 256)
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
