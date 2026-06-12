// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: mkdir -p test
// RUN: cd test
// RUN: make -f %S/Makefile clean
// RUN: make -f %S/Makefile 
// RUN: make -f %S/Makefile diff

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @infactor(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1xi32>> 
    aie.objectfifo @in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo @out(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    func.func private @vector_scalar_mul_vector(memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) attributes {link_with = "scale.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @infactor(Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
        %c0_0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
          %2 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xi16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xi16>> -> memref<1024xi16>
          %4 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xi16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi16>> -> memref<1024xi16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @vector_scalar_mul_vector(%3, %5, %1, %c1024_i32) : (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @infactor(Consume, 1)
      }
      aie.end
    }
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.runtime_sequence @sequence(%arg0: memref<4096xi16>, %arg1: memref<1xi32>, %arg2: memref<4096xi16>) {
      %w32_addr = arith.constant 213200 : i32
      %w32_val = arith.constant 2038038528 : i32
      aiex.npu.write32(%w32_addr, %w32_val) {column = 0 : i32, row = 2 : i32} : i32, i32
      %w32_addr_1 = arith.constant 213204 : i32
      %w32_val_1 = arith.constant 1 : i32
      aiex.npu.write32(%w32_addr_1, %w32_val_1) {column = 0 : i32, row = 2 : i32} : i32, i32
      %w32_addr_2 = arith.constant 213216 : i32
      %w32_val_2 = arith.constant 1260724769 : i32
      aiex.npu.write32(%w32_addr_2, %w32_val_2) {column = 0 : i32, row = 2 : i32} : i32, i32
      %w32_addr_3 = arith.constant 213220 : i32
      %w32_val_3 = arith.constant 439168079 : i32
      aiex.npu.write32(%w32_addr_3, %w32_val_3) {column = 0 : i32, row = 2 : i32} : i32, i32
      %w32_addr_4 = arith.constant 261888 : i32
      %w32_val_4 = arith.constant 289 : i32
      aiex.npu.write32(%w32_addr_4, %w32_val_4) {column = 0 : i32, row = 2 : i32} : i32, i32
      %w32_addr_5 = arith.constant 261892 : i32
      %w32_val_5 = arith.constant 0 : i32
      aiex.npu.write32(%w32_addr_5, %w32_val_5) {column = 0 : i32, row = 2 : i32} : i32, i32
      %w32_addr_6 = arith.constant 212992 : i32
      %w32_val_6 = arith.constant 31232 : i32
      aiex.npu.write32(%w32_addr_6, %w32_val_6) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %ap_arg_plus = arith.constant 0 : i32
      aiex.npu.address_patch(%ap_arg_plus : i32) {addr = 119268 : ui32, arg_idx = 4 : i32}
      %w32_addr_7 = arith.constant 119308 : i32
      %w32_val_7 = arith.constant 15 : i32
      aiex.npu.write32(%w32_addr_7, %w32_val_7) {column = 0 : i32, row = 0 : i32} : i32, i32
      %w32_addr_8 = arith.constant 212992 : i32
      %w32_val_8 = arith.constant 32512 : i32
      aiex.npu.write32(%w32_addr_8, %w32_val_8) {column = 0 : i32, row = 0 : i32} : i32, i32
      %w32_addr_9 = arith.constant 213068 : i32
      %w32_val_9 = arith.constant 127 : i32
      aiex.npu.write32(%w32_addr_9, %w32_val_9) {column = 0 : i32, row = 0 : i32} : i32, i32
      %w32_addr_10 = arith.constant 213000 : i32
      %w32_val_10 = arith.constant 127 : i32
      aiex.npu.write32(%w32_addr_10, %w32_val_10) {column = 0 : i32, row = 0 : i32} : i32, i32
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      %w32_addr_11 = arith.constant 213064 : i32
      %w32_val_11 = arith.constant 126 : i32
      aiex.npu.write32(%w32_addr_11, %w32_val_11) {column = 0 : i32, row = 0 : i32} : i32, i32
      %w32_addr_12 = arith.constant 213000 : i32
      %w32_val_12 = arith.constant 126 : i32
      aiex.npu.write32(%w32_addr_12, %w32_val_12) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

