// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %python %S/../../../python/utils/trace/parse.py --input %S/trace_test1.txt --mlir %s --output %t.json
// RUN: %python %S/../../../python/utils/trace/get_trace_summary.py --input %t.json
// RUN: %python %S/../Inputs/check_golden.py --actual %t.json --expected %S/golden_json.txt

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
      %cst_npu_0 = arith.constant 213200 : i32
      %cst_npu_1 = arith.constant 2038038528 : i32
      aiex.npu.write32(%cst_npu_0, %cst_npu_1) {column = 0 : i32, row = 2 : i32} : i32, i32
      %cst_npu_2 = arith.constant 213204 : i32
      %cst_npu_3 = arith.constant 1 : i32
      aiex.npu.write32(%cst_npu_2, %cst_npu_3) {column = 0 : i32, row = 2 : i32} : i32, i32
      %cst_npu_4 = arith.constant 213216 : i32
      %cst_npu_5 = arith.constant 1260724769 : i32
      aiex.npu.write32(%cst_npu_4, %cst_npu_5) {column = 0 : i32, row = 2 : i32} : i32, i32
      %cst_npu_6 = arith.constant 213220 : i32
      %cst_npu_7 = arith.constant 439168079 : i32
      aiex.npu.write32(%cst_npu_6, %cst_npu_7) {column = 0 : i32, row = 2 : i32} : i32, i32
      %cst_npu_8 = arith.constant 261888 : i32
      %cst_npu_9 = arith.constant 289 : i32
      aiex.npu.write32(%cst_npu_8, %cst_npu_9) {column = 0 : i32, row = 2 : i32} : i32, i32
      %cst_npu_10 = arith.constant 261892 : i32
      %cst_npu_11 = arith.constant 0 : i32
      aiex.npu.write32(%cst_npu_10, %cst_npu_11) {column = 0 : i32, row = 2 : i32} : i32, i32
      %cst_npu_12 = arith.constant 212992 : i32
      %cst_npu_13 = arith.constant 31232 : i32
      aiex.npu.write32(%cst_npu_12, %cst_npu_13) {column = 0 : i32, row = 2 : i32} : i32, i32
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      %cst_npu_14 = arith.constant 0 : i32
      aiex.npu.address_patch(%cst_npu_14 : i32) {addr = 119268 : ui32, arg_idx = 4 : i32}
      %cst_npu_15 = arith.constant 119308 : i32
      %cst_npu_16 = arith.constant 15 : i32
      aiex.npu.write32(%cst_npu_15, %cst_npu_16) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_17 = arith.constant 212992 : i32
      %cst_npu_18 = arith.constant 32512 : i32
      aiex.npu.write32(%cst_npu_17, %cst_npu_18) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_19 = arith.constant 213068 : i32
      %cst_npu_20 = arith.constant 127 : i32
      aiex.npu.write32(%cst_npu_19, %cst_npu_20) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_21 = arith.constant 213000 : i32
      %cst_npu_22 = arith.constant 127 : i32
      aiex.npu.write32(%cst_npu_21, %cst_npu_22) {column = 0 : i32, row = 0 : i32} : i32, i32
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi16> offset = 0 len = 4096 sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32> offset = 0 len = 1 sizes = [1, 1, 1, 1] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi16> offset = 0 len = 4096 sizes = [1, 1, 1, 4096] strides = [0, 0, 0, 1]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      %cst_npu_23 = arith.constant 213064 : i32
      %cst_npu_24 = arith.constant 126 : i32
      aiex.npu.write32(%cst_npu_23, %cst_npu_24) {column = 0 : i32, row = 0 : i32} : i32, i32
      %cst_npu_25 = arith.constant 213000 : i32
      %cst_npu_26 = arith.constant 126 : i32
      aiex.npu.write32(%cst_npu_25, %cst_npu_26) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}
