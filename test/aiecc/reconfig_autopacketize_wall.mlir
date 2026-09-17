//===- reconfig_autopacketize_wall.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A single column with two circuit shim inputs claims both of the column's shim
// MM2S channels, leaving none for the resident control-packet overlay. With
// auto-packetize disabled the control overlay cannot ingress, so the build must
// stop at the shim-MM2S wall rather than emit an unroutable design. (Auto-packetize
// is on by default and clears this wall by packet-switching one ingress leg; here
// it is turned off to assert the diagnostic still fires.)

// RUN: not aiecc --get-full-elf --reconfig-method=ctrlpkt --ctrlpkt-auto-packetize=false %s 2>&1 | FileCheck %s

// CHECK: all shim mm2s dma channels for column 0 are reserved by circuit-switched flows

module {
    aie.device(npu2) @main {
        aie.runtime_sequence @config_1(%arg0_1 : memref<4xi32>, %arg1_1 : memref<4xi32>, %arg2_1 : memref<4xi32>) {
            aiex.configure @baseline_1 {
                aiex.run @baseline_1_sequence (%arg0_1, %arg1_1, %arg2_1) : (memref<4xi32>, memref<4xi32>, memref<4xi32>)
            }
        }
    }
    aie.device(npu2) @baseline_1 {
        %tshim_1 = aie.tile(0, 0)
        %tmem_1 = aie.tile(0, 1)
        %tcore_1 = aie.tile(0, 2)

        aie.objectfifo @in0_shim_1 (%tshim_1, {%tmem_1}, 2 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo @in0_mem_1 (%tmem_1, {%tcore_1}, 2 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo.link [@in0_shim_1] -> [@in0_mem_1]([] [0])

        aie.objectfifo @in1_shim_1 (%tshim_1, {%tmem_1}, 2 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo @in1_mem_1 (%tmem_1, {%tcore_1}, 2 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo.link [@in1_shim_1] -> [@in1_mem_1]([] [0])

        aie.objectfifo @out_1(%tcore_1, {%tshim_1}, 2 : i32) : !aie.objectfifo<memref<4xi32>>

        aie.core(%tcore_1) {
            %c0_1 = arith.constant 0 : index
            %c1_1 = arith.constant 1 : index
            %cn_1 = arith.constant 4 : index
            %cmax_1 = arith.constant 0xFFFFFE : index
            scf.for %niter_1 = %c0_1 to %cmax_1 step %c1_1 {
                %ein0_1 = aie.objectfifo.acquire @in0_mem_1 (Consume, 1) : memref<4xi32>
                %ein1_1 = aie.objectfifo.acquire @in1_mem_1 (Consume, 1) : memref<4xi32>
                %eout_1 = aie.objectfifo.acquire @out_1(Produce, 1) : memref<4xi32>
                scf.for %ii_1 = %c0_1 to %cn_1 step %c1_1 {
                    %v0_1 = memref.load %ein0_1[%ii_1] : memref<4xi32>
                    %v1_1 = memref.load %ein1_1[%ii_1] : memref<4xi32>
                    %r_1 = arith.addi %v0_1, %v1_1 : i32
                    memref.store %r_1, %eout_1[%ii_1] : memref<4xi32>
                }
                aie.objectfifo.release @in0_mem_1 (Consume, 1)
                aie.objectfifo.release @in1_mem_1 (Consume, 1)
                aie.objectfifo.release @out_1(Produce, 1)
            }
            aie.end
        }

        aie.runtime_sequence @baseline_1_sequence(%a0_1 : memref<4xi32>, %a1_1 : memref<4xi32>, %ao_1 : memref<4xi32>) {
            %t_in0_1 = aiex.dma_configure_task_for @in0_shim_1 {
                aie.dma_bd(%a0_1 : memref<4xi32> offset = 0 len = 4)
                aie.end
            }
            %t_in1_1 = aiex.dma_configure_task_for @in1_shim_1 {
                aie.dma_bd(%a1_1 : memref<4xi32> offset = 0 len = 4)
                aie.end
            }
            %t_out_1 = aiex.dma_configure_task_for @out_1 {
                aie.dma_bd(%ao_1 : memref<4xi32> offset = 0 len = 4)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in0_1)
            aiex.dma_start_task(%t_in1_1)
            aiex.dma_start_task(%t_out_1)
            aiex.dma_await_task(%t_out_1)
            aiex.dma_free_task(%t_in0_1)
            aiex.dma_free_task(%t_in1_1)
            aiex.dma_free_task(%t_out_1)
        }
    }
}
