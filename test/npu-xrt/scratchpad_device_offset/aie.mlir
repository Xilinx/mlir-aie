// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test: a DMA offset the device computes within the same command.
//
// The core reads a selector s and writes s * 8 into the run's own ctrl
// scratchpad (the host binds a user-pointer buffer over it as %pad), over
// the host's value there. The sequence waits for that write, and only then
// copies the scratchpad into the command processor
// (sync_scratchpad_parameters_from_host) and configures the fill that
// offset_parameter @row_offset patches: the core receives row s of the
// table, not the row the host wrote. The command processor copies the
// scratchpad where that op runs, once a command, so it comes after the write.
//
module {
    aiex.scratchpad_parameter @row_offset : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        aie.objectfifo @sel (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo @off (%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo @row (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
        aie.objectfifo @out (%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<8xi32>>

        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c8 = arith.constant 8 : index
            %c8_i32 = arith.constant 8 : i32

            %sel_buf = aie.objectfifo.acquire @sel (Consume, 1) : memref<1xi32>
            %off_buf = aie.objectfifo.acquire @off (Produce, 1) : memref<1xi32>
            %s = memref.load %sel_buf[%c0] : memref<1xi32>
            %o = arith.muli %s, %c8_i32 : i32
            memref.store %o, %off_buf[%c0] : memref<1xi32>
            aie.objectfifo.release @sel (Consume, 1)
            aie.objectfifo.release @off (Produce, 1)

            %row_buf = aie.objectfifo.acquire @row (Consume, 1) : memref<8xi32>
            %out_buf = aie.objectfifo.acquire @out (Produce, 1) : memref<8xi32>
            scf.for %i = %c0 to %c8 step %c1 {
                %v = memref.load %row_buf[%i] : memref<8xi32>
                memref.store %v, %out_buf[%i] : memref<8xi32>
            }
            aie.objectfifo.release @row (Consume, 1)
            aie.objectfifo.release @out (Produce, 1)

            aie.end
        }

        aie.runtime_sequence @sequence(%sel : memref<1xi32>, %table : memref<64xi32>,
                                       %out : memref<8xi32>, %pad : memref<1xi32>) {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            %t_sel = aiex.dma_configure_task_for @sel {
                aie.dma_bd(%sel : memref<1xi32> offset = 0 len = 1)
                aie.end
            }
            %t_off = aiex.dma_configure_task_for @off {
                aie.dma_bd(%pad : memref<1xi32> offset = 0 len = 1)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_sel)
            aiex.dma_start_task(%t_off)
            aiex.dma_await_task(%t_off)

            // The scratchpad, the core's offset in it, copied in.
            aiex.sync_scratchpad_parameters_from_host

            %t_row = aiex.dma_configure_task_for @row {
                aie.dma_bd(%table : memref<64xi32> offset = 0 len = 8) {offset_parameter = @row_offset}
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @out {
                aie.dma_bd(%out : memref<8xi32> offset = 0 len = 8)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_row)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }
    }
}
