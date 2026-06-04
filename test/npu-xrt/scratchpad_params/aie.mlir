// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
module {
    // Parameters (declared at module scope; global across all devices)
    aiex.parameter @foo : bf16
    aiex.parameter @bar : bf16

    // Empty device needed to force load_pdi reconfiguration
    aie.device(npu2) @empty { }

    // Actual test configuration PDI
    aie.device(npu2) @test {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        // Output ObjectFIFO
        aie.objectfifo @objfifo_out (%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<2xbf16>>

        // Core: blocked on parameter-sync lock inserted by --aie-lower-parameters
        aie.core(%t02) {
            %c0 = arith.constant 0 : index

            // Read the bf16 parameters written by UPDATE_REG
            %foo = aiex.read_parameter @foo : bf16
            %bar = aiex.read_parameter @bar : bf16

            // Calculate result: foo * bar in bf16
            %val_bf16 = arith.mulf %foo, %bar : bf16

            %out_view = aie.objectfifo.acquire @objfifo_out (Produce, 1) : !aie.objectfifosubview<memref<2xbf16>>
            %out_buf = aie.objectfifo.subview.access %out_view[0] : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
            memref.store %val_bf16, %out_buf[%c0] : memref<2xbf16>
            aie.objectfifo.release @objfifo_out (Produce, 1)

            aie.end
        }

        // Runtime sequence: parameter-sync preamble inserted by --aie-lower-parameters
        aie.runtime_sequence @sequence(%out : memref<2xbf16>) attributes { emit_parameter_sync_preamble = false } {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            aiex.sync_parameters_from_host

            // Configure output DMA
            %t_out = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%out : memref<2xbf16>, 0, 2)
                aie.end
            } {issue_token = true}

            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        } 

    }
}
