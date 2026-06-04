// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test: DMA address offset patching via offset_parameter.
//
// The host prepares an input buffer with monotonically increasing i32 values
// [0, 1, 2, ..., 31].  The core does a simple passthrough of 8 elements.
// The offset_parameter @input_offset controls where in the input buffer
// the DMA begins reading.
//
//   Run 1: input_offset = 0  → output = [0, 1, 2, 3, 4, 5, 6, 7]
//   Run 2: input_offset = 8  → output = [8, 9, 10, 11, 12, 13, 14, 15]
//   Run 3: input_offset = 16 → output = [16, 17, 18, 19, 20, 21, 22, 23]
//
module {
    // Runtime parameter: element offset into the input buffer.
    // aiex.parameter ops are declared at module scope (global across devices).
    aiex.parameter @input_offset : i32

    aie.device(npu2) @empty { }

    aie.device(npu2) @test {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        // Lock to gate the core until parameters are loaded
        %sync_lock = aie.lock(%t02, 0) {init = 0 : i32, sym_name = "sync_lock"}

        // ObjectFIFOs
        aie.objectfifo @objfifo_in  (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
        aie.objectfifo @objfifo_out (%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<8xi32>>

        // Core: passthrough — copy input to output
        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c8 = arith.constant 8 : index

            // Wait for parameters + DMA to be ready
            aie.use_lock(%sync_lock, Acquire, 1)
            aie.use_lock(%sync_lock, Release, 0)

            %in_view = aie.objectfifo.acquire @objfifo_in (Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
            %in_buf  = aie.objectfifo.subview.access %in_view[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>

            %out_view = aie.objectfifo.acquire @objfifo_out (Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
            %out_buf  = aie.objectfifo.subview.access %out_view[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>

            scf.for %i = %c0 to %c8 step %c1 {
                %v = memref.load %in_buf[%i] : memref<8xi32>
                memref.store %v, %out_buf[%i] : memref<8xi32>
            }

            aie.objectfifo.release @objfifo_in  (Consume, 1)
            aie.objectfifo.release @objfifo_out (Produce, 1)

            aie.end
        }

        // Runtime sequence
        aie.runtime_sequence @sequence(%in : memref<32xi32>, %out : memref<8xi32>) {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @test }

            // Load scratchpad parameters from host
            aiex.sync_parameters_from_host

            // Unblock core
            aiex.set_lock(%sync_lock, 1)

            // Input DMA — offset_parameter patches the BD address at runtime
            %t_in = aiex.dma_configure_task_for @objfifo_in {
                aie.dma_bd(%in : memref<32xi32>, 0, 8) {offset_parameter = @input_offset}
                aie.end
            }

            // Output DMA
            %t_out = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%out : memref<8xi32>, 0, 8)
                aie.end
            } {issue_token = true}

            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)

            aiex.set_lock(%sync_lock, 0)
        }
    }
}
