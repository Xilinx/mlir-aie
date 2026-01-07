// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
module {
    aie.device(npu2) @add_two {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)
        
        aie.objectfifo @objfifo_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<128xi32>>
        aie.objectfifo @objfifo_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<128xi32>>
        
        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c2_i32 = arith.constant 2 : i32
            %c8 = arith.constant 8 : index
            %c128 = arith.constant 128 : index
            %c_intmax = arith.constant 0xFFFFFE : index

            scf.for %niter = %c0 to %c_intmax step %c1 {
            %subview_in  = aie.objectfifo.acquire @objfifo_in (Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
            %subview_out = aie.objectfifo.acquire @objfifo_out(Produce, 1) : !aie.objectfifosubview<memref<128xi32>>
            %elem_in     = aie.objectfifo.subview.access %subview_in [0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
            %elem_out    = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
            scf.for %i = %c0 to %c128 step %c1 {
                %0 = memref.load %elem_in[%i] : memref<128xi32>
                %1 = arith.addi %0, %c2_i32 : i32
                memref.store %1, %elem_out[%i] : memref<128xi32>
            }
            aie.objectfifo.release @objfifo_in (Consume, 1)
            aie.objectfifo.release @objfifo_out(Produce, 1)
            }
            aie.end
        }

        aie.runtime_sequence @sequence(%a : memref<512xi32>) {
            
            // The "ID" attribute must match the ID given in "config.json".
            // During compilation, aiebu will package the PDI for this core together with this runtime sequence into a ELF file.
            // At runtime, XRT will load the PDI into memory and patch the address of this load_pdi to the correct address, so we can set 0 for the address.
            // As far as I can tell currently, the size attribute is also patched at runtime or ignored, so we can set 0 for it as well.
            // Current load_pdi:
            //aiex.npu.load_pdi { id = 1 : i32, size = 0 : i32, address = 0 : ui64 }
            // Goal for new load_pdi after refactoring:
            aiex.npu.load_pdi { device_ref = @add_two }

            %t_in = aiex.dma_configure_task_for @objfifo_in {
                aie.dma_bd(%a : memref<512xi32>, 0, 512)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%a: memref<512xi32>, 0, 512)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }

    }
}
