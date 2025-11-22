// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
module {
    aie.device(npu2) @main {

        aiex.runtime_sequence @sequence(%a : memref<512xi32>) {

            %c2_i32 = arith.constant 2 : i32
            %c3_i32 = arith.constant 3 : i32

            aiex.configure @add_n(%c2_i32) {
                %arg = aiex.arg_slice %a[0..4] : memref<512xi32> -> memref<4xi32>
                aiex.run @sequence (%arg) : (memref<4xi32>)
            }

            aiex.configure @add_n(%c3_i32) {
                %arg = aiex.arg_slice %a[4..8] : memref<512xi32> -> memref<4xi32>
                aiex.run @sequence (%arg) : (memref<4xi32>)
            }

        }

    }

    aie.device(npu2) @add_n(%c_n_i32 : i32) {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)
        
        aie.objectfifo @objfifo_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo @objfifo_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        
        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c8 = arith.constant 8 : index
            %c4 = arith.constant 4 : index
            %c_intmax = arith.constant 0xFFFFFE : index

            scf.for %niter = %c0 to %c_intmax step %c1 {
            %subview_in  = aie.objectfifo.acquire @objfifo_in (Consume, 1) : !aie.objectfifosubview<memref<4xi32>>
            %subview_out = aie.objectfifo.acquire @objfifo_out(Produce, 1) : !aie.objectfifosubview<memref<4xi32>>
            %elem_in     = aie.objectfifo.subview.access %subview_in [0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
            %elem_out    = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<4xi32>> -> memref<4xi32>
            scf.for %i = %c0 to %c4 step %c1 {
                %0 = memref.load %elem_in[%i] : memref<4xi32>
                %1 = arith.addi %0, %c_n_i32 : i32
                memref.store %1, %elem_out[%i] : memref<4xi32>
            }
            aie.objectfifo.release @objfifo_in (Consume, 1)
            aie.objectfifo.release @objfifo_out(Produce, 1)
            }
            aie.end
        }

        aiex.runtime_sequence @sequence(%a : memref<512xi32>) {
            
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
