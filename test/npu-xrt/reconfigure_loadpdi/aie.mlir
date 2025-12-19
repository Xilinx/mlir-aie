// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This test:
//  1. Calls the @add_two design on the first four i32 elements of the input.
//  2. Calls the @add_two design on the last four i32 elements of the input.
//  3. Calls the @add_three design on i32 elements 4-7 of the input.
//  4. Calls the @add_three_design on the last four i32 elements of the input.
// Elements 8-11 remain untouched by any design.
// This tests, end-to-end:
//  - Inlining ('calling') runtime sequences multiple times.
//  - Slicing input argument buffers to the runtime sequence in different ways.
//  - Reconfiguration between different designs.

module {

    aie.device(npu2) @main {

        aie.runtime_sequence @sequence(%arg : memref<512xi32>) {

            %c3_i32 = arith.constant 3 : i32

            aiex.configure @add_two {
                %arg1_subview = memref.subview %arg[0] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 0>>
                %arg1 = memref.reinterpret_cast %arg1_subview to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 0>> to memref<4xi32>
                aiex.run @add_two_sequence (%arg1) : (memref<4xi32>)
                %arg2_subview = memref.subview %arg[12] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 12>>
                %arg2 = memref.reinterpret_cast %arg2_subview to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 12>> to memref<4xi32>
                aiex.run @add_two_sequence (%arg2) : (memref<4xi32>)
            }

            aiex.configure @add_three {
                %arg1_subview = memref.subview %arg[4] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 4>>
                %arg1 = memref.reinterpret_cast %arg1_subview to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 4>> to memref<4xi32>
                aiex.run @add_three_sequence (%arg1) : (memref<4xi32>)
                %arg2_subview = memref.subview %arg[12] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 12>>
                %arg2 = memref.reinterpret_cast %arg2_subview to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 12>> to memref<4xi32>
                aiex.run @add_three_sequence (%arg2) : (memref<4xi32>)
            }

        }

    }

    aie.device(npu2) @add_two {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)
        
        aie.objectfifo @objfifo_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo @objfifo_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        
        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c2_i32 = arith.constant 2 : i32
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
                %1 = arith.addi %0, %c2_i32 : i32
                memref.store %1, %elem_out[%i] : memref<4xi32>
            }
            aie.objectfifo.release @objfifo_in (Consume, 1)
            aie.objectfifo.release @objfifo_out(Produce, 1)
            }
            aie.end
        }

        aie.runtime_sequence @add_two_sequence(%a : memref<4xi32>) {
            
            %t_in = aiex.dma_configure_task_for @objfifo_in {
                aie.dma_bd(%a : memref<4xi32>, 0, 4)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%a: memref<4xi32>, 0, 4)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }

    }

    aie.device(npu2) @add_three {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)
        
        aie.objectfifo @objfifo_in (%t00, {%t02}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        aie.objectfifo @objfifo_out(%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<4xi32>>
        
        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c3_i32 = arith.constant 3 : i32
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
                %1 = arith.addi %0, %c3_i32 : i32
                memref.store %1, %elem_out[%i] : memref<4xi32>
            }
            aie.objectfifo.release @objfifo_in (Consume, 1)
            aie.objectfifo.release @objfifo_out(Produce, 1)
            }
            aie.end
        }

        aie.runtime_sequence @add_three_sequence(%a : memref<4xi32>) {
            
            %t_in = aiex.dma_configure_task_for @objfifo_in {
                aie.dma_bd(%a : memref<4xi32>, 0, 4)
                aie.end
            }
            %t_out = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%a: memref<4xi32>, 0, 4)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_in)
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }

    }
}
