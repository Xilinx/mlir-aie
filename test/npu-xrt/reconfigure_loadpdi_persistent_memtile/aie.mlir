// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This test:
// 1. Calls the @yield_const_memtile design. This sets up a buffer in mem tile 1, 1 and writes data (11, 22, 33, 44) to it, then streams that data to a core that passes it through back to DDR.
// 2. Calls the @yield_uninitialized_memtile design. This does the same as the other design except it doesn't initialize the buffer. This tests whether the memtile data is overwritten between reconfigurations.

module {

    aie.device(npu2) @main {

        aie.runtime_sequence @sequence(%arg : memref<512xi32>) {

            aiex.configure @yield_const_memtile {
                %arg1_subview = memref.subview %arg[0] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 0>>
                %arg1 = memref.reinterpret_cast %arg1_subview to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 0>> to memref<4xi32>
                aiex.run @sequence (%arg1) : (memref<4xi32>)
            }

            aiex.configure @yield_uninitialized_memtile {
                %arg1_subview = memref.subview %arg[4] [4] [1] : memref<512xi32> to memref<4xi32, strided<[1], offset: 4>>
                %arg1 = memref.reinterpret_cast %arg1_subview to offset: [0], sizes: [4], strides: [1] : memref<4xi32, strided<[1], offset: 4>> to memref<4xi32>
                aiex.run @sequence (%arg1) : (memref<4xi32>)
            }

        }

    }

    aie.device(npu2) @yield_const_memtile {

        %t00 = aie.tile(0, 0)
        %t01 = aie.tile(0, 1)
        %t11 = aie.tile(1, 1)
        %t02 = aie.tile(0, 2)
        %t12 = aie.tile(1, 2)
        
        %t11_lock = aie.lock(%t11) { init = 1 : i32 }
        %t11_buf = aie.buffer(%t11) { initial_value = dense<[11, 22, 33, 44]> : tensor<4xi32> } : memref<4xi32>

        %t11_dma = aie.memtile_dma(%t11) {
            %srcDma = aie.dma_start(MM2S, 0, ^bd0, ^end)
            ^bd0:
                aie.dma_bd(%t11_buf : memref<4xi32>, 0, 4, [])
                aie.next_bd ^bd0
            ^end:
                aie.end
        }

        %t12_buf_inp = aie.buffer(%t12) : memref<4xi32>
        %t12_buf_out = aie.buffer(%t12) : memref<4xi32>
        %t12_lock_input_consume = aie.lock(%t12) { init = 0 : i32 }
        %t12_lock_input_produce = aie.lock(%t12) { init = 1 : i32 }
        %t12_lock_output_consume = aie.lock(%t12) { init = 0 : i32 }
        %t12_lock_output_produce = aie.lock(%t12) { init = 1 : i32 }
        %t12_dma = aie.mem(%t12) {
            %dma1 = aie.dma_start(S2MM, 0, ^dma1_bd0, ^dma2)
            ^dma1_bd0:
                aie.use_lock(%t12_lock_input_produce, AcquireGreaterEqual, 1)
                aie.dma_bd(%t12_buf_inp : memref<4xi32>, 0, 4, [])
                aie.use_lock(%t12_lock_input_consume, Release, 1)
                aie.next_bd ^dma1_bd0
            ^dma2:
                %dma2 = aie.dma_start("MM2S", 0, ^dma2_bd0, ^end)
            ^dma2_bd0:
                aie.use_lock(%t12_lock_output_consume, AcquireGreaterEqual, 1)
                aie.dma_bd(%t12_buf_out : memref<4xi32>, 0, 4, [])
                aie.use_lock(%t12_lock_output_produce, Release, 1)
                aie.next_bd ^dma2_bd0
            ^end:
                aie.end
        }

        aie.flow(%t11, DMA : 0, %t12, DMA : 0)
        aie.flow(%t12, DMA : 0, %t00, DMA : 0)
        
        aie.core(%t12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c2_i32 = arith.constant 2 : i32
            %c8 = arith.constant 8 : index
            %c4 = arith.constant 4 : index
            %c_intmax = arith.constant 0xFFFFFE : index

            scf.for %niter = %c0 to %c_intmax step %c1 {
                aie.use_lock(%t12_lock_input_consume, AcquireGreaterEqual, 1)
                aie.use_lock(%t12_lock_output_produce, AcquireGreaterEqual, 1)
                memref.copy %t12_buf_inp, %t12_buf_out : memref<4xi32> to memref<4xi32>
                aie.use_lock(%t12_lock_input_produce, Release, 1)
                aie.use_lock(%t12_lock_output_consume, Release, 1)
            }
            aie.end
        }

        aie.runtime_sequence (%a : memref<4xi32>) {
            %t_out = aiex.dma_configure_task (%t00, S2MM, 0) {
                aie.dma_bd(%a: memref<4xi32>, 0, 4)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }

    }
    
    aie.device(npu2) @yield_uninitialized_memtile {

        %t00 = aie.tile(0, 0)
        %t01 = aie.tile(0, 1)
        %t11 = aie.tile(1, 1)
        %t02 = aie.tile(0, 2)
        %t12 = aie.tile(1, 2)
        
        %t11_lock = aie.lock(%t11) { init = 1 : i32 }
        %t11_buf = aie.buffer(%t11) : memref<4xi32>

        %t11_dma = aie.memtile_dma(%t11) {
            %srcDma = aie.dma_start(MM2S, 0, ^bd0, ^end)
            ^bd0:
                aie.dma_bd(%t11_buf : memref<4xi32>, 0, 4, [])
                aie.next_bd ^bd0
            ^end:
                aie.end
        }

        %t12_buf_inp = aie.buffer(%t12) : memref<4xi32>
        %t12_buf_out = aie.buffer(%t12) : memref<4xi32>
        %t12_lock_input_consume = aie.lock(%t12) { init = 0 : i32 }
        %t12_lock_input_produce = aie.lock(%t12) { init = 1 : i32 }
        %t12_lock_output_consume = aie.lock(%t12) { init = 0 : i32 }
        %t12_lock_output_produce = aie.lock(%t12) { init = 1 : i32 }
        %t12_dma = aie.mem(%t12) {
            %dma1 = aie.dma_start(S2MM, 0, ^dma1_bd0, ^dma2)
            ^dma1_bd0:
                aie.use_lock(%t12_lock_input_produce, AcquireGreaterEqual, 1)
                aie.dma_bd(%t12_buf_inp : memref<4xi32>, 0, 4, [])
                aie.use_lock(%t12_lock_input_consume, Release, 1)
                aie.next_bd ^dma1_bd0
            ^dma2:
                %dma2 = aie.dma_start("MM2S", 0, ^dma2_bd0, ^end)
            ^dma2_bd0:
                aie.use_lock(%t12_lock_output_consume, AcquireGreaterEqual, 1)
                aie.dma_bd(%t12_buf_out : memref<4xi32>, 0, 4, [])
                aie.use_lock(%t12_lock_output_produce, Release, 1)
                aie.next_bd ^dma2_bd0
            ^end:
                aie.end
        }

        aie.flow(%t11, DMA : 0, %t12, DMA : 0)
        aie.flow(%t12, DMA : 0, %t00, DMA : 0)
        
        aie.core(%t12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %c2_i32 = arith.constant 2 : i32
            %c8 = arith.constant 8 : index
            %c4 = arith.constant 4 : index
            %c_intmax = arith.constant 0xFFFFFE : index

            scf.for %niter = %c0 to %c_intmax step %c1 {
                aie.use_lock(%t12_lock_input_consume, AcquireGreaterEqual, 1)
                aie.use_lock(%t12_lock_output_produce, AcquireGreaterEqual, 1)
                memref.copy %t12_buf_inp, %t12_buf_out : memref<4xi32> to memref<4xi32>
                aie.use_lock(%t12_lock_input_produce, Release, 1)
                aie.use_lock(%t12_lock_output_consume, Release, 1)
            }
            aie.end
        }

        aie.runtime_sequence (%a : memref<4xi32>) {
            %t_out = aiex.dma_configure_task (%t00, S2MM, 0) {
                aie.dma_bd(%a: memref<4xi32>, 0, 4)
                aie.end
            } {issue_token = true}
            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)
        }

    }
}
