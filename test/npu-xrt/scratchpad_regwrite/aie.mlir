// (c) Copyright 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Test: Use XAIE_IO_UPDATE_REG to write a runtime parameter into a core's
// local buffer, then pass it back out to DDR via an ObjectFIFO.
//
// The host writes a value (pre-shifted left by 2) into the scratchpad
// StateTable[0]. The runtime sequence uses UPDATE_REG to add that value
// to a zero-initialized buffer in the core tile's data memory.
//
// Because UPDATE_REG:
//   - Is read-modify-write (adds to existing value): buffer must be 0-init
//   - Masks lower 2 bits of the lower word to 0: value must be pre-shifted
//     left by 2, then the core right-shifts by 2 to recover the original
//   - Always writes 8 contiguous bytes: we use a 2xi32 buffer (only [0]
//     carries meaningful data; [1] will be 0 for values < 2^32)
//
// Synchronization: The core blocks on a lock acquire. The runtime sequence
// performs update_reg BEFORE setting the lock, so by the time the core
// unblocks, the buffer is already written.
//
module {
    // Empty device needed to force load_pdi reconfiguration
    aie.device(npu2) @empty { }

    // Actual test configuration PDI
    aie.device(npu2) @regwrite_test {

        %t00 = aie.tile(0, 0)
        %t02 = aie.tile(0, 2)

        // Local buffer for the runtime parameter (zero-initialized by default).
        // UPDATE_REG writes 8 bytes here (2 x i32).
        %params_buf = aie.buffer(%t02) {sym_name = "params_buf"} : memref<2xi32>

        // Lock for synchronization between runtime sequence and core.
        %sync_lock = aie.lock(%t02, 0) {init = 0 : i32, sym_name = "sync_lock"}

        // Output ObjectFIFO: core sends the recovered value to DDR.
        aie.objectfifo @objfifo_out (%t02, {%t00}, 1 : i32) : !aie.objectfifo<memref<1xi32>>

        // Core
        aie.core(%t02) {
            %c0 = arith.constant 0 : index
            %c2_i32 = arith.constant 2 : i32

            // Block until run-time parameters are ready
            aie.use_lock(%sync_lock, Acquire, 1)

            // Read the parameter written by UPDATE_REG
            %raw_val = memref.load %params_buf[%c0] : memref<2xi32>

            // Reset run-time parameter lock for next iteration
            aie.use_lock(%sync_lock, Release, 0)

            // Undo the left-shift-by-2 that was applied to survive the 0xFFFFFFFC masking in the firmware
            %val = arith.shrui %raw_val, %c2_i32 : i32

            // Output the recovered value
            %out_view = aie.objectfifo.acquire @objfifo_out (Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
            %out_buf = aie.objectfifo.subview.access %out_view[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
            memref.store %val, %out_buf[%c0] : memref<1xi32>
            aie.objectfifo.release @objfifo_out (Produce, 1)

            aie.end
        }

        // Runtime sequence:
        // 1. Load PDI (starts core, which immediately blocks on lock acquire)
        // 2. Create scratchpad (copies DDR value written by host to firmware SRAM; host has written value<<2 to StateTable[0] in test_elf.cpp)
        // 3. UPDATE_REG: adds StateTable[0] to params_buf (zero + value<<2)
        // 4. Set lock to 1 (unblocks core)
        // 5. Configure and start output DMA
        // 6. Await output completion
        aie.runtime_sequence @sequence(%out : memref<1xi32>) {

            aiex.npu.load_pdi { device_ref = @empty }
            aiex.npu.load_pdi { device_ref = @regwrite_test }

            // Create scratchpad: pulls host-written StateTable from DDR.
            // Host writes (desired_value << 2) to StateTable[0].
            aiex.npu.create_scratchpad { size = 4 : ui32 }

            // Write the runtime parameter into params_buf.
            // Func = INCR(1), FuncArg = 0: offset = StateTable[0] + 0
            // Result: params_buf[0] = (0 + StateTable[0]) & 0xFFFFFFFC
            //         params_buf[1] = 0 (for values < 2^32)
            aiex.npu.update_from_scratchpad {
                state_table_idx = 0 : ui8,
                buffer = @params_buf,
                address = 0 : ui32
            }

            // Unblock the core (lock was init=0, now set to 1)
            aiex.set_lock(%sync_lock, 1)

            // Configure output DMA
            %t_out = aiex.dma_configure_task_for @objfifo_out {
                aie.dma_bd(%out : memref<1xi32>, 0, 1)
                aie.end
            } {issue_token = true}

            aiex.dma_start_task(%t_out)
            aiex.dma_await_task(%t_out)

        }

    }
}
