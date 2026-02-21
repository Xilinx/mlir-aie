// XFAIL: *
// Test for dynamic runtime sequence generation
// This demonstrates runtime-parameterized transaction sequences
// where the buffer size is determined at runtime

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Input/output buffers on shim tile
    %buf_in = aie.buffer(%tile_0_0) : memref<2048xi32>
    %buf_out = aie.buffer(%tile_0_0) : memref<2048xi32>

    // Core buffer
    %buf_core = aie.buffer(%tile_0_2) : memref<1024xi32>

    // Locks for synchronization
    %lock_in_prod = aie.lock(%tile_0_0, 0) { init = 1 : i32 }
    %lock_in_cons = aie.lock(%tile_0_0, 1) { init = 0 : i32 }
    %lock_out_prod = aie.lock(%tile_0_0, 2) { init = 1 : i32 }
    %lock_out_cons = aie.lock(%tile_0_0, 3) { init = 0 : i32 }

    // Shim DMA allocations
    %shim_dma_in = aie.shim_dma_allocation @in(MM2S, 0, 0)
    %shim_dma_out = aie.shim_dma_allocation @out(S2MM, 0, 0)

    // Dynamic runtime sequence with size parameter
    // The num_lines parameter determines how many cache lines to transfer
    // line_size is fixed at 64 bytes (16 x i32)
    aie.runtime_sequence @dynamic_sequence(%buf_in_arg: memref<2048xi32>,
                                           %buf_out_arg: memref<2048xi32>,
                                           %num_lines: index) {
      // Constants
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index  // Elements per line (64 bytes / 4 bytes)
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32

      // Loop over number of lines, transferring one line at a time
      scf.for %line = %c0 to %num_lines step %c1 {
        // Compute offset for this line (in elements)
        %offset = arith.muli %line, %c16 : index

        // Dynamic DMA configuration for input
        // This would generate BD programming with runtime-computed offsets
        aiex.npu.dyn_dma_memcpy_nd(
          %buf_in_arg,
          [%offset], [%c16], [%c1],
          metadata = @in, id = 0
        ) : memref<2048xi32>, [index], [index], [index]

        // Dynamic DMA configuration for output
        aiex.npu.dyn_dma_memcpy_nd(
          %buf_out_arg,
          [%offset], [%c16], [%c1],
          metadata = @out, id = 1
        ) {issue_token = true} : memref<2048xi32>, [index], [index], [index]

        // Wait for output transfer to complete
        %col = arith.constant 0 : i32
        %row = arith.constant 0 : i32
        %dir = arith.constant 0 : i32  // S2MM
        %chan = arith.constant 0 : i32
        %col_num = arith.constant 1 : i32
        %row_num = arith.constant 1 : i32
        aiex.npu.dyn_sync(%col, %row, %dir, %chan, %col_num, %row_num)
          : i32, i32, i32, i32, i32, i32
      }
    }
  }
}
