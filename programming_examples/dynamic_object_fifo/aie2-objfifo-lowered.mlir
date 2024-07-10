module {
  aie.device(npu1_1col) {
    memref.global "public" @output_fifo_cons : memref<64xi32>
    memref.global "public" @output_fifo : memref<64xi32>
    memref.global "public" @input_fifo_cons : memref<64xi32>
    memref.global "public" @input_fifo : memref<64xi32>
    func.func private @passthrough_64_i32(memref<64xi32>, memref<64xi32>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %output_fifo_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "output_fifo_cons_prod_lock"}
    %output_fifo_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "output_fifo_cons_cons_lock"}
    %output_fifo_buff_0 = aie.buffer(%tile_0_2) {sym_name = "output_fifo_buff_0"} : memref<64xi32> 
    %output_fifo_buff_1 = aie.buffer(%tile_0_2) {sym_name = "output_fifo_buff_1"} : memref<64xi32> 
    %output_fifo_prod_lock = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "output_fifo_prod_lock"}
    %output_fifo_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "output_fifo_cons_lock"}
    %input_fifo_cons_buff_0 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_0"} : memref<64xi32> 
    %input_fifo_cons_buff_1 = aie.buffer(%tile_0_2) {sym_name = "input_fifo_cons_buff_1"} : memref<64xi32> 
    %input_fifo_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "input_fifo_cons_prod_lock"}
    %input_fifo_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "input_fifo_cons_cons_lock"}
    %input_fifo_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "input_fifo_prod_lock"}
    %input_fifo_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "input_fifo_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {

        // -- NEW --
        // Loop count modulo object fifo depth, in this case depth == 2
        %arg0_mod_depth = arith.remsi %arg0, %c2 : index
        // There will be depth-1 of these variables
        // arg0_mod_depth_is_0
        // arg0_mod_depth_is_1
        // arg0_mod_depth_is_...
        // arg0_mod_depth_is_{depth-1}
        // Maybe you will need to give it a prefix with the fifo name to not have SSA value name collisions,
        // eg. input_fifo_arg0_mod_depth_is_...
        %arg0_mod_depth_is_0 = arith.cmpi eq, %arg0_mod_depth, %c0 : index

        // Choose which buffer to use this iteration.
        // Since this is ping-pong, every even iteration is ping buffer, every odd is pong buffer.
        // There will be as many if, else if, else ... blocks here as the fifo depth.
        // If unclear, I'm happy to give an example for deeper fifos.
        %input_fifo_cons_buff = scf.if %arg0_mod_depth_is_0 -> (memref<64xi32>) {
          scf.yield %input_fifo_cons_buff_0 : memref<64xi32>
        } else {
          scf.yield %input_fifo_cons_buff_1 : memref<64xi32>
        }

        // -- --
        // Everything below is identical but for the output fifo.
        // A smart pass could combine all of this into one scf.if statement,
        // as long as the depths of fifos are equal. 
        // Let's start easy generating the code for each fifo separately.
        // I think that's fine.
        %output_fifo_buff = scf.if %arg0_mod_depth_is_0 -> (memref<64xi32>) {
          scf.yield %output_fifo_buff_0 : memref<64xi32>
        } else {
          scf.yield %output_fifo_buff_1 : memref<64xi32>
        }

        // -- END NEW --
        // The only other difference is that the loops are not unrolled at all, they remain untouched.
        // And the buffers below reference the ones we selected above.


        aie.use_lock(%input_fifo_cons_cons_lock, AcquireGreaterEqual, 1)
        aie.use_lock(%output_fifo_prod_lock, AcquireGreaterEqual, 1)
        func.call @passthrough_64_i32(%input_fifo_cons_buff, %output_fifo_buff) : (memref<64xi32>, memref<64xi32>) -> ()
        aie.use_lock(%output_fifo_cons_lock, Release, 1)
        aie.use_lock(%input_fifo_cons_prod_lock, Release, 1)
      }
      aie.end
    } {link_with = "kernel.o"}
    aie.shim_dma_allocation @input_fifo(MM2S, 0, 0)
    func.func @sequence(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @input_fifo} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 2 : i64, metadata = @output_fifo} : memref<64xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    aie.shim_dma_allocation @output_fifo(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_0 : memref<64xi32>, 0, 64)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%input_fifo_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%input_fifo_cons_buff_1 : memref<64xi32>, 0, 64)
      aie.use_lock(%input_fifo_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_0 : memref<64xi32>, 0, 64)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%output_fifo_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%output_fifo_buff_1 : memref<64xi32>, 0, 64)
      aie.use_lock(%output_fifo_prod_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      aie.end
    }
  }
}

