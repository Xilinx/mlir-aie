// RUN: aie-translate --aie-generate-txn-cpp %s
//
// Dynamic runtime_sequence that takes buffer_length as a runtime parameter.
// Uses BLOCKWRITE for BD configuration (required for address_patch to work),
// then overwrites BD word[0] with dyn_write32 for the dynamic buffer_length.
//
// The buffer_length argument is in units of 32-bit words (bytes / 4).
// For example, to transfer 4096 bytes, pass buffer_length = 1024.
//
// The core loops forever processing fixed-size ObjectFIFO elements.
// As long as total transfer is a multiple of the element size, it works.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in_shim_alloc(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out_shim_alloc(%tile_0_0, S2MM, 0)

    // Trace BD data (tile 0,0 BD 15) - size-independent
    memref.global "private" constant @blockwrite_data_trace : memref<8xi32> = dense<[0, 0, 1073741824, 0, 0, 33554432, 0, 33554432]>

    // Input BD (address 0x1D000 = 118784): word[0]=placeholder, patched by dyn_write32
    // word[4]=0xC0000000 (burst_length=3<<30), word[5]=0x02000000 (AXCache=2<<24)
    // word[7]=0x02000000 (valid_bd=1<<25)
    memref.global "private" constant @blockwrite_data_in_bd : memref<8xi32> = dense<[0, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>

    // Output BD (address 0x1D020 = 118816): same layout
    memref.global "private" constant @blockwrite_data_out_bd : memref<8xi32> = dense<[0, 0, 0, 0, -1073741824, 33554432, 0, 33554432]>

    aie.runtime_sequence(%arg0: memref<4096xui8>, %arg1: memref<4096xui8>, %arg2: memref<4096xui8>, %buffer_length: i32, %input_bd_addr: i32, %output_bd_addr: i32) {

      // =====================================================================
      // Trace configuration (tile 0,2) - all static
      // =====================================================================
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 1260724769 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 439168079 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}

      // Trace BD (tile 0,0 BD 15)
      %trace_data = memref.get_global @blockwrite_data_trace : memref<8xi32>
      aiex.npu.blockwrite(%trace_data) {address = 119264 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 119268 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}
      aiex.npu.maskwrite32 {address = 119304 : ui32, column = 0 : i32, mask = 7936 : ui32, row = 0 : i32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 2147483663 : ui32}

      // =====================================================================
      // DMA queue configuration (tile 0,0) - static
      // =====================================================================
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}

      // =====================================================================
      // Input BD (0x1D000 = 118784, BD 0)
      // 1. BLOCKWRITE all 8 words (word[0]=0 placeholder, makes address_patch work)
      // 2. address_patch word[1] with input buffer physical address
      // 3. dyn_write32 overwrites word[0] with dynamic buffer_length
      // =====================================================================
      %in_bd_data = memref.get_global @blockwrite_data_in_bd : memref<8xi32>
      aiex.npu.blockwrite(%in_bd_data) {address = 118784 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
      aiex.npu.dyn_write32(%input_bd_addr, %buffer_length) : i32, i32

      // Start input DMA queue
      aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}

      // =====================================================================
      // Output BD (0x1D020 = 118816, BD 1)
      // Same pattern: BLOCKWRITE → address_patch → dyn_write32
      // =====================================================================
      %out_bd_data = memref.get_global @blockwrite_data_out_bd : memref<8xi32>
      aiex.npu.blockwrite(%out_bd_data) {address = 118816 : ui32} : memref<8xi32>
      aiex.npu.address_patch {addr = 118820 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}
      aiex.npu.dyn_write32(%output_bd_addr, %buffer_length) : i32, i32

      // Start output DMA and configure for sync token
      aiex.npu.maskwrite32 {address = 119296 : ui32, mask = 7936 : ui32, value = 3840 : ui32}
      aiex.npu.write32 {address = 119300 : ui32, value = 2147483649 : ui32}

      // =====================================================================
      // Wait for output DMA completion
      // =====================================================================
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}

      // Cleanup (disable trace)
      aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
    }
  }
}
