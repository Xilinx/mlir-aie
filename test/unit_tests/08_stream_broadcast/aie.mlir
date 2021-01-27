
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module @test3_core_llvm1 {
  %tile13 = AIE.tile(1, 3)
  %tile23 = AIE.tile(2, 3)

  %tile32 = AIE.tile(3, 2)
  %tile33 = AIE.tile(3, 3)
  %tile34 = AIE.tile(3, 4)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a13" } : memref<256xi32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b13" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3) // input buffer lock
  %lock13_5 = AIE.lock(%tile13, 5) // interbuffer lock

//  AIE.switchbox(%tile13) { AIE.connect<"DMA": 0, "East": 1> }
//  AIE.switchbox(%tile23) { AIE.connect<"West": 1, "East": 3> }
//  AIE.switchbox(%tile33) { AIE.connect<"West": 3, "DMA": 1> }

  AIE.flow(%tile13, "DMA" : 0, %tile32, "DMA" : 1)
  AIE.flow(%tile13, "DMA" : 0, %tile33, "DMA" : 1)
  AIE.flow(%tile13, "DMA" : 0, %tile34, "DMA" : 1)

  // Broadcast source tile (tile13)
  %core13 = AIE.core(%tile13) {
    AIE.useLock(%lock13_3, "Acquire", 1, 0) // acquire for read(e.g. input ping)
    AIE.useLock(%lock13_5, "Acquire", 0, 0) // acquire for write
    %idx1 = constant 3 : index
    %val1 = load %buf13_0[%idx1] : memref<256xi32>
    %2    = addi %val1, %val1 : i32
    %3 = addi %2, %val1 : i32
    %4 = addi %3, %val1 : i32
    %5 = addi %4, %val1 : i32
    %idx2 = constant 5 : index
    store %5, %buf13_1[%idx2] : memref<256xi32>
    AIE.useLock(%lock13_3, "Release", 0, 0) // release for write
    AIE.useLock(%lock13_5, "Release", 1, 0) // release for read
    AIE.end
  } { elf_file = "aie13.elf" }

  %mem13 = AIE.mem(%tile13) {
//    %dma0 = AIE.dmaStart("MM2S0")
//    AIE.terminator(^dma0, ^end)
//    %dma0 = AIE.dmaStart("MM2S0", ^dma0, ^end)
    %dma0 = AIE.dmaStart("MM2S0", ^bd0, ^end)
//    ^dma0:
//      cond_br %dma0, ^bd0, ^end // point to the start BD
    ^bd0:
      AIE.useLock(%lock13_5, "Acquire", 1, 0)
      AIE.dmaBd(<%buf13_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%lock13_5, "Release", 0, 0)
      br ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }


  // Broadcast target tile #1 (tile31)
  %buf31_0 = AIE.buffer(%tile31) { sym_name = "a31" } : memref<256xi32>
  %buf31_1 = AIE.buffer(%tile31) { sym_name = "b31" } : memref<256xi32>

  %lock31_6 = AIE.lock(%tile31, 6) // interbuffer lock
  %lock31_7 = AIE.lock(%tile31, 7) // output buffer lock

  %core31 = AIE.core(%tile31) {
    AIE.useLock(%lock31_6, "Acquire", 1, 0) // acquire for read(e.g. input ping)
    AIE.useLock(%lock31_7, "Acquire", 0, 0) // acquire for write
    %idx1 = constant 3 : index
    %val1 = load %buf31_0[%idx1] : memref<256xi32>
    %2    = addi %val1, %val1 : i32
    %3 = addi %2, %val1 : i32
    %4 = addi %3, %val1 : i32
    %5 = addi %4, %val1 : i32
    %idx2 = constant 5 : index
    store %5, %buf31_1[%idx2] : memref<256xi32>
    AIE.useLock(%lock31_6, "Release", 0, 0) // release for write
    AIE.useLock(%lock31_7, "Release", 1, 0) // release for read
    AIE.end

  } { elf_file = "aie31.elf" }

  %mem31 = AIE.mem(%tile31) {
    %dma0 = AIE.dmaStart("S2MM0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock31_7, "Acquire", 0, 0)
      AIE.dmaBd(<%buf31_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%lock31_7, "Release", 1, 0)
      br ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }

  // Broadcast target tile #2 (tile32)
  %buf32_0 = AIE.buffer(%tile32) { sym_name = "a32" } : memref<256xi32>
  %buf32_1 = AIE.buffer(%tile32) { sym_name = "b32" } : memref<256xi32>

  %lock32_6 = AIE.lock(%tile32, 6) // interbuffer lock
  %lock32_7 = AIE.lock(%tile32, 7) // output buffer lock

  %core32 = AIE.core(%tile32) {
    AIE.useLock(%lock32_6, "Acquire", 1, 0) // acquire for read(e.g. input ping)
    AIE.useLock(%lock32_7, "Acquire", 0, 0) // acquire for write
    %idx1 = constant 3 : index
    %val1 = load %buf32_0[%idx1] : memref<256xi32>
    %2    = addi %val1, %val1 : i32
    %3 = addi %2, %val1 : i32
    %4 = addi %3, %val1 : i32
    %5 = addi %4, %val1 : i32
    %idx2 = constant 5 : index
    store %5, %buf32_1[%idx2] : memref<256xi32>
    AIE.useLock(%lock32_6, "Release", 0, 0) // release for write
    AIE.useLock(%lock32_7, "Release", 1, 0) // release for read
    AIE.end

  } { elf_file = "aie32.elf" }

  %mem32 = AIE.mem(%tile32) {
    %dma0 = AIE.dmaStart("S2MM0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock32_7, "Acquire", 0, 0)
      AIE.dmaBd(<%buf32_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%lock32_7, "Release", 1, 0)
      br ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }

  // Broadcast target tile #3 (tile33)
  %buf33_0 = AIE.buffer(%tile33) { sym_name = "a33" } : memref<256xi32>
  %buf33_1 = AIE.buffer(%tile33) { sym_name = "b33" } : memref<256xi32>

  %lock33_6 = AIE.lock(%tile33, 6) // interbuffer lock
  %lock33_7 = AIE.lock(%tile33, 7) // output buffer lock

  %core33 = AIE.core(%tile33) {
    AIE.useLock(%lock33_6, "Acquire", 1, 0) // acquire for read(e.g. input ping)
    AIE.useLock(%lock33_7, "Acquire", 0, 0) // acquire for write
    %idx1 = constant 3 : index
    %val1 = load %buf33_0[%idx1] : memref<256xi32>
    %2    = addi %val1, %val1 : i32
    %3 = addi %2, %val1 : i32
    %4 = addi %3, %val1 : i32
    %5 = addi %4, %val1 : i32
    %idx2 = constant 5 : index
    store %5, %buf33_1[%idx2] : memref<256xi32>
    AIE.useLock(%lock33_6, "Release", 0, 0) // release for write
    AIE.useLock(%lock33_7, "Release", 1, 0) // release for read
    AIE.end

  } { elf_file = "aie33.elf" }

  %mem32 = AIE.mem(%tile32) {
    %dma0 = AIE.dmaStart("S2MM0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock32_7, "Acquire", 0, 0)
      AIE.dmaBd(<%buf32_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%lock32_7, "Release", 1, 0)
      br ^end // point to the next BD, or termination
    ^end:
      AIE.end
  }

}
