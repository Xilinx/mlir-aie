
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module @test3_core_llvm1 {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3)
  %lock13_5 = AIE.lock(%tile13, 5)

  %core13 = AIE.core(%tile13) { AIE.end } { elf_file = "aie.elf" }

}
