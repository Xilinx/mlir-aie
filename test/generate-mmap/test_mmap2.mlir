// Note: This test *might* fail due to the random order that the code statements are generated

// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s

// CHECK-LABEL: // Tile(5, 4)
// CHECK: _symbol a 0x28000 16
// CHECK-LABEL: // Tile(4, 4)
// CHECK: _symbol a 0x28000 16
// CHECK-LABEL: // Tile(4, 5)
// CHECK: _symbol a 0x20000 16
// CHECK-LABEL: // Tile(4, 3)
// CHECK: _symbol a 0x30000 16
// CHECK-LABEL: // Tile(3, 4)
// CHECK-NOT: _symbol a

module @test_mmap1 {
  %t44 = AIE.tile(4, 4)
  %t34 = AIE.tile(3, 4) // Different column
  %t54 = AIE.tile(5, 4) // Different column
  %t43 = AIE.tile(4, 3) // Different row
  %t45 = AIE.tile(4, 5) // Different row

  %buf44_0 = AIE.buffer(%t44) { sym_name = "a" } : memref<4xi32>
}

