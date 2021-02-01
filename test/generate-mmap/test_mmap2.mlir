// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s

// CHECK-LABEL: Tile(3, 2)
// CHECK: _symbol a 0x30000 16
// CHECK-LABEL: Tile(3, 3)
// CHECK: _symbol a 0x38000 16
// CHECK-LABEL: Tile(2, 3)
// CHECK-NOT: _symbol a
// CHECK-LABEL: Tile(4, 3)
// CHECK: _symbol a 0x28000 16
// CHECK-LABEL: Tile(3, 4)
// CHECK: _symbol a 0x20000 16

module @test_mmap1 {
  %tsame = AIE.tile(3, 3)
  %twest = AIE.tile(2, 3) // Different column
  %teast = AIE.tile(4, 3) // Different column
  %tsouth = AIE.tile(3, 2) // Different row
  %tnorth = AIE.tile(3, 4) // Different row

  %bufsame = AIE.buffer(%tsame) { sym_name = "a", address = 0x0 } : memref<4xi32>
}

