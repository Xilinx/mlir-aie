// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s

// CHECK: // Tile(5, 4)
// CHECK: // Memory map: name base_address num_bytes
// CHECK: _symbol y 0x28000 32
// CHECK: // Tile(4, 4)
// CHECK: // Memory map: name base_address num_bytes
// CHECK: _symbol t 0x30000 32
// CHECK: _symbol y 0x38000 32
// CHECK: _symbol z 0x20000 32
// CHECK: _symbol a 0x28000 16
// CHECK: _symbol b 0x28010 64
// CHECK: _symbol c 0x28050 1024
// CHECK: // Tile(4, 5)
// CHECK: // Memory map: name base_address num_bytes
// CHECK: _symbol t 0x38000 32
// CHECK: _symbol a 0x20000 16
// CHECK: _symbol b 0x20010 64
// CHECK: _symbol c 0x20050 1024
// CHECK: // Tile(4, 3)
// CHECK: // Memory map: name base_address num_bytes
// CHECK: _symbol z 0x38000 32
// CHECK: _symbol a 0x30000 16
// CHECK: _symbol b 0x30010 64
// CHECK: _symbol c 0x30050 1024
// CHECK: // Tile(3, 4)
// CHECK: // Memory map: name base_address num_bytes
// CHECK: _symbol x 0x28000 32
// CHECK: _symbol a 0x38000 16
// CHECK: _symbol b 0x38010 64
// CHECK: _symbol c 0x38050 1024

module @test_mmap0 {
  %t44 = AIE.tile(4, 4)
  %t34 = AIE.tile(3, 4)
  %t54 = AIE.tile(5, 4)
  %t43 = AIE.tile(4, 3)
  %t45 = AIE.tile(4, 5)

  %buf44_0 = AIE.buffer(%t44) { sym_name = "a" } : memref<4xi32>
  %buf44_1 = AIE.buffer(%t44) { sym_name = "b" } : memref<16xi32>
  %buf44_2 = AIE.buffer(%t44) { sym_name = "c" } : memref<256xi32>
  %buf34_0 = AIE.buffer(%t34) { sym_name = "x" } : memref<8xi32>
  %buf54_0 = AIE.buffer(%t54) { sym_name = "y" } : memref<8xi32>
  %buf43_0 = AIE.buffer(%t43) { sym_name = "z" } : memref<8xi32>
  %buf45_0 = AIE.buffer(%t45) { sym_name = "t" } : memref<8xi32>

}

