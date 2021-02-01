// RUN: aie-translate --aie-generate-mmap %s | FileCheck %s
// RUN: aie-translate --tilecol=4 --tilerow=4 --aie-generate-bcf %s | FileCheck --check-prefix=BCF44 %s

// CHECK-LABEL: Tile(5, 4)
// CHECK: _symbol y 0x28000 32
// CHECK-LABEL: Tile(4, 4)
// CHECK: _symbol z 0x20000 32
// CHECK: _symbol a 0x28000 16
// CHECK: _symbol b 0x28010 64
// CHECK: _symbol c 0x28050 1024
// CHECK: _symbol t 0x30000 32
// CHECK: _symbol y 0x38000 32
// CHECK-LABEL: Tile(4, 5)
// CHECK: _symbol a 0x20000 16
// CHECK: _symbol b 0x20010 64
// CHECK: _symbol c 0x20050 1024
// CHECK: _symbol t 0x38000 32
// CHECK-LABEL: Tile(4, 3)
// CHECK: _symbol a 0x30000 16
// CHECK: _symbol b 0x30010 64
// CHECK: _symbol c 0x30050 1024
// CHECK: _symbol z 0x38000 32
// CHECK-LABEL: Tile(3, 4)
// CHECK: _symbol x 0x28000 32
// CHECK: _symbol a 0x38000 16
// CHECK: _symbol b 0x38010 64
// CHECK: _symbol c 0x38050 1024

// BCF44: _entry_point _main_init
// BCF44: _symbol      _main _after _main_init
// BCF44: _symbol      _main_init 0
// BCF44: _reserved DMb      0x00000 0x20000 //Don't put data in code memory
// BCF44: _symbol z 0x20000 0x20
// BCF44: _extern z
// BCF44: _symbol a 0x28000 0x10
// BCF44: _extern a
// BCF44: _symbol b 0x28010 0x40
// BCF44: _extern b
// BCF44: _symbol c 0x28050 0x400
// BCF44: _extern c
// BCF44: _symbol t 0x30000 0x20
// BCF44: _extern t
// BCF44: _symbol y 0x38000 0x20
// BCF44: _extern y
// BCF44: _stack    DM_stack 0x20000  0x400 //stack for core
// BCF44: _reserved DMb 0x40000 0xc0000 // And everything else the core can't see

module @test_mmap0 {
  %t44 = AIE.tile(4, 4)
  %t34 = AIE.tile(3, 4)
  %t54 = AIE.tile(5, 4)
  %t43 = AIE.tile(4, 3)
  %t45 = AIE.tile(4, 5)

  %buf44_0 = AIE.buffer(%t44) { sym_name = "a", address = 0x0 } : memref<4xi32>
  %buf44_1 = AIE.buffer(%t44) { sym_name = "b", address = 0x10  } : memref<16xi32>
  %buf44_2 = AIE.buffer(%t44) { sym_name = "c", address = 0x50 } : memref<256xi32>
  %buf34_0 = AIE.buffer(%t34) { sym_name = "x", address = 0x0 } : memref<8xi32>
  %buf54_0 = AIE.buffer(%t54) { sym_name = "y", address = 0x0 } : memref<8xi32>
  %buf43_0 = AIE.buffer(%t43) { sym_name = "z", address = 0x0 } : memref<8xi32>
  %buf45_0 = AIE.buffer(%t45) { sym_name = "t", address = 0x0 } : memref<8xi32>

}

