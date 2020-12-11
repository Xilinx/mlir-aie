// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s
// CHECK:    AIE.buffer({{.*}}) {address = 6176 : i32, sym_name = "a" } : memref<16xi8>
// CHECK:    AIE.buffer({{.*}}) {address = 4096 : i32, sym_name = "b" } : memref<512xi32>
// CHECK:    AIE.buffer({{.*}}) {address = 6144 : i32, sym_name = "c" } : memref<16xi16>
// CHECK:    AIE.buffer({{.*}}) {address = 4096 : i32, sym_name = "d" } : memref<500xi32>

module @test {
  %0 = AIE.tile(3, 3)
  %b1 = AIE.buffer(%0) { sym_name = "a" } : memref<16xi8>
  %1 = AIE.buffer(%0) { sym_name = "b" } : memref<512xi32>
  %b2 = AIE.buffer(%0) { sym_name = "c" } : memref<16xi16>
  %3 = AIE.tile(4, 4)
  %4 = AIE.buffer(%3) { sym_name = "d" } : memref<500xi32>
}
