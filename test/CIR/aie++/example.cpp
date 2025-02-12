/*
https://github.com/Xilinx/mlir-aie/blob/main/mlir_tutorials/tutorial-1/answers/aie_q5.mlir
module @tutorial_1 {
  // Declare tile object of the AIE class located at position col 1, row 4
  %tile14 = aie.tile(1, 4)

  // Declare buffer for tile(1, 4) with symbolic name "a14" and
  // size 256 deep x int32 wide. By default, the address of
  // this buffer begins after the stack (1024 Bytes offset) and
  // all subsequent buffers are allocated one after another in memory.
  %buf = aie.buffer(%tile14) { sym_name = "a14" } : memref<8192xi32>

  // Define the algorithm for the core of tile(1, 4)
  // buf[3] = 14
  %core14 = aie.core(%tile14) {
    %val = arith.constant 14 : i32 // declare a constant (int32)
    %idx = arith.constant 3 : index // declare a constant (index)
    memref.store %val, %buf[%idx] : memref<8192xi32> // store val in buf[3]
    aie.end
  }
}
*/

#include "aie++.hpp"

int main() {
  aie::device<aie::npu1> d;
  auto t = d.tile<1, 4>();
  auto b = t.buffer<int, 8192>();
  t.program([&] { b[3] = 14; });
  d.tile<2, 3>().program([] {});
  d.run();
}
