//===- alloc_group_propagation.mlir ----------------------------*- MLIR -*-===//
//
//
//===----------------------------------------------------------------------===//

// An objectFifo's alloc_group reaches every buffer it lowers to, unmodified. A
// fifo's own depth slots are live together -- that is what depth means -- which
// is exactly what one group asserts, so they all carry the same name and stay
// distinct. Two fifos that never run together take DIFFERENT groups and overlay.
//
// The group rides the pool, not the fifo: split gives a shim-fed fifo a
// separate consumer-side pool, and that is the one whose buffers land in core
// L1. Without the pool carrying it, the overlay would apply to the producer
// half only.

// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-allocate %s | FileCheck %s
// CHECK-DAG: aie.buffer({{.*}}) {alloc_group = "a", sym_name = "mode_a_cons_buff_0"}
// CHECK-DAG: aie.buffer({{.*}}) {alloc_group = "a", sym_name = "mode_a_cons_buff_1"}
// CHECK-DAG: aie.buffer({{.*}}) {alloc_group = "b", sym_name = "mode_b_cons_buff_0"}
// CHECK-DAG: aie.buffer({{.*}}) {alloc_group = "b", sym_name = "mode_b_cons_buff_1"}
// CHECK-DAG: aie.buffer({{.*}}) {sym_name = "plain_cons_buff_0"}
module @alloc_group_propagation {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %core = aie.tile(0, 2)
    %core2 = aie.tile(0, 3)
    aie.objectfifo @mode_a (%shim, { %core }, 2 : i32) {alloc_group = "a"} : !aie.objectfifo<memref<512xi32>>
    aie.objectfifo @mode_b (%shim, { %core }, 2 : i32) {alloc_group = "b"} : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @plain  (%shim1, { %core2 }, 2 : i32) : !aie.objectfifo<memref<64xi32>>
  }
}
