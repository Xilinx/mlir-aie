//===- allocate_link_pool_depth.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s | FileCheck %s

// A link pool that participants share through their allocates holds as many
// objects as the deepest of their ends uses, as the participants' ends would
// without the allocates: each end keeps enough for its largest acquire plus
// one, or its declared depth when no core acquires there. A declared depth no
// end uses would make the pool's DMA walk more BDs than a tile has, or fill its
// memory.

// in0 is declared 17 deep at its producer; the core there acquires one.
// CHECK-LABEL: module @owner_declared_17
// CHECK: aie.lock({{.*}}) {init = 2 : i32, sym_name = "in0__cons__prodlock0"}
// CHECK-NOT: in0__cons__buf2
module @owner_declared_17 {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    aie.objectfifo @in0(%a, {%l}, [17, 2]) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// A time dispatch to four outputs walks lcm(depth, 4) BDs: 4 for the two
// objects the ends use, 20 for the declared five.
// CHECK-LABEL: module @dispatch_declared_5
// CHECK: aie.lock({{.*}}) {init = 2 : i32, sym_name = "in0__cons__prodlock0"}
// CHECK-NOT: in0__cons__buf2
module @dispatch_declared_5 {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    aie.objectfifo @in0(%a, {%l}, [5, 2]) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o0(%l, {%s0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o1(%l, {%s1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o2(%l, {%s2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @o3(%l, {%s3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@o0, @o1, @o2, @o3] ([] []) {mode = #aie.link_mode<time>}
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// Six 8 KB objects and a 24 KB buffer do not fit the link tile; two do.
// CHECK-LABEL: module @owner_declared_6_large
// CHECK: aie.lock({{.*}}) {init = 2 : i32, sym_name = "in0__cons__prodlock0"}
// CHECK-NOT: in0__cons__buf2
module @owner_declared_6_large {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    aie.objectfifo @in0(%a, {%l}, [6, 2]) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    %big = aie.buffer(%l) {sym_name = "big"} : memref<6000xi32>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<2048xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// out shares in0's pool through its own allocate, and its core acquires three
// objects, so the pool takes four, though in0 owns it.
// CHECK-LABEL: module @sharer_deeper
// CHECK: aie.lock({{.*}}) {init = 4 : i32, sym_name = "in0__cons__prodlock0"}
module @sharer_deeper {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %l = aie.tile(0, 2)
    %c = aie.tile(0, 3)
    aie.objectfifo @in0(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%l, {%c}, [2, 4]) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @out (%l)
    aie.core(%c) {
      %0, %1, %2 = aie.objectfifo.acquire @out(Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @out(Consume, 3)
      aie.end
    }
  }
}

// -----

// Both participants share the pool; out's consumer end is the deepest.
// CHECK-LABEL: module @owner_and_sharer
// CHECK: aie.lock({{.*}}) {init = 4 : i32, sym_name = "in0__cons__prodlock0"}
module @owner_and_sharer {
  aie.device(npu2_1col) {
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%l, {%c}, [2, 4]) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    aie.objectfifo.allocate @out (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%c) {
      %0, %1, %2 = aie.objectfifo.acquire @out(Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @out(Consume, 3)
      aie.end
    }
  }
}

// -----

// out shares in0's pool through its own allocate. Its end on the link's tile
// has no core, so no end uses its declared 17 objects: the pool takes the two
// its consumer end uses.
// CHECK-LABEL: module @sharer_declared_17
// CHECK: aie.lock({{.*}}) {init = 2 : i32, sym_name = "in0__cons__prodlock0"}
// CHECK-NOT: in0__cons__buf2
module @sharer_declared_17 {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %l = aie.tile(0, 2)
    %c = aie.tile(0, 3)
    aie.objectfifo @in0(%s, {%l}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%l, {%c}, 17 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @out (%l)
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
  }
}

// -----

// The input side: in0 shares the pool out's initial objects make out's; its
// producer core acquires one of its declared 17.
// CHECK-LABEL: module @input_sharer_declared_17
// CHECK: aie.lock({{.*}}) {init = 2 : i32, sym_name = "out__conslock0"}
// CHECK-NOT: out__buf2
module @input_sharer_declared_17 {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %l = aie.tile(0, 2)
    %a = aie.tile(0, 3)
    aie.objectfifo @in0(%a, {%l}, 17 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>, dense<2> : memref<16xi32>]
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}
