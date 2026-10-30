//===- consumed_by.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shim input records the shim outputs whose progress proves its bytes
// consumed: once the host has received R bytes through the output, the first
// R * input_bytes / output_bytes bytes the input sent have been taken.

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s --check-prefix=ALLOC

// A core taking one object of `in` before releasing each object of `out`.
// Allocation moves the proof onto the input's shim_dma_allocation, its output
// named by the allocation the runtime sequence reaches it through.

// CHECK-LABEL: module @one_to_one
// ALLOC-LABEL: module @one_to_one
// ALLOC: aie.shim_dma_allocation @in_shim_alloc(%{{.*}}, MM2S, {{[0-9]+}}) {consumed_by = [#aie.consumed_by<output = @out_shim_alloc, input_bytes = 1, output_bytes = 1>]}
// CHECK: aie.route_endpoint @in_prod_dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out_cons_dma, input_bytes = 1, output_bytes = 1>], fifoName = "in"}
// CHECK-NOT: consumed_by
module @one_to_one {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// One angle object a row of 24 outputs, the inner loop's trip count; `x`
// pairs one to one with the outputs, whatever the outer loop's count.

// CHECK-LABEL: module @rows_of_outputs
// CHECK-DAG: aie.route_endpoint @angle_prod_dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out_cons_dma, input_bytes = 1, output_bytes = 24>], fifoName = "angle"}
// CHECK-DAG: aie.route_endpoint @x_prod_dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out_cons_dma, input_bytes = 1, output_bytes = 1>], fifoName = "x"}
module @rows_of_outputs {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @x(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @angle(%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    %rows = aie.buffer(%core) : memref<1xi32>
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c24 = arith.constant 24 : index
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %n = memref.load %rows[%c0] : memref<1xi32>
        %rows_n = arith.index_cast %n : i32 to index
        scf.for %r = %c0 to %rows_n step %c1 {
          %a = aie.objectfifo.acquire @angle(Consume, 1) : memref<128xbf16>
          scf.for %j = %c0 to %c24 step %c1 {
            %i = aie.objectfifo.acquire @x(Consume, 1) : memref<128xbf16>
            %o = aie.objectfifo.acquire @out(Produce, 1) : memref<128xbf16>
            aie.objectfifo.release @x(Consume, 1)
            aie.objectfifo.release @out(Produce, 1)
          }
          aie.objectfifo.release @angle(Consume, 1)
        }
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// A memtile relay: the bytes the input sends reach the output one to one.

// CHECK-LABEL: module @relay
// ALLOC-LABEL: module @relay
// ALLOC: aie.shim_dma_allocation @a_shim_alloc(%{{.*}}, MM2S, {{[0-9]+}}) {consumed_by = [#aie.consumed_by<output = @b_shim_alloc, input_bytes = 1, output_bytes = 1>]}
// CHECK: aie.route_endpoint @a_prod_dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @b_cons_dma, input_bytes = 1, output_bytes = 1>], fifoName = "a"}
module @relay {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    aie.objectfifo @a(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @b(%mem, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo.link [@a] -> [@b] ([] [])
  }
}

// -----

// Through memtile links on both sides of the core, with objects of different
// sizes on each hop: the proof is in bytes.

// CHECK-LABEL: module @link_chain
// CHECK: aie.route_endpoint @in_prod_dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @res_cons_dma, input_bytes = 1, output_bytes = 1>], fifoName = "in"}
module @link_chain {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @in_l1(%mem, {%core}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@in] -> [@in_l1] ([] [])
    aie.objectfifo @out_l1(%core, {%mem}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @res(%mem, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@out_l1] -> [@res] ([] [])
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in_l1(Consume, 1) : memref<32xi32>
        %o = aie.objectfifo.acquire @out_l1(Produce, 1) : memref<32xi32>
        aie.objectfifo.release @in_l1(Consume, 1)
        aie.objectfifo.release @out_l1(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The core releases each output before taking the input that goes with it,
// so an output's arrival proves nothing about the input.

// CHECK-LABEL: module @release_before_take
// CHECK-NOT: consumed_by
module @release_before_take {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @out(Produce, 1)
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// A window holding one object of `in` from one pass into the next: the pass
// does not end where it began, so no ratio is read off it.

// CHECK-LABEL: module @window_across_passes
// CHECK-NOT: consumed_by
module @window_across_passes {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i:2 = aie.objectfifo.acquire @in(Consume, 2) : memref<16xi32>, memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The input feeds one core and the output comes from another: nothing ties
// the output's progress to the input.

// CHECK-LABEL: module @other_core
// CHECK-NOT: consumed_by
module @other_core {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core2 = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)
    aie.objectfifo @in(%shim, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid(%core2, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core3, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %m = aie.objectfifo.acquire @mid(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @mid(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%core3) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %m = aie.objectfifo.acquire @mid(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @mid(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// A broadcast input reaches two cores; only one of them produces the output,
// so the other may still be taking its copy.

// CHECK-LABEL: module @broadcast
// CHECK-NOT: consumed_by
module @broadcast {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core2 = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)
    aie.objectfifo @in(%shim, {%core2, %core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core2, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%core2) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
        scf.yield
      }
      aie.end
    }
    aie.core(%core3) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        scf.yield
      }
      aie.end
    }
  }
}
