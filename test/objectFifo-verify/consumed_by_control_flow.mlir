//===- consumed_by_control_flow.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's ratio of inputs taken to outputs released holds through branches
// and loops whose trip counts are runtime words, for every value of what
// decides them: each arm and any number of passes must keep the core's takes
// ahead of its releases.

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify %s | FileCheck %s

// Either arm of a branch on data takes the input before the output goes.

// CHECK-LABEL: module @both_arms_take
// CHECK: aie.route_endpoint @in__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 1>], fifoName = "in"}
module @both_arms_take {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %flag = aie.buffer(%core) : memref<1xi32>
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %f = memref.load %flag[%c0] : memref<1xi32>
        %emit = arith.cmpi ne, %f, %c0_i32 : i32
        scf.if %emit {
          %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
          %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        } else {
          %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
          aie.objectfifo.release @in(Consume, 1)
        }
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// A loop whose count is a word nothing fixes takes two inputs ahead of each
// output: however often it runs, two for one.

// CHECK-LABEL: module @unknown_count_two_per_output
// CHECK: aie.route_endpoint @in__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 2, output_bytes = 1>], fifoName = "in"}
module @unknown_count_two_per_output {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %count = aie.buffer(%core) : memref<1xi32>
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %n = memref.load %count[%c0] : memref<1xi32>
        %rows = arith.index_cast %n : i32 to index
        scf.for %r = %c0 to %rows step %c1 {
          %i:2 = aie.objectfifo.acquire @in(Consume, 2) : memref<16xi32>, memref<16xi32>
          aie.objectfifo.release @in(Consume, 2)
          %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
          aie.objectfifo.release @out(Produce, 1)
        }
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The shape of llama's lm_head core: one take per dispatch, then passes that
// either take one and emit one, or take sixteen and emit one, chosen at run
// time. At least one input precedes each output.

// CHECK-LABEL: module @dispatch_passes
// CHECK: aie.route_endpoint @in__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 1>], fifoName = "in"}
module @dispatch_passes {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %rtp = aie.buffer(%core) : memref<2xi32>
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c0_i32 = arith.constant 0 : i32
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %first = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        %n = memref.load %rtp[%c0] : memref<2xi32>
        %passes = arith.index_cast %n : i32 to index
        scf.for %p = %c0 to %passes step %c1 {
          %f = memref.load %rtp[%c1] : memref<2xi32>
          %wide = arith.cmpi ne, %f, %c0_i32 : i32
          scf.if %wide {
            scf.for %k = %c0 to %c16 step %c1 {
              %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
              aie.objectfifo.release @in(Consume, 1)
            }
            %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
            aie.objectfifo.release @out(Produce, 1)
          } else {
            %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
            %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
            aie.objectfifo.release @out(Produce, 1)
            aie.objectfifo.release @in(Consume, 1)
          }
        }
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// One arm emits an output without taking an input: a run of that arm leaves
// outputs no input paid for.

// CHECK-LABEL: module @arm_emits_without_taking
// CHECK-NOT: consumed_by
module @arm_emits_without_taking {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %flag = aie.buffer(%core) : memref<1xi32>
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %f = memref.load %flag[%c0] : memref<1xi32>
        %take = arith.cmpi ne, %f, %c0_i32 : i32
        scf.if %take {
          %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
          %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        } else {
          %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
          aie.objectfifo.release @out(Produce, 1)
        }
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// The arms leave different numbers of inputs held, so what the next take
// adds depends on which ran. The analysis does not merge such arms, so it
// records nothing here, though both paths take two inputs per output.

// CHECK-LABEL: module @arms_hold_differently
// CHECK-NOT: consumed_by
module @arms_hold_differently {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %flag = aie.buffer(%core) : memref<1xi32>
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %f = memref.load %flag[%c0] : memref<1xi32>
      %keep = arith.cmpi ne, %f, %c0_i32 : i32
      scf.if %keep {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
      }
      %j:2 = aie.objectfifo.acquire @in(Consume, 2) : memref<16xi32>, memref<16xi32>
      %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @out(Produce, 1)
      aie.objectfifo.release @in(Consume, 2)
      aie.end
    }
  }
}

// -----

// The shape of llama's attention and projection cores: one input taken, then
// a loop whose count is a runtime word releasing outputs. However few inputs
// the word allows per output, no ratio holds for every value of it.

// CHECK-LABEL: module @outputs_per_runtime_word
// CHECK-NOT: consumed_by
module @outputs_per_runtime_word {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @in(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %count = aie.buffer(%core) : memref<1xi32>
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %i = aie.objectfifo.acquire @in(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @in(Consume, 1)
        %n = memref.load %count[%c0] : memref<1xi32>
        %rows = arith.index_cast %n : i32 to index
        scf.for %r = %c0 to %rows step %c1 {
          %o = aie.objectfifo.acquire @out(Produce, 1) : memref<16xi32>
          aie.objectfifo.release @out(Produce, 1)
        }
        scf.yield
      }
      aie.end
    }
  }
}
