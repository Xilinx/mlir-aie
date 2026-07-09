//===- dynamic_cyclostatic_switch_unbalanced.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Negative test for scf.index_switch: the branches *disagree* on their per-fifo
// net (case 0 nets +1: acquire 2 / release 1; the default nets 0: acquire 1 /
// release 1). Because the delta is data dependent (it varies with the runtime
// switch value) the carry cannot be computed statically. Since the same fifo
// is also used unconditionally, the pass must emit the diagnostic rather than
// peel on an incomplete carry.

// RUN: aie-opt --verify-diagnostics --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %u = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
        aie.objectfifo.release @fifo(Consume, 1)
        scf.index_switch %arg0
        case 0 {
          // net +1 on this path
          %a = aie.objectfifo.acquire @fifo(Consume, 2) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
          scf.yield
        }
        default {
          // net 0 on this path -> branches disagree -> unanalyzable
          // expected-error@+1 {{cannot statically analyze cyclostatic acquire pattern: acquire/release is inside a conditional}}
          %d = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi8>>
          aie.objectfifo.release @fifo(Consume, 1)
          scf.yield
        }
      }
      aie.end
    }
  }
}
