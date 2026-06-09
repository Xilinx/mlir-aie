//===- dynamic_cyclostatic_conditional_release.mlir -----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Negative test: release is conditional on a runtime predicate, so we can't
// statically compute the per-iter carry. The fix should detect this and
// either (a) emit a diagnostic and leave alone, or (b) be conservative and
// not hoist.
//
// Today, the pass silently lowers to AcquireGE(3) every iter (broken). After
// the fix, the user should at least know their program is suspect. The
// expected-error directive below pins the desired diagnostic.
//
// Until the diagnostic is implemented, this test will fail because no error
// is emitted. Once the diagnostic lands, this CHECK will pass.

// RUN: aie-opt --verify-diagnostics --aie-objectFifo-stateful-transform="dynamic-objFifos=true" %s

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo(%tile_0_1, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<8xi8>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c14 = arith.constant 14 : index
      %true = arith.constant true
      scf.for %arg0 = %c0 to %c14 step %c1 {
        %x = aie.objectfifo.acquire @fifo(Consume, 3) : !aie.objectfifosubview<memref<8xi8>>
        scf.if %true {
          // expected-error@+1 {{cannot statically analyze cyclostatic acquire pattern: acquire/release is inside a conditional}}
          aie.objectfifo.release @fifo(Consume, 1)
        }
      }
      aie.end
    }
  }
}
