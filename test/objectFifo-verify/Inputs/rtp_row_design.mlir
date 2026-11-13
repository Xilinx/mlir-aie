//===- rtp_row_design.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The RTP-row design several consumed_by*.mlir cases share: a RoPE-shaped
// core reads a column count from the `rtp` buffer under %barrier and relays
// that many @x objects to @out for each @angle object. A case's
// `// RTP-ROW-DESIGN` line, inside its aie.device, expands to this body.
    %shim = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @x(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @angle(%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @out(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    %rtp = aie.buffer(%core) {sym_name = "rtp"} : memref<1xi32>
    %barrier = aie.lock(%core)
    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aie.use_lock(%barrier, AcquireGreaterEqual, %one)
        %n = memref.load %rtp[%c0] : memref<1xi32>
        aie.use_lock(%barrier, Release, %one)
        %cols = arith.index_cast %n : i32 to index
        %a = aie.objectfifo.acquire @angle(Consume, 1) : memref<128xbf16>
        scf.for %j = %c0 to %cols step %c1 {
          %i = aie.objectfifo.acquire @x(Consume, 1) : memref<128xbf16>
          %o = aie.objectfifo.acquire @out(Produce, 1) : memref<128xbf16>
          aie.objectfifo.release @x(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @angle(Consume, 1)
        scf.yield
      }
      aie.end
    }
