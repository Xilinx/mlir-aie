//===- check_core_start_gating.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-check-core-start-gating --split-input-file -verify-diagnostics | FileCheck %s

// A memory tile relay of host data. The MM2S channel waits on a lock only the
// S2MM descriptor releases, and only the host's shim channel feeds that one.

// CHECK-LABEL: @relay
// CHECK:       aie.dma_start(S2MM, 0, ^{{.*}}, ^{{.*}}){{$}}
// CHECK:       aie.dma_start(MM2S, 0, ^{{.*}}, ^{{.*}}) {aiex.ctrl_pkt_gated}
module {
  aie.device(npu2) @relay {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    aie.shim_dma_allocation @in(%t00, MM2S, 0)
    aie.flow(%t00, DMA : 0, %t01, DMA : 0)
    aie.flow(%t01, DMA : 0, %t00, DMA : 0)
    %buf = aie.buffer(%t01) : memref<16xi32>
    %prod = aie.lock(%t01, 0) {init = 1 : i32}
    %cons = aie.lock(%t01, 1) {init = 0 : i32}
    %dma = aie.memtile_dma(%t01) {
      %0 = aie.dma_start(S2MM, 0, ^in, ^out)
    ^in:
      %v1 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %v1)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      %v2 = arith.constant 1 : i32
      aie.use_lock(%cons, Release, %v2)
      aie.next_bd ^in
    ^out:
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      %v3 = arith.constant 1 : i32
      aie.use_lock(%cons, AcquireGreaterEqual, %v3)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      %v4 = arith.constant 1 : i32
      aie.use_lock(%prod, Release, %v4)
      aie.next_bd ^send
    ^end:
      aie.end
    }
  }
}

// -----

// A lock whose initial value satisfies the acquire, so the channel streams as
// soon as it is started.

// CHECK-LABEL: @self_start
// CHECK-NOT:   aiex.ctrl_pkt_gated
module {
  aie.device(npu2) @self_start {
    %t01 = aie.tile(0, 1)
    %t10 = aie.tile(1, 0)
    aie.flow(%t01, DMA : 0, %t10, DMA : 0)
    %buf = aie.buffer(%t01) : memref<16xi32>
    %full = aie.lock(%t01, 0) {init = 1 : i32}
    %dma = aie.memtile_dma(%t01) {
      // expected-remark @+1 {{MM2S channel 0 of tile (0, 1) not proved gated: the lock's initial value satisfies its first buffer descriptor's acquire}}
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      %v5 = arith.constant 1 : i32
      aie.use_lock(%full, AcquireGreaterEqual, %v5)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// A first descriptor with no acquire, and one whose acquire does not wait.

// CHECK-LABEL: @no_wait
// CHECK-NOT:   aiex.ctrl_pkt_gated
module {
  aie.device(npu2) @no_wait {
    %t01 = aie.tile(0, 1)
    %buf = aie.buffer(%t01) : memref<16xi32>
    %l = aie.lock(%t01, 0) {init = 0 : i32}
    %dma = aie.memtile_dma(%t01) {
      // expected-remark @+1 {{MM2S channel 0 of tile (0, 1) not proved gated: its first buffer descriptor does not acquire a lock}}
      %0 = aie.dma_start(MM2S, 0, ^a, ^next)
    ^a:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^next:
      // expected-remark @+1 {{MM2S channel 1 of tile (0, 1) not proved gated: its first buffer descriptor's acquire does not wait}}
      %1 = aie.dma_start(MM2S, 1, ^b, ^end)
    ^b:
      %v6 = arith.constant 1 : i32
      aie.use_lock(%l, AcquireGreaterEqual, %v6) {acq_en = false}
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// A core tile's MM2S channel waits on a lock its core releases (cores start
// after the configuration), and the memory tile relays what that channel sends,
// so both are proved. A second core tile receives from a self-starting memory
// tile channel, so the lock its S2MM descriptor releases is not quiet.

// CHECK-LABEL: @through_core
// CHECK:       aie.mem(%{{.*}}) {
// CHECK-NEXT:    aie.dma_start(MM2S, 0, ^{{.*}}, ^{{.*}}) {aiex.ctrl_pkt_gated}
// CHECK:       aie.dma_start(MM2S, 0, ^{{.*}}, ^{{.*}}) {aiex.ctrl_pkt_gated}
// CHECK:       aie.dma_start(MM2S, 1, ^{{.*}}, ^{{.*}}){{$}}
// CHECK:       aie.mem(%{{.*}}) {
// CHECK:         aie.dma_start(MM2S, 0, ^{{.*}}, ^{{.*}}){{$}}
module {
  aie.device(npu2) @through_core {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t01, DMA : 0)
    aie.flow(%t01, DMA : 1, %t03, DMA : 0)
    %cbuf = aie.buffer(%t02) : memref<16xi32>
    %cl = aie.lock(%t02, 0)
    // expected-remark @+1 {{core (0, 2) not proved gated: a release before the gating acquire}}
    %core = aie.core(%t02) {
      %v7 = arith.constant 1 : i32
      aie.use_lock(%cl, Release, %v7)
      aie.end
    }
    %mem02 = aie.mem(%t02) {
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      %v8 = arith.constant 1 : i32
      aie.use_lock(%cl, AcquireGreaterEqual, %v8)
      aie.dma_bd(%cbuf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %mbuf = aie.buffer(%t01) : memref<16xi32>
    %prod = aie.lock(%t01, 0) {init = 1 : i32}
    %cons = aie.lock(%t01, 1) {init = 0 : i32}
    %full = aie.lock(%t01, 2) {init = 1 : i32}
    %dma = aie.memtile_dma(%t01) {
      %0 = aie.dma_start(S2MM, 0, ^in, ^out)
    ^in:
      %v9 = arith.constant 1 : i32
      aie.use_lock(%prod, AcquireGreaterEqual, %v9)
      aie.dma_bd(%mbuf : memref<16xi32> offset = 0 len = 16)
      %v10 = arith.constant 1 : i32
      aie.use_lock(%cons, Release, %v10)
      aie.next_bd ^in
    ^out:
      %1 = aie.dma_start(MM2S, 0, ^send, ^loose)
    ^send:
      %v11 = arith.constant 1 : i32
      aie.use_lock(%cons, AcquireGreaterEqual, %v11)
      aie.dma_bd(%mbuf : memref<16xi32> offset = 0 len = 16)
      %v12 = arith.constant 1 : i32
      aie.use_lock(%prod, Release, %v12)
      aie.next_bd ^send
    ^loose:
      // expected-remark @+1 {{MM2S channel 1 of tile (0, 1) not proved gated: the lock's initial value satisfies its first buffer descriptor's acquire}}
      %2 = aie.dma_start(MM2S, 1, ^go, ^end)
    ^go:
      %v13 = arith.constant 1 : i32
      aie.use_lock(%full, AcquireGreaterEqual, %v13)
      aie.dma_bd(%mbuf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %dbuf = aie.buffer(%t03) : memref<16xi32>
    %dl = aie.lock(%t03, 0)
    // expected-remark @+1 {{core (0, 3) not proved gated: no gating acquire}}
    %core3 = aie.core(%t03) {
      aie.end
    }
    %mem03 = aie.mem(%t03) {
      %0 = aie.dma_start(S2MM, 0, ^in, ^out)
    ^in:
      aie.dma_bd(%dbuf : memref<16xi32> offset = 0 len = 16)
      %v14 = arith.constant 1 : i32
      aie.use_lock(%dl, Release, %v14)
      aie.next_bd ^end
    ^out:
      // expected-remark @+1 {{MM2S channel 0 of tile (0, 3) not proved gated: the lock's initial value satisfies its first buffer descriptor's acquire}}
      %1 = aie.dma_start(MM2S, 0, ^send, ^end)
    ^send:
      %v15 = arith.constant 1 : i32
      aie.use_lock(%dl, AcquireGreaterEqual, %v15)
      aie.dma_bd(%dbuf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// A shim channel the device starts itself is a channel like any other. Here its
// descriptor does not wait, so neither it nor the relay it feeds is proved. A
// shim channel the device does not start (column 1's) is started only by the
// runtime sequence.

// CHECK-LABEL: @static_shim
// CHECK:       aie.dma_start(MM2S, 0, ^{{.*}}, ^{{.*}}){{$}}
// CHECK:       aie.dma_start(MM2S, 0, ^{{.*}}, ^{{.*}}){{$}}
// CHECK:       aie.dma_start(MM2S, 1, ^{{.*}}, ^{{.*}}) {aiex.ctrl_pkt_gated}
module {
  aie.device(npu2) @static_shim {
    %t00 = aie.tile(0, 0)
    %t10 = aie.tile(1, 0)
    %t01 = aie.tile(0, 1)
    aie.flow(%t00, DMA : 0, %t01, DMA : 0)
    aie.flow(%t10, DMA : 0, %t01, DMA : 1)
    %src = aie.external_buffer : memref<16xi32>
    %shim = aie.shim_dma(%t00) {
      // expected-remark @+1 {{MM2S channel 0 of tile (0, 0) not proved gated: its first buffer descriptor does not acquire a lock}}
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %buf = aie.buffer(%t01) : memref<16xi32>
    %c0 = aie.lock(%t01, 0) {init = 0 : i32}
    %c1 = aie.lock(%t01, 1) {init = 0 : i32}
    %dma = aie.memtile_dma(%t01) {
      %0 = aie.dma_start(S2MM, 0, ^in0, ^s1)
    ^in0:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      %v16 = arith.constant 1 : i32
      aie.use_lock(%c0, Release, %v16)
      aie.next_bd ^end
    ^s1:
      %1 = aie.dma_start(S2MM, 1, ^in1, ^o0)
    ^in1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      %v17 = arith.constant 1 : i32
      aie.use_lock(%c1, Release, %v17)
      aie.next_bd ^end
    ^o0:
      // expected-remark @+1 {{MM2S channel 0 of tile (0, 1) not proved gated: its first buffer descriptor does not acquire a lock}}
      %2 = aie.dma_start(MM2S, 0, ^out0, ^o1)
    ^out0:
      %v18 = arith.constant 1 : i32
      aie.use_lock(%c0, AcquireGreaterEqual, %v18)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^o1:
      %3 = aie.dma_start(MM2S, 1, ^out1, ^end)
    ^out1:
      %v19 = arith.constant 1 : i32
      aie.use_lock(%c1, AcquireGreaterEqual, %v19)
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// (0, 2) writes its own scratch buffer, then waits on a lock only an S2MM
// descriptor fed by the host's shim channel releases, so it is proved. (0, 3)
// writes (0, 2)'s buffer before its gate. (0, 4) waits on a lock fed from a
// self-starting memory tile channel. (0, 5) is given as an ELF. (1, 2) calls a
// kernel before its gate and sends on a stream.

// CHECK-LABEL: @cores
// CHECK:       aie.core(%{{.*}}) {
// CHECK:       } {aiex.ctrl_pkt_gated}
// CHECK-NOT:   aiex.ctrl_pkt_gated
module {
  aie.device(npu2) @cores {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    %t12 = aie.tile(1, 2)
    aie.flow(%t00, DMA : 0, %t02, DMA : 0)
    aie.flow(%t01, DMA : 0, %t04, DMA : 0)
    aie.flow(%t12, Core : 0, %t02, DMA : 1)
    func.func private @kernel(memref<16xi32>)

    %in = aie.buffer(%t02) : memref<16xi32>
    %scratch = aie.buffer(%t02) : memref<16xi32>
    %full = aie.lock(%t02, 0)
    %core02 = aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : i32
      memref.store %c7, %scratch[%c0] : memref<16xi32>
      %v1 = arith.constant 1 : i32
      aie.use_lock(%full, AcquireGreaterEqual, %v1)
      aie.end
    }
    %mem02 = aie.mem(%t02) {
      %0 = aie.dma_start(S2MM, 0, ^in, ^end)
    ^in:
      aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
      %v1 = arith.constant 1 : i32
      aie.use_lock(%full, Release, %v1)
      aie.next_bd ^in
    ^end:
      aie.end
    }

    %nb = aie.buffer(%t02) : memref<16xi32>
    %l3 = aie.lock(%t03, 0)
    // expected-remark @+1 {{core (0, 3) not proved gated: an access to another tile's buffer before the gating acquire}}
    %core03 = aie.core(%t03) {
      %c0 = arith.constant 0 : index
      %c7 = arith.constant 7 : i32
      memref.store %c7, %nb[%c0] : memref<16xi32>
      %v1 = arith.constant 1 : i32
      aie.use_lock(%l3, AcquireGreaterEqual, %v1)
      aie.end
    }

    %src = aie.buffer(%t01) : memref<16xi32>
    %go = aie.lock(%t01, 0) {init = 1 : i32}
    %dma01 = aie.memtile_dma(%t01) {
      // expected-remark @+1 {{MM2S channel 0 of tile (0, 1) not proved gated: the lock's initial value satisfies its first buffer descriptor's acquire}}
      %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      %v1 = arith.constant 1 : i32
      aie.use_lock(%go, AcquireGreaterEqual, %v1)
      aie.dma_bd(%src : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b4 = aie.buffer(%t04) : memref<16xi32>
    %l4 = aie.lock(%t04, 0)
    // expected-remark @+1 {{core (0, 4) not proved gated: the lock's initial value satisfies its first buffer descriptor's acquire}}
    %core04 = aie.core(%t04) {
      %v1 = arith.constant 1 : i32
      aie.use_lock(%l4, AcquireGreaterEqual, %v1)
      aie.end
    }
    %mem04 = aie.mem(%t04) {
      %0 = aie.dma_start(S2MM, 0, ^in, ^end)
    ^in:
      aie.dma_bd(%b4 : memref<16xi32> offset = 0 len = 16)
      %v1 = arith.constant 1 : i32
      aie.use_lock(%l4, Release, %v1)
      aie.next_bd ^in
    ^end:
      aie.end
    }

    // expected-remark @+1 {{core (0, 5) not proved gated: a core given as elf_file}}
    %core05 = aie.core(%t05) {
      aie.end
    } {elf_file = "core_0_5.elf"}

    %b12 = aie.buffer(%t12) : memref<16xi32>
    %l12 = aie.lock(%t12, 0)
    // expected-remark @+1 {{core (1, 2) not proved gated: a call before the gating acquire on a core that sends on a stream or cascade}}
    %core12 = aie.core(%t12) {
      func.call @kernel(%b12) : (memref<16xi32>) -> ()
      %v1 = arith.constant 1 : i32
      aie.use_lock(%l12, AcquireGreaterEqual, %v1)
      aie.end
    }
  }
}

// -----

// A loop lowered to branches. The walk follows a branch whose condition the
// values on the path decide (the loop's first test), so (0, 2) is proved; (0,
// 3)'s bound is loaded from memory, so its loop may not run.

// CHECK-LABEL: @lowered_loop
// CHECK:       %{{.*}} = aie.core(%{{.*}}) {
// CHECK:       } {aiex.ctrl_pkt_gated}
// CHECK-NOT:   aiex.ctrl_pkt_gated
module {
  aie.device(npu2) @lowered_loop {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t00, DMA : 0, %t02, DMA : 0)
    %in = aie.buffer(%t02) : memref<16xi32>
    %full = aie.lock(%t02, 0)
    %core02 = aie.core(%t02) {
      %c0 = arith.constant 0 : i32
      %c8 = arith.constant 8 : i32
      %c1 = arith.constant 1 : i32
      cf.br ^head(%c0 : i32)
    ^head(%i: i32):
      %go = arith.cmpi slt, %i, %c8 : i32
      cf.cond_br %go, ^body, ^exit
    ^body:
      aie.use_lock(%full, AcquireGreaterEqual, %c1)
      %n = arith.addi %i, %c1 : i32
      cf.br ^head(%n : i32)
    ^exit:
      aie.end
    }
    %mem02 = aie.mem(%t02) {
      %0 = aie.dma_start(S2MM, 0, ^in, ^end)
    ^in:
      aie.dma_bd(%in : memref<16xi32> offset = 0 len = 16)
      %v1 = arith.constant 1 : i32
      aie.use_lock(%full, Release, %v1)
      aie.next_bd ^in
    ^end:
      aie.end
    }
    %bound = aie.buffer(%t03) : memref<1xi32>
    %l3 = aie.lock(%t03, 0)
    // expected-remark @+1 {{core (0, 3) not proved gated: a branch that may not be taken before the gating acquire}}
    %core03 = aie.core(%t03) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %z = arith.constant 0 : index
      %n = memref.load %bound[%z] : memref<1xi32>
      cf.br ^head(%c0 : i32)
    ^head(%i: i32):
      %go = arith.cmpi slt, %i, %n : i32
      cf.cond_br %go, ^body, ^exit
    ^body:
      aie.use_lock(%l3, AcquireGreaterEqual, %c1)
      cf.br ^exit
    ^exit:
      aie.end
    }
  }
}
