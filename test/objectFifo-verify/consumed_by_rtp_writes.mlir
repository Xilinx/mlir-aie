//===- consumed_by_rtp_writes.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify %s | FileCheck %s

// The trip count is the RTP word @seq writes. Any other raw write that may
// reach the RTP buffer's tile may rewrite it, so the row length is unknown and
// angle is not proven consumed; one that lands on another tile cannot.

// A write32 by column and row onto the core tile.
// CHECK-LABEL: module @rtp_write32_col_row
// CHECK: aie.route_endpoint @angle__prod__dma({{.*}}) DMA {fifoName = "angle"}
module @rtp_write32_col_row {
  aie.device(npu2) {
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
    aie.runtime_sequence @seq(%a: memref<128xbf16>) {
      %c24 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp, 0, %c24) : i32
      %c1 = arith.constant 1 : i32
      aiex.set_lock(%barrier, %c1)
    }
    aie.runtime_sequence @other(%a: memref<128xbf16>, %where: i32) {
      %addr = arith.constant 1024 : i32
      %v = arith.constant 96 : i32
      aiex.npu.write32(%addr, %v) {column = 0 : i32, row = 2 : i32} : i32, i32
    }
  }
}

// -----

// A maskwrite32 by column and row onto the core tile.
// CHECK-LABEL: module @rtp_maskwrite32_col_row
// CHECK: aie.route_endpoint @angle__prod__dma({{.*}}) DMA {fifoName = "angle"}
module @rtp_maskwrite32_col_row {
  aie.device(npu2) {
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
    aie.runtime_sequence @seq(%a: memref<128xbf16>) {
      %c24 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp, 0, %c24) : i32
      %c1 = arith.constant 1 : i32
      aiex.set_lock(%barrier, %c1)
    }
    aie.runtime_sequence @other(%a: memref<128xbf16>, %where: i32) {
      %addr = arith.constant 1024 : i32
      %v = arith.constant 96 : i32
      %m = arith.constant 255 : i32
      aiex.npu.maskwrite32(%addr, %v, %m) {column = 0 : i32, row = 2 : i32} : i32, i32, i32
    }
  }
}

// -----

// A blockwrite by column and row onto the core tile.
// CHECK-LABEL: module @rtp_blockwrite_col_row
// CHECK: aie.route_endpoint @angle__prod__dma({{.*}}) DMA {fifoName = "angle"}
module @rtp_blockwrite_col_row {
  aie.device(npu2) {
    memref.global "private" constant @word : memref<1xi32> = dense<96>
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
    aie.runtime_sequence @seq(%a: memref<128xbf16>) {
      %c24 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp, 0, %c24) : i32
      %c1 = arith.constant 1 : i32
      aiex.set_lock(%barrier, %c1)
    }
    aie.runtime_sequence @other(%a: memref<128xbf16>, %where: i32) {
      %w = memref.get_global @word : memref<1xi32>
      aiex.npu.blockwrite(%w) {address = 1024 : ui32, column = 0 : i32, row = 2 : i32} : memref<1xi32>
    }
  }
}

// -----

// A DMA task pushed onto the core tile's channel may write any of its memory.
// CHECK-LABEL: module @rtp_push_queue
// CHECK: aie.route_endpoint @angle__prod__dma({{.*}}) DMA {fifoName = "angle"}
module @rtp_push_queue {
  aie.device(npu2) {
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
    aie.runtime_sequence @seq(%a: memref<128xbf16>) {
      %c24 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp, 0, %c24) : i32
      %c1 = arith.constant 1 : i32
      aiex.set_lock(%barrier, %c1)
    }
    aie.runtime_sequence @other(%a: memref<128xbf16>, %where: i32) {
      %bd = arith.constant 0 : i32
      aiex.npu.push_queue(0, 2, S2MM : 0) bd_id %bd repeat %bd {issue_token = false} : i32, i32
    }
  }
}

// -----

// A write32 whose address the host passes in may land anywhere.
// CHECK-LABEL: module @rtp_write32_unfixed
// CHECK: aie.route_endpoint @angle__prod__dma({{.*}}) DMA {fifoName = "angle"}
module @rtp_write32_unfixed {
  aie.device(npu2) {
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
    aie.runtime_sequence @seq(%a: memref<128xbf16>) {
      %c24 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp, 0, %c24) : i32
      %c1 = arith.constant 1 : i32
      aiex.set_lock(%barrier, %c1)
    }
    aie.runtime_sequence @other(%a: memref<128xbf16>, %where: i32) {
      %v = arith.constant 96 : i32
      aiex.npu.write32(%where, %v) : i32, i32
    }
  }
}

// -----

// A write32 whose absolute address names the core tile (column 0, row 2).
// CHECK-LABEL: module @rtp_write32_absolute
// CHECK: aie.route_endpoint @angle__prod__dma({{.*}}) DMA {fifoName = "angle"}
module @rtp_write32_absolute {
  aie.device(npu2) {
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
    aie.runtime_sequence @seq(%a: memref<128xbf16>) {
      %c24 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp, 0, %c24) : i32
      %c1 = arith.constant 1 : i32
      aiex.set_lock(%barrier, %c1)
    }
    aie.runtime_sequence @other(%a: memref<128xbf16>, %where: i32) {
      %addr = arith.constant 2098176 : i32
      %v = arith.constant 96 : i32
      aiex.npu.write32(%addr, %v) : i32, i32
    }
  }
}

// -----

// A write32, a blockwrite and a push_queue on another core tile leave the
// RTP word alone, so angle is proven consumed.
// CHECK-LABEL: module @rtp_writes_elsewhere
// CHECK: aie.route_endpoint @angle__prod__dma({{.*}}) DMA {consumed_by = [#aie.consumed_by<output = @out__cons__dma, input_bytes = 1, output_bytes = 24>], fifoName = "angle"}
module @rtp_writes_elsewhere {
  aie.device(npu2) {
    memref.global "private" constant @word : memref<1xi32> = dense<96>
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
    aie.runtime_sequence @seq(%a: memref<128xbf16>) {
      %c24 = arith.constant 24 : i32
      aiex.npu.rtp_write(@rtp, 0, %c24) : i32
      %c1 = arith.constant 1 : i32
      aiex.set_lock(%barrier, %c1)
    }
    aie.runtime_sequence @other(%a: memref<128xbf16>, %where: i32) {
      %addr = arith.constant 1024 : i32
      %v = arith.constant 96 : i32
      aiex.npu.write32(%addr, %v) {column = 0 : i32, row = 3 : i32} : i32, i32
      %w = memref.get_global @word : memref<1xi32>
      aiex.npu.blockwrite(%w) {address = 1024 : ui32, column = 1 : i32, row = 2 : i32} : memref<1xi32>
      %bd = arith.constant 0 : i32
      aiex.npu.push_queue(0, 3, S2MM : 0) bd_id %bd repeat %bd {issue_token = false} : i32, i32
    }
  }
}
