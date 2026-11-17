//===- write_teardown.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: split-file %s %t
// RUN: aie-opt --aie-annotate-core-lock-acquires %t/config.mlir -o %t/annotated.mlir
// RUN: FileCheck %s --check-prefix=ANNOTATE < %t/annotated.mlir
// RUN: aie-opt --aie-annotate-core-lock-acquires %t/annotated.mlir -o %t/annotated.twice.mlir
// RUN: diff %t/annotated.mlir %t/annotated.twice.mlir
// RUN: aie-opt --aie-expand-load-pdi="teardown=true" %t/config.mlir | FileCheck %s

// The NPU translation, cert and C++ TXN lowerings write the ops inline-config
// writes.
// RUN: aie-opt --aie-expand-load-pdi="teardown=true" --mlir-print-debuginfo %t/config.mlir -o %t.lazy.mlir
// RUN: aie-opt --aie-expand-load-pdi="teardown=true inline-config=true" --mlir-print-debuginfo %t/config.mlir -o %t.inline.mlir
// RUN: FileCheck %s --check-prefix=LOCKS < %t.inline.mlir
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=seq --aie-npu-emit-locmap=%t.lazy.json %t.lazy.mlir -o %t.lazy.bin
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=seq --aie-npu-emit-locmap=%t.inline.json %t.inline.mlir -o %t.inline.bin
// RUN: cmp %t.lazy.bin %t.inline.bin
// RUN: cmp %t.lazy.json %t.inline.json
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=lock_seq --aie-npu-emit-locmap=%t.lock.lazy.json %t.lazy.mlir -o %t.lock.lazy.bin
// RUN: aie-translate --aie-npu-to-binary --aie-output-binary --aie-device-name=main --aie-sequence-name=lock_seq --aie-npu-emit-locmap=%t.lock.inline.json %t.inline.mlir -o %t.lock.inline.bin
// RUN: cmp %t.lock.lazy.bin %t.lock.inline.bin
// RUN: cmp %t.lock.lazy.json %t.lock.inline.json
// RUN: aie-opt --aie-npu-to-cert %t.lazy.mlir -o %t.lazy.cert.mlir
// RUN: aie-opt --aie-npu-to-cert %t.inline.mlir -o %t.inline.cert.mlir
// RUN: diff %t.lazy.cert.mlir %t.inline.cert.mlir
// RUN: aie-opt --convert-aiex-to-emitc %t.lazy.mlir -o %t.lazy.cpp.mlir
// RUN: aie-opt --convert-aiex-to-emitc %t.inline.mlir -o %t.inline.cpp.mlir
// RUN: diff %t.lazy.cpp.mlir %t.inline.cpp.mlir
// RUN: aie-opt --aie-expand-load-pdi="teardown=true inline-config=true" --split-input-file --verify-diagnostics %t/locks.mlir | FileCheck %s --check-prefix=ACQUIRES
// RUN: aie-opt %t/summary.mlir -o %t/summary.once.mlir
// RUN: aie-opt --aie-annotate-core-lock-acquires %t/summary.once.mlir -o %t/summary.twice.mlir
// RUN: diff %t/summary.once.mlir %t/summary.twice.mlir
// RUN: FileCheck %s --check-prefix=SUMMARY < %t/summary.twice.mlir
// RUN: aie-opt --split-input-file --verify-diagnostics %t/summary-invalid.mlir -o %t/summary-invalid.out
// RUN: aie-opt --aie-expand-load-pdi="teardown=true" %t/stop-ctrlpkt.mlir | FileCheck %s --check-prefix=STOP
// RUN: aie-opt --aie-annotate-core-lock-acquires %t/opaque-llvm.mlir | FileCheck %s --check-prefix=OPAQUE
// RUN: aie-opt --aie-expand-load-pdi="teardown=true inline-config=true" --verify-diagnostics %t/opaque-llvm.mlir | FileCheck %s --check-prefix=OPAQUE-DRAIN

// ANNOTATE-LABEL: aie.device(npu2_1col) @lock_cfg
// ANNOTATE: lock_acquires = #aie.core_lock_acquires<complete = true, acquires = [<column = 0, row = 2, lock_id = 5, action = AcquireGreaterEqual, value = 1>, <column = 0, row = 2, lock_id = 5, action = Acquire, value = 3>, <column = 0, row = 2, lock_id = 6, action = AcquireGreaterEqual, value = 2>, <column = 0, row = 2, lock_id = 6, action = AcquireGreaterEqual, value = 3>]>

// CHECK-LABEL: aie.runtime_sequence @seq
// CHECK:      aiex.npu.load_pdi {device_ref = @empty_0
// CHECK-NEXT: aiex.npu.write_config @cfg
// CHECK-NEXT: aiex.npu.write32
// CHECK-NEXT: aiex.npu.write_teardown @cfg
// CHECK-NEXT: aiex.npu.write_config @cfg
// CHECK-NEXT: aiex.npu.write_teardown @cfg
// CHECK-NEXT: {{^}}    }

// LOCKS-LABEL: aie.runtime_sequence @lock_seq
// LOCKS-DAG: %[[FIRST_LOCK:.*]] = arith.constant 2224208 : i32
// LOCKS-DAG: %[[SECOND_LOCK:.*]] = arith.constant 2224224 : i32
// LOCKS-DAG: %[[STATUS:.*]] = arith.constant 2301956 : i32
// LOCKS-DAG: %[[STALL_MASK:.*]] = arith.constant 960 : i32
// LOCKS-DAG: %[[THREE:.*]] = arith.constant 3 : i32
// LOCKS-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// LOCKS-DAG: %[[ENABLE:.*]] = arith.constant 1 : i32
// LOCKS-DAG: %[[MARKER:.*]] = arith.constant 123 : i32
// LOCKS: aiex.npu.write32(%{{.*}}, %[[MARKER]])
// LOCKS: aiex.npu.maskwrite32
// LOCKS-NEXT: aiex.npu.maskpoll(%[[STATUS]], %[[ZERO]], %[[ENABLE]])
// LOCKS-NEXT: aiex.npu.write32(%[[FIRST_LOCK]], %[[THREE]])
// LOCKS-NEXT: aiex.npu.write32(%[[SECOND_LOCK]], %[[THREE]])
// LOCKS-NEXT: aiex.npu.maskpoll(%[[STATUS]], %[[ZERO]], %[[STALL_MASK]])

//--- config.mlir
module {
  aie.device(npu2_1col) @cfg {
    %mem = aie.tile(0, 1)
    %compute = aie.tile(0, 2)
    aie.switchbox(%compute) {
      aie.connect<South : 1, DMA : 0>
    }
    %buf = aie.buffer(%mem) {address = 0 : i32, sym_name = "buf"} : memref<16xi32>
    %memdma = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2)
    ^bb1:
      aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }

  aie.device(npu2_1col) @main {
    aie.runtime_sequence @seq(%arg0: memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %a = arith.constant 119316 : i32
      aiex.npu.load_pdi {device_ref = @cfg}
      aiex.npu.write32(%a, %c0) : i32, i32
      aiex.npu.load_pdi {device_ref = @cfg}
    }
    aie.runtime_sequence @lock_seq() {
      %marker = arith.constant 123 : i32
      %address = arith.constant 119316 : i32
      aiex.npu.load_pdi {device_ref = @lock_cfg}
      aiex.npu.write32(%address, %marker) : i32, i32
      aiex.npu.load_pdi {device_ref = @lock_cfg}
    }
  }

  aie.device(npu2_1col) @lock_cfg {
    %tile = aie.tile(0, 2)
    %first = aie.lock(%tile, 5) {init = 0 : i32}
    %second = aie.lock(%tile, 6) {init = 0 : i32}
    aie.core(%tile) {
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      %three = arith.constant 3 : i32
      aie.use_lock(%first, AcquireGreaterEqual, %one)
      aie.use_lock(%first, Acquire, %three)
      aie.use_lock(%second, AcquireGreaterEqual, %two)
      aie.use_lock(%second, AcquireGreaterEqual, %three)
      aie.end
    }
  }
}

//--- locks.mlir

// ACQUIRES-LABEL: aie.runtime_sequence @incompatible
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %tile = aie.tile(0, 2)
    %first = aie.lock(%tile, 5) {init = 0 : i32}
    %second = aie.lock(%tile, 6) {init = 0 : i32}
    // expected-warning@+2 {{teardown cannot drain lock 5 on tile (0, 2): no single writable value satisfies every acquire}}
    // expected-warning@+1 {{teardown cannot drain lock 6 on tile (0, 2): no single writable value satisfies every acquire}}
    aie.core(%tile) {
      %one = arith.constant 1 : i32
      %two = arith.constant 2 : i32
      aie.use_lock(%first, Acquire, %one)
      aie.use_lock(%first, Acquire, %two)
      aie.use_lock(%second, Acquire, %one)
      aie.use_lock(%second, AcquireGreaterEqual, %two)
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @incompatible() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// ACQUIRES-LABEL: aie.runtime_sequence @neighbor
// ACQUIRES-DAG: %[[LOCK:.*]] = arith.constant 3272784 : i32
// ACQUIRES-DAG: %[[STATUS:.*]] = arith.constant 2301956 : i32
// ACQUIRES-DAG: %[[MASK:.*]] = arith.constant 960 : i32
// ACQUIRES-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// ACQUIRES-DAG: %[[FOUR:.*]] = arith.constant 4 : i32
// ACQUIRES-DAG: %[[ENABLE:.*]] = arith.constant 1 : i32
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES: aiex.npu.maskwrite32
// ACQUIRES: aiex.npu.maskpoll(%[[STATUS]], %[[ZERO]], %[[ENABLE]])
// ACQUIRES-NEXT: aiex.npu.write32(%[[LOCK]], %[[FOUR]])
// ACQUIRES-NEXT: aiex.npu.maskpoll(%[[STATUS]], %[[ZERO]], %[[MASK]])
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %core_tile = aie.tile(0, 2)
    %lock_tile = aie.tile(0, 3)
    %lock = aie.lock(%lock_tile, 5) {init = 0 : i32}
    aie.core(%core_tile) {
      %one = arith.constant 1 : i32
      %four = arith.constant 4 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %one)
      aie.use_lock(%lock, AcquireGreaterEqual, %four)
      aie.use_lock(%lock, AcquireGreaterEqual, %four)
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @neighbor() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// ACQUIRES-LABEL: aie.runtime_sequence @shared
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %first_tile = aie.tile(0, 2)
    %second_tile = aie.tile(0, 3)
    %lock = aie.lock(%second_tile, 5) {init = 0 : i32}
    // expected-warning@+1 {{multiple cores may consume the supplied count}}
    aie.core(%first_tile) {
      %one = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %one)
      aie.end
    }
    // expected-warning@+1 {{multiple cores may consume the supplied count}}
    aie.core(%second_tile) {
      %three = arith.constant 3 : i32
      aie.use_lock(%lock, Acquire, %three)
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @shared() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// ACQUIRES-LABEL: aie.runtime_sequence @zero_acquire
// ACQUIRES-DAG: %[[LOCK:.*]] = arith.constant 2224208 : i32
// ACQUIRES-DAG: %[[CONTROL:.*]] = arith.constant 2301952 : i32
// ACQUIRES-DAG: %[[STATUS:.*]] = arith.constant 2301956 : i32
// ACQUIRES-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// ACQUIRES-DAG: %[[MASK:.*]] = arith.constant 960 : i32
// ACQUIRES-DAG: %[[ENABLE:.*]] = arith.constant 1 : i32
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES: aiex.npu.maskwrite32(%[[CONTROL]], %[[ZERO]], %[[ENABLE]])
// ACQUIRES-NEXT: aiex.npu.maskpoll(%[[STATUS]], %[[ZERO]], %[[ENABLE]])
// ACQUIRES-NEXT: aiex.npu.write32(%[[LOCK]], %[[ZERO]])
// ACQUIRES-NEXT: aiex.npu.maskpoll(%[[STATUS]], %[[ZERO]], %[[MASK]])
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 5) {init = 0 : i32}
    aie.core(%tile) {
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %zero)
      aie.use_lock(%lock, Release, %one)
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @zero_acquire() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// ACQUIRES-LABEL: aie.runtime_sequence @multiple_cores
// ACQUIRES-DAG: %[[FIRST_LOCK:.*]] = arith.constant 2224208 : i32
// ACQUIRES-DAG: %[[SECOND_LOCK:.*]] = arith.constant 3272784 : i32
// ACQUIRES-DAG: %[[FIRST_STATUS:.*]] = arith.constant 2301956 : i32
// ACQUIRES-DAG: %[[SECOND_STATUS:.*]] = arith.constant 3350532 : i32
// ACQUIRES-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// ACQUIRES-DAG: %[[THREE:.*]] = arith.constant 3 : i32
// ACQUIRES-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// ACQUIRES-DAG: %[[MASK:.*]] = arith.constant 960 : i32
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES: aiex.npu.maskwrite32(%{{.*}}, %[[ZERO]], %[[ONE]])
// ACQUIRES-NEXT: aiex.npu.maskwrite32(%{{.*}}, %[[ZERO]], %[[ONE]])
// ACQUIRES-NEXT: aiex.npu.maskpoll(%[[FIRST_STATUS]], %[[ZERO]], %[[ONE]])
// ACQUIRES-NEXT: aiex.npu.maskpoll(%[[SECOND_STATUS]], %[[ZERO]], %[[ONE]])
// ACQUIRES-NEXT: aiex.npu.write32(%[[FIRST_LOCK]], %[[ONE]])
// ACQUIRES-NEXT: aiex.npu.write32(%[[SECOND_LOCK]], %[[THREE]])
// ACQUIRES-NEXT: aiex.npu.maskpoll(%[[FIRST_STATUS]], %[[ZERO]], %[[MASK]])
// ACQUIRES-NEXT: aiex.npu.maskpoll(%[[SECOND_STATUS]], %[[ZERO]], %[[MASK]])
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %first_tile = aie.tile(0, 2)
    %second_tile = aie.tile(0, 3)
    %first = aie.lock(%first_tile, 5) {init = 0 : i32}
    %second = aie.lock(%second_tile, 5) {init = 0 : i32}
    aie.core(%first_tile) {
      %one = arith.constant 1 : i32
      aie.use_lock(%first, AcquireGreaterEqual, %one) {acq_en = false}
      aie.end
    }
    aie.core(%second_tile) {
      %three = arith.constant 3 : i32
      aie.use_lock(%second, Acquire, %three)
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @multiple_cores() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// ACQUIRES-LABEL: aie.runtime_sequence @dma_acquire
// ACQUIRES-DAG: %[[LOCK:.*]] = arith.constant 2224208 : i32
// ACQUIRES-DAG: %[[THREE:.*]] = arith.constant 3 : i32
// ACQUIRES-DAG: %[[STATUS:.*]] = arith.constant 2301956 : i32
// ACQUIRES-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// ACQUIRES-DAG: %[[MASK:.*]] = arith.constant 960 : i32
// ACQUIRES-DAG: %[[ENABLE:.*]] = arith.constant 1 : i32
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES: aiex.npu.maskwrite32
// ACQUIRES: aiex.npu.maskpoll(%[[STATUS]], %[[ZERO]], %[[ENABLE]])
// ACQUIRES-NEXT: aiex.npu.write32(%[[LOCK]], %[[THREE]])
// ACQUIRES-NEXT: aiex.npu.maskpoll(%[[STATUS]], %[[ZERO]], %[[MASK]])
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 5) {init = 0 : i32}
    %buffer = aie.buffer(%tile) {address = 0 : i32} : memref<16xi32>
    aie.core(%tile) {
      %one = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %one)
      aie.end
    }
    aie.mem(%tile) {
      %three = arith.constant 3 : i32
      aie.dma_start(MM2S, 0, ^transfer, ^end)
    ^transfer:
      aie.use_lock(%lock, Acquire, %three)
      aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.use_lock(%lock, Release, %three)
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @dma_acquire() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// ACQUIRES-LABEL: aie.runtime_sequence @dynamic_acquire
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 5) {init = 0 : i32}
    %buffer = aie.buffer(%tile) {address = 0 : i32} : memref<1xi32>
    // expected-warning@+1 {{teardown cannot determine every lock acquire}}
    aie.core(%tile) {
      %zero = arith.constant 0 : index
      %value = memref.load %buffer[%zero] : memref<1xi32>
      aie.use_lock(%lock, AcquireGreaterEqual, %value)
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @dynamic_acquire() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// ACQUIRES-LABEL: aie.runtime_sequence @opaque_callee
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %first_tile = aie.tile(0, 2)
    %second_tile = aie.tile(0, 3)
    %lock = aie.lock(%second_tile, 5) {init = 0 : i32}
    func.func private @kernel()
    // expected-warning@+1 {{teardown cannot determine every lock acquire}}
    aie.core(%first_tile) {
      func.call @kernel() : () -> ()
      aie.end
    }
    aie.core(%second_tile) {
      %one = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %one)
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @opaque_callee() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

// -----

// ACQUIRES-LABEL: aie.runtime_sequence @counter_range
// ACQUIRES: aiex.npu.load_pdi
// ACQUIRES-NOT: aiex.npu.maskpoll
// ACQUIRES: {{^    }}}

module {
  aie.device(npu2_1col) @cfg {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 5) {init = 0 : i32}
    // expected-warning@+1 {{no single writable value satisfies every acquire}}
    aie.core(%tile) {
      %value = arith.constant 64 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %value)
      aie.end
    }
  }
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @counter_range() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

//--- opaque-llvm.mlir

// OPAQUE-LABEL: aie.device(npu2_2col) @cfg
// OPAQUE-COUNT-4: lock_acquires = #aie.core_lock_acquires<complete = false, acquires = []>
// OPAQUE: lock_acquires = #aie.core_lock_acquires<complete = true, acquires = [<column = 1, row = 2, lock_id = 5, action = AcquireGreaterEqual, value = 1>]>
// OPAQUE-DRAIN-LABEL: aie.runtime_sequence @opaque_llvm
// OPAQUE-DRAIN: aiex.npu.load_pdi
// OPAQUE-DRAIN-NOT: aiex.npu.maskpoll
// OPAQUE-DRAIN: {{^    }}}

module {
  aie.device(npu2_2col) @cfg {
    llvm.func @kernel()
    func.func private @call_kernel() {
      llvm.call @kernel() : () -> ()
      return
    }
    func.func private @assembly() {
      llvm.inline_asm has_side_effects "", "" : () -> ()
      return
    }
    %direct_call = aie.tile(0, 2)
    %direct_assembly = aie.tile(0, 3)
    %callee_call = aie.tile(0, 4)
    %callee_assembly = aie.tile(0, 5)
    %known_acquire = aie.tile(1, 2)
    %lock = aie.lock(%known_acquire, 5) {init = 0 : i32}
    // expected-warning@+1 {{teardown cannot determine every lock acquire}}
    aie.core(%direct_call) {
      llvm.call @kernel() : () -> ()
      aie.end
    }
    // expected-warning@+1 {{teardown cannot determine every lock acquire}}
    aie.core(%direct_assembly) {
      llvm.inline_asm has_side_effects "", "" : () -> ()
      aie.end
    }
    // expected-warning@+1 {{teardown cannot determine every lock acquire}}
    aie.core(%callee_call) {
      func.call @call_kernel() : () -> ()
      aie.end
    }
    // expected-warning@+1 {{teardown cannot determine every lock acquire}}
    aie.core(%callee_assembly) {
      func.call @assembly() : () -> ()
      aie.end
    }
    aie.core(%known_acquire) {
      %one = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %one)
      aie.end
    }
  }
  aie.device(npu2_2col) @main {
    aie.runtime_sequence @opaque_llvm() {
      aiex.npu.load_pdi {device_ref = @cfg}
    }
  }
}

//--- summary.mlir

// SUMMARY-LABEL: module @summaries
// SUMMARY: elf_file = "acquires.elf"
// SUMMARY-SAME: lock_acquires = #aie.core_lock_acquires<complete = true, acquires = [<column = 0, row = 3, lock_id = 5, action = AcquireGreaterEqual, value = 1>, <column = 0, row = 3, lock_id = 5, action = Acquire, value = 3>]>
// SUMMARY: elf_file = "empty.elf"
// SUMMARY-SAME: lock_acquires = #aie.core_lock_acquires<complete = true, acquires = []>
// SUMMARY: elf_file = "partial.elf"
// SUMMARY-SAME: lock_acquires = #aie.core_lock_acquires<complete = false, acquires = [<column = 0, row = 4, lock_id = 1, action = Acquire, value = 0>]>
// SUMMARY: elf_file = "unknown.elf"
// SUMMARY-SAME: lock_acquires = #aie.core_lock_acquires<complete = false, acquires = []>

module @summaries {
  aie.device(npu2_1col) {
    %first = aie.tile(0, 2)
    %second = aie.tile(0, 3)
    %third = aie.tile(0, 4)
    %fourth = aie.tile(0, 5)
    aie.core(%first) {
      aie.end
    } {elf_file = "acquires.elf", lock_acquires = #aie.core_lock_acquires<complete = true, acquires = [<column = 0, row = 3, lock_id = 5, action = AcquireGreaterEqual, value = 1>, <column = 0, row = 3, lock_id = 5, action = Acquire, value = 3>]>}
    aie.core(%second) {
      aie.end
    } {elf_file = "empty.elf", lock_acquires = #aie.core_lock_acquires<complete = true, acquires = []>}
    aie.core(%third) {
      aie.end
    } {elf_file = "partial.elf", lock_acquires = #aie.core_lock_acquires<complete = false, acquires = [<column = 0, row = 4, lock_id = 1, action = Acquire, value = 0>]>}
    aie.core(%fourth) {
      aie.end
    } {elf_file = "unknown.elf", lock_acquires = #aie.core_lock_acquires<complete = false, acquires = []>}
  }
}

//--- summary-invalid.mlir

// expected-error@+1 {{expected Acquire or AcquireGreaterEqual, got Release}}
module attributes {invalid = #aie.lock_acquire<column = 0, row = 2, lock_id = 5, action = Release, value = 1>} {}

// -----

// expected-error@+1 {{lock column, row, and ID must be non-negative}}
module attributes {invalid = #aie.lock_acquire<column = -1, row = 2, lock_id = 5, action = Acquire, value = 1>} {}

// -----

// expected-error@+1 {{lock column, row, and ID must be non-negative}}
module attributes {invalid = #aie.lock_acquire<column = 0, row = 2, lock_id = -1, action = Acquire, value = 1>} {}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    // expected-error@+3 {{attribute 'lock_acquires' failed to satisfy constraint}}
    aie.core(%tile) {
      aie.end
    } {lock_acquires = [array<i32: 0, 2, 5, 1, 1>]}
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    // expected-error@+1 {{lock_acquires references a tile outside the device: (1, 2)}}
    aie.core(%tile) {
      aie.end
    } {lock_acquires = #aie.core_lock_acquires<complete = true, acquires = [<column = 1, row = 2, lock_id = 5, action = Acquire, value = 1>]>}
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    // expected-error@+1 {{lock_acquires references a tile outside the device: (0, 6)}}
    aie.core(%tile) {
      aie.end
    } {lock_acquires = #aie.core_lock_acquires<complete = true, acquires = [<column = 0, row = 6, lock_id = 5, action = Acquire, value = 1>]>}
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %tile = aie.tile(0, 2)
    // expected-error@+1 {{lock_acquires references an invalid lock ID 16 on tile (0, 2)}}
    aie.core(%tile) {
      aie.end
    } {lock_acquires = #aie.core_lock_acquires<complete = true, acquires = [<column = 0, row = 2, lock_id = 16, action = Acquire, value = 1>]>}
  }
}

// -----

module {
  aie.device(xcvc1902) {
    %tile = aie.tile(0, 2)
    // expected-error@+1 {{lock_acquires uses AcquireGreaterEqual on a target without semaphore locks}}
    aie.core(%tile) {
      aie.end
    } {lock_acquires = #aie.core_lock_acquires<complete = true, acquires = [<column = 0, row = 2, lock_id = 5, action = AcquireGreaterEqual, value = 1>]>}
  }
}

//--- stop-ctrlpkt.mlir

// STOP-LABEL: aie.runtime_sequence()
// STOP: aiex.npu.sync
// STOP-NEXT: aiex.control_packet {address = 2301952 : ui32, data = array<i32: 0>
// STOP-NEXT: aiex.control_packet {address = 3350528 : ui32, data = array<i32: 0>
// STOP: %[[FIRST_STATUS:.*]] = arith.constant 2301956 : i32
// STOP: %[[ZERO:.*]] = arith.constant 0 : i32
// STOP: %[[ENABLE:.*]] = arith.constant 1 : i32
// STOP-NEXT: aiex.npu.maskpoll(%[[FIRST_STATUS]], %[[ZERO]], %[[ENABLE]])
// STOP: %[[SECOND_STATUS:.*]] = arith.constant 3350532 : i32
// STOP: aiex.npu.maskpoll(%[[SECOND_STATUS]], %{{.*}}, %{{.*}})
// STOP-NEXT: aiex.control_packet {address = 2224208 : ui32, data = array<i32: 1>
// STOP: %[[STALL:.*]] = arith.constant 960 : i32
// STOP-NEXT: aiex.npu.maskpoll(%{{.*}}, %{{.*}}, %[[STALL]])

module {
  aie.device(npu2_1col) @overlay {
    %low = aie.tile(0, 2)
    %high = aie.tile(0, 3)
    aie.switchbox(%low) {
      aie.connect<South : 0, TileControl : 0>
    }
    aie.switchbox(%high) {
      aie.connect<South : 0, TileControl : 0>
    }
  }
  aie.device(npu2_1col) @cfg {
    %low = aie.tile(0, 2)
    %high = aie.tile(0, 3)
    %lock = aie.lock(%low, 5) {init = 0 : i32}
    aie.core(%low) {
      aie.end
    }
    aie.core(%high) {
      %one = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %one)
      aie.end
    }
  } {overlay = @overlay}
  aie.device(npu2_1col) @main {
    aie.runtime_sequence @sequence() {
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      aiex.npu.load_pdi {device_ref = @cfg, expand_mode = 2 : i32}
      aiex.npu.sync(%zero, %zero, %zero, %zero, %one, %one) : i32, i32, i32, i32, i32, i32
    }
  }
}
