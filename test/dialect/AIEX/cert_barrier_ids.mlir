//===- cert_barrier_ids.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Barrier ids are named as the assembler names them: lb0..lb15 and rb0..rb6.
// Any other name, or a bare integer, does not parse.

// RUN: aie-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.local_barrier(lb0, 2)
// CHECK-NEXT: aiex.cert.local_barrier(lb15, 2)
// CHECK-NEXT: aiex.cert.remote_barrier(rb0, 1)
// CHECK-NEXT: aiex.cert.remote_barrier(rb6, 1)
aie.device(xcve3858) {
  aiex.cert.job(0) {
    aiex.cert.local_barrier(lb0, 2)
    aiex.cert.local_barrier(lb15, 2)
    aiex.cert.remote_barrier(rb0, 0x1)
    aiex.cert.remote_barrier(rb6, 0x1)
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.job(0) {
    // expected-error@+1 {{expected string or keyword containing one of the following enum values for attribute 'local_barrier_id' [lb0, lb1, lb2, lb3, lb4, lb5, lb6, lb7, lb8, lb9, lb10, lb11, lb12, lb13, lb14, lb15]}}
    aiex.cert.local_barrier(lb16, 2)
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.job(0) {
    // expected-error@+1 {{expected string or keyword containing one of the following enum values for attribute 'remote_barrier_id' [rb0, rb1, rb2, rb3, rb4, rb5, rb6]}}
    aiex.cert.remote_barrier(rb7, 0x1)
  }
}

// -----

aie.device(xcve3858) {
  aiex.cert.job(0) {
    // expected-error@+1 {{expected string or keyword containing one of the following enum values for attribute 'remote_barrier_id'}}
    aiex.cert.remote_barrier(1, 0x1)
  }
}
