//===- isolate_full_page_order.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// load_pdi and preempt each need a page to themselves, but pages execute in
// order, so pulling one out must not move it past the jobs that surround it.
// The isolation rewrite used to anchor the pages it creates on the *parent
// page* rather than on the job being isolated, which silently reordered the
// program whenever that job was not the page's only job. Nothing downstream
// checks op order, so these cases used to compile clean and run wrong.
//
// Every write32 address below is unique so the ordered CHECKs cannot be
// satisfied by a match in a later input chunk.

// RUN: aie-opt -cert-legalize-pages --split-input-file %s | FileCheck %s

// A single-op load_pdi job that is NOT the last job on its page: the isolated
// page used to land after the whole parent page, so the write32 ran before the
// reconfiguration it was supposed to follow.
// CHECK: aiex.cert.load_pdi(1, @cfg)
// CHECK: aiex.cert.write32(4096, 1)
aie.device(npu2) {
  aiex.cert.section @cfg { }
  aiex.cert.page {
    aiex.cert.job(1) { aiex.cert.load_pdi(1, @cfg) }
    aiex.cert.job(2) { aiex.cert.write32(4096, 1) }
  }
}

// -----

// Same shape with preempt.
// CHECK: aiex.cert.preempt(1, @save, @restore)
// CHECK: aiex.cert.write32(4104, 2)
aie.device(npu2) {
  aiex.cert.section @save { }
  aiex.cert.section @restore { }
  aiex.cert.page {
    aiex.cert.job(1) { aiex.cert.preempt(1, @save, @restore) }
    aiex.cert.job(2) { aiex.cert.write32(4104, 2) }
  }
}

// -----

// Mixed job (ops before and after the load_pdi) preceded by an unrelated job:
// the "before ops" page used to be inserted ahead of the whole parent page,
// hoisting write32(4116) over write32(4112).
// CHECK: aiex.cert.write32(4112, 3)
// CHECK: aiex.cert.write32(4116, 4)
// CHECK: aiex.cert.load_pdi(1, @cfg)
// CHECK: aiex.cert.write32(4120, 5)
aie.device(npu2) {
  aiex.cert.section @cfg { }
  aiex.cert.page {
    aiex.cert.job(1) { aiex.cert.write32(4112, 3) }
    aiex.cert.job(2) {
      aiex.cert.write32(4116, 4)
      aiex.cert.load_pdi(1, @cfg)
      aiex.cert.write32(4120, 5)
    }
  }
}

// -----

// Leading and trailing siblings around a mixed job, plus a second full-page op
// later on the same page: every fragment keeps its source position.
// CHECK: aiex.cert.write32(4128, 6)
// CHECK: aiex.cert.write32(4132, 7)
// CHECK: aiex.cert.load_pdi(1, @cfg)
// CHECK: aiex.cert.write32(4136, 8)
// CHECK: aiex.cert.write32(4140, 9)
// CHECK: aiex.cert.load_pdi(2, @cfg)
// CHECK: aiex.cert.write32(4144, 10)
aie.device(npu2) {
  aiex.cert.section @cfg { }
  aiex.cert.page {
    aiex.cert.job(1) { aiex.cert.write32(4128, 6) }
    aiex.cert.job(2) {
      aiex.cert.write32(4132, 7)
      aiex.cert.load_pdi(1, @cfg)
      aiex.cert.write32(4136, 8)
    }
    aiex.cert.job(3) { aiex.cert.write32(4140, 9) }
    aiex.cert.job(4) { aiex.cert.load_pdi(2, @cfg) }
    aiex.cert.job(5) { aiex.cert.write32(4144, 10) }
  }
}
