//===- page_split_diagnostics.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Diagnostics when auto-splitting an oversized page.
//  - splitting an implicit page (formed from contiguous top-level jobs)
//    into sequential pages emits a remark;
//  - if no split keeps a local_barrier group intact, splitting is refused.
//    Leaving the page whole is the safer of two bad options -- severing a
//    barrier hangs the firmware -- but that only holds while the page still
//    fits, so it is an error once the page passes the 8192-byte limit. Fires
//    for implicit and explicit pages alike; only the remedy differs.
//  - Same 8192-byte rule when the page offers no split point at all, which
//    happens when its bulk sits in a single one-op job.
//
// Note the two thresholds: 8000 triggers a split attempt, 8192 is the hardware
// page limit and the only thing worth erroring about. A page between them trips
// the trigger yet loads fine, so failing to split it is not a diagnostic.

// RUN: aie-opt -cert-legalize-pages --split-input-file --verify-diagnostics %s

// Remark: an oversized implicit page with no blocking barrier splits into two
// sequential pages -> remark on the implicit page.
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<1300xi32> = dense<0>
  memref.global "private" constant @d2 : memref<1300xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 1300, false
  }
  aiex.cert.uc_dma_chain @c2 {
    aiex.cert.uc_dma_bd @d2, 0, 1300, false
  }
  // expected-remark@+1 {{auto-split of implicit cert.page serializes cooperatively-scheduled jobs}}
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@c1)
    aiex.cert.uc_dma_write_des_sync(@c2)
  }
}

// -----

// No remark: the same oversized job, but tagged {cert.configure}. That page is
// synthesized by the compiler from the @configure runtime sequence, so the
// remark's advice -- add an explicit cert.page boundary -- is not something the
// user can act on, and config transactions are routinely large enough to split.
// formImplicitPagesInBlock therefore isolates it WITHOUT the cert.implicit tag.
// --verify-diagnostics fails on an unexpected remark, so this case passing is
// the assertion. It still splits; it just splits quietly.
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<1300xi32> = dense<0>
  memref.global "private" constant @d2 : memref<1300xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 1300, false
  }
  aiex.cert.uc_dma_chain @c2 {
    aiex.cert.uc_dma_bd @d2, 0, 1300, false
  }
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@c1)
    aiex.cert.uc_dma_write_des_sync(@c2)
  } {cert.configure}
}

// -----

// Error: an oversized implicit page whose only local_barrier group spans the
// whole page cannot be split without separating the participants -> error.
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<1300xi32> = dense<0>
  memref.global "private" constant @d2 : memref<1300xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 1300, false
  }
  aiex.cert.uc_dma_chain @c2 {
    aiex.cert.uc_dma_bd @d2, 0, 1300, false
  }
  // expected-error@+1 {{over the 8192-byte microcontroller page limit, and cannot be split without separating a local_barrier participant set across pages; insert an explicit cert.page boundary}}
  aiex.cert.job(1) {
    aiex.cert.local_barrier(0, 2)
    aiex.cert.uc_dma_write_des_sync(@c1)
    aiex.cert.uc_dma_write_des_sync(@c2)
    aiex.cert.local_barrier(0, 2)
  }
}

// -----

// Error: same situation, but the oversized page is an EXPLICIT user-authored
// cert.page. There is still no legal lowering, so it must not pass silently --
// the remedy is to shrink the page or regroup the participants, since the user
// already chose this boundary. Compare page_split_localbar.mlir, where an
// explicit page of the same shape DOES have a legal cut and splits cleanly.
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<1300xi32> = dense<0>
  memref.global "private" constant @d2 : memref<1300xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 1300, false
  }
  aiex.cert.uc_dma_chain @c2 {
    aiex.cert.uc_dma_bd @d2, 0, 1300, false
  }
  // expected-error@+1 {{over the 8192-byte microcontroller page limit, and cannot be split without separating a local_barrier participant set across pages; reduce the page's contents or regroup its local_barrier participants}}
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.local_barrier(0, 2)
      aiex.cert.uc_dma_write_des_sync(@c1)
      aiex.cert.uc_dma_write_des_sync(@c2)
      aiex.cert.local_barrier(0, 2)
    }
  }
}

// -----

// Error: an oversized page whose bulk is a single one-op job. Every cut has to
// leave at least one op on each side and this page holds exactly one op, so it
// offers no split point at all -- no barrier is involved. Over the 8192-byte
// limit, so it is an error rather than a silent oversized page.
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<2500xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 2500, false
  }
  // expected-error@+1 {{offers no split point (a split must leave at least one op on each page); break the oversized job into smaller jobs}}
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@c1)
  }
}

// -----

// Remark, not an error: two one-op jobs, each individually fine and jointly over
// the limit, grouped into one implicit page. The only cut available is the
// boundary between them, which is a real split -- so this takes the ordinary
// implicit-page serialization remark and not the "offers no split point" error.
// See page_split_job_boundary.mlir for the explicit-page form of this shape.
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<1200xi32> = dense<0>
  memref.global "private" constant @d2 : memref<1200xi32> = dense<1>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 1200, false
  }
  aiex.cert.uc_dma_chain @c2 {
    aiex.cert.uc_dma_bd @d2, 0, 1200, false
  }
  // expected-remark@+1 {{auto-split of implicit cert.page serializes cooperatively-scheduled jobs}}
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@c1)
  }
  aiex.cert.job(2) {
    aiex.cert.uc_dma_write_des_sync(@c2)
  }
}

// -----

// The 8192 boundary, both sides. These two cases were validated end to end
// against the real toolchain -- aie-translate --aie-cert-to-asm piped into
// `aiebu-asm` -- and the assembler's own accounting agrees with
// estimateCost byte for byte: it accepts 2032 and rejects 2034 with
// "text and data section size 8200 > pagesize(8192)". Both text and chain data
// count toward the page, which is why data_cost is charged to the page budget.
//
// Under the limit: estimate is exactly 8192 (32 page + 8 START_JOB + 4
// write_des_sync + 4 END_JOB + 16 BD + 2032 x 4 data). It trips the 8000 split
// trigger and has no split point, but it fits, so it must pass silently. This
// is the case that keeps the two thresholds from collapsing into one.
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<2032xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 2032, false
  }
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@c1)
  }
}

// -----

// Two elements over: estimate 8200, past the limit, no split point -> error.
aie.device(npu2) {
  memref.global "private" constant @d1 : memref<2034xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @d1, 0, 2034, false
  }
  // expected-error@+1 {{cert.page is an estimated 8200 bytes, over the 8192-byte microcontroller page limit, and offers no split point}}
  aiex.cert.job(1) {
    aiex.cert.uc_dma_write_des_sync(@c1)
  }
}
