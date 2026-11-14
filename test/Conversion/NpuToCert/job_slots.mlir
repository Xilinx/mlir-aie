//===- job_slots.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The CERT firmware indexes job contexts by job_id % 64, so job IDs on one
// page must differ modulo 64. cert-legalize-pages keeps noncolliding IDs and
// moves colliding ones to free slots.

// RUN: aie-opt -cert-legalize-pages -aie-cert-verify --split-input-file %s | FileCheck %s

// Splitting the first page twice must not shift the IDs on the second page:
// 63 would become 65 and share slot 1 with job 1.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.page
// CHECK-NEXT: aiex.cert.job(1)
// CHECK: aiex.cert.page
// CHECK-NEXT: aiex.cert.job(2)
// CHECK: aiex.cert.page
// CHECK-NEXT: aiex.cert.job(2)
// CHECK-NEXT: aiex.cert.wait_tcts(0, 6, 1)
// CHECK: aiex.cert.page
// CHECK-NEXT: aiex.cert.job(1)
// CHECK: aiex.cert.job(63)
aie.device(xcve3858) {
  memref.global "private" constant @c1data : memref<900xi32> = dense<0>
  memref.global "private" constant @c2data : memref<1100xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @c1data, 0, 900, false
  }
  aiex.cert.uc_dma_chain @c2 {
    aiex.cert.uc_dma_bd @c2data, 0, 1100, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@c1)
      aiex.cert.uc_dma_write_des_sync(@c2)
      aiex.cert.wait_tcts(0, 6, 1)
    }
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.write32(4096, 1)
    }
    aiex.cert.job(63) {
      aiex.cert.write32(4096, 2)
    }
  }
}

// -----

// Splitting job 1 gives its second half ID 2, which it shares with the job
// already on that page. The first job in textual order keeps its ID; the later
// one moves to the lowest free slot.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.page
// CHECK-NEXT: aiex.cert.job(1)
// CHECK: aiex.cert.page
// CHECK-NEXT: aiex.cert.job(2)
// CHECK: aiex.cert.page
// CHECK-NEXT: aiex.cert.job(2)
// CHECK-NEXT: aiex.cert.wait_tcts(0, 6, 1)
// CHECK: aiex.cert.job(0)
// CHECK-NEXT: aiex.cert.write32(4096, 3)
aie.device(xcve3858) {
  memref.global "private" constant @c1data : memref<900xi32> = dense<0>
  memref.global "private" constant @c2data : memref<1100xi32> = dense<0>
  aiex.cert.uc_dma_chain @c1 {
    aiex.cert.uc_dma_bd @c1data, 0, 900, false
  }
  aiex.cert.uc_dma_chain @c2 {
    aiex.cert.uc_dma_bd @c2data, 0, 1100, false
  }
  aiex.cert.page {
    aiex.cert.job(1) {
      aiex.cert.uc_dma_write_des_sync(@c1)
      aiex.cert.uc_dma_write_des_sync(@c2)
      aiex.cert.wait_tcts(0, 6, 1)
    }
    aiex.cert.job(2) {
      aiex.cert.write32(4096, 3)
    }
  }
}

// -----

// Hand-written collisions are resolved in textual order: the first job keeps
// its slot, later ones take the lowest free slots. Cooperating jobs keep their
// order.
// CHECK-LABEL: aie.device(xcve3858)
// CHECK: aiex.cert.job(1)
// CHECK-NEXT: aiex.cert.local_barrier(lb0, 2)
// CHECK: aiex.cert.job(3)
// CHECK-NEXT: aiex.cert.local_barrier(lb0, 2)
// CHECK: aiex.cert.job(0)
// CHECK: aiex.cert.job(2)
aie.device(xcve3858) {
  aiex.cert.page {
    aiex.cert.job(1) { aiex.cert.local_barrier(lb0, 2) }
    aiex.cert.job(65) { aiex.cert.local_barrier(lb0, 2) }
    aiex.cert.job(0) {}
    aiex.cert.job(2) {}
  }
}
