-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Net

namespace IronVerify.Designs.WholeArrayC7

open IronVerify

-- Wide designs nest their runtime sequence deeper than Lean's default limit.
set_option maxRecDepth 20000

def params : List String := ["M", "K", "N"]

def net : Net where
  fifos := [
    { name := "A_L3L2_0", objSize := 4096, consSize := 4096, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 0), none)] },
    { name := "A_L3L2_1", objSize := 4096, consSize := 4096, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 1), none)] },
    { name := "A_L3L2_2", objSize := 4096, consSize := 4096, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 2), none)] },
    { name := "A_L3L2_3", objSize := 4096, consSize := 4096, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 3), none)] },
    { name := "B_L3L2_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 4), none)] },
    { name := "B_L3L2_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 5), none)] },
    { name := "B_L3L2_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 6), none)] },
    { name := "B_L3L2_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 7), none)] },
    { name := "B_L3L2_4", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 8), none)] },
    { name := "B_L3L2_5", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 9), none)] },
    { name := "B_L3L2_6", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 10), none)] },
    { name := "C_L2L3_0", objSize := 8192, consSize := 8192, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 11), cons := [(.shim, none)] },
    { name := "C_L2L3_1", objSize := 8192, consSize := 8192, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 12), cons := [(.shim, none)] },
    { name := "C_L2L3_2", objSize := 8192, consSize := 8192, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 13), cons := [(.shim, none)] },
    { name := "C_L2L3_3", objSize := 8192, consSize := 8192, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 14), cons := [(.shim, none)] },
    { name := "C_L2L3_4", objSize := 8192, consSize := 8192, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 15), cons := [(.shim, none)] },
    { name := "C_L2L3_5", objSize := 8192, consSize := 8192, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 16), cons := [(.shim, none)] },
    { name := "C_L2L3_6", objSize := 8192, consSize := 8192, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 17), cons := [(.shim, none)] },
    { name := "A_L2L1_0", objSize := 4096, consSize := 4096, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (256 : Int)), ((16 : Int), (4 : Int)), ((4 : Int), (64 : Int)), ((4 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 0), cons := [((.worker 0), none), ((.worker 1), none), ((.worker 2), none), ((.worker 3), none), ((.worker 4), none), ((.worker 5), none), ((.worker 6), none)] },
    { name := "B_L2L1_0", objSize := 2048, consSize := 2048, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 4), cons := [((.worker 0), none), ((.worker 7), none), ((.worker 14), none), ((.worker 21), none)] },
    { name := "C_L1L2_0_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 0), cons := [((.link 11), none)] },
    { name := "B_L2L1_1", objSize := 2048, consSize := 2048, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 5), cons := [((.worker 1), none), ((.worker 8), none), ((.worker 15), none), ((.worker 22), none)] },
    { name := "C_L1L2_1_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 1), cons := [((.link 12), none)] },
    { name := "B_L2L1_2", objSize := 2048, consSize := 2048, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 6), cons := [((.worker 2), none), ((.worker 9), none), ((.worker 16), none), ((.worker 23), none)] },
    { name := "C_L1L2_2_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 2), cons := [((.link 13), none)] },
    { name := "B_L2L1_3", objSize := 2048, consSize := 2048, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 7), cons := [((.worker 3), none), ((.worker 10), none), ((.worker 17), none), ((.worker 24), none)] },
    { name := "C_L1L2_3_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 3), cons := [((.link 14), none)] },
    { name := "B_L2L1_4", objSize := 2048, consSize := 2048, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 8), cons := [((.worker 4), none), ((.worker 11), none), ((.worker 18), none), ((.worker 25), none)] },
    { name := "C_L1L2_4_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 4), cons := [((.link 15), none)] },
    { name := "B_L2L1_5", objSize := 2048, consSize := 2048, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 9), cons := [((.worker 5), none), ((.worker 12), none), ((.worker 19), none), ((.worker 26), none)] },
    { name := "C_L1L2_5_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 5), cons := [((.link 16), none)] },
    { name := "B_L2L1_6", objSize := 2048, consSize := 2048, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 10), cons := [((.worker 6), none), ((.worker 13), none), ((.worker 20), none), ((.worker 27), none)] },
    { name := "C_L1L2_6_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 6), cons := [((.link 17), none)] },
    { name := "A_L2L1_1", objSize := 4096, consSize := 4096, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (256 : Int)), ((16 : Int), (4 : Int)), ((4 : Int), (64 : Int)), ((4 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 1), cons := [((.worker 7), none), ((.worker 8), none), ((.worker 9), none), ((.worker 10), none), ((.worker 11), none), ((.worker 12), none), ((.worker 13), none)] },
    { name := "C_L1L2_0_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 7), cons := [((.link 11), none)] },
    { name := "C_L1L2_1_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 8), cons := [((.link 12), none)] },
    { name := "C_L1L2_2_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 9), cons := [((.link 13), none)] },
    { name := "C_L1L2_3_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 10), cons := [((.link 14), none)] },
    { name := "C_L1L2_4_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 11), cons := [((.link 15), none)] },
    { name := "C_L1L2_5_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 12), cons := [((.link 16), none)] },
    { name := "C_L1L2_6_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 13), cons := [((.link 17), none)] },
    { name := "A_L2L1_2", objSize := 4096, consSize := 4096, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (256 : Int)), ((16 : Int), (4 : Int)), ((4 : Int), (64 : Int)), ((4 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 2), cons := [((.worker 14), none), ((.worker 15), none), ((.worker 16), none), ((.worker 17), none), ((.worker 18), none), ((.worker 19), none), ((.worker 20), none)] },
    { name := "C_L1L2_0_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 14), cons := [((.link 11), none)] },
    { name := "C_L1L2_1_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 15), cons := [((.link 12), none)] },
    { name := "C_L1L2_2_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 16), cons := [((.link 13), none)] },
    { name := "C_L1L2_3_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 17), cons := [((.link 14), none)] },
    { name := "C_L1L2_4_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 18), cons := [((.link 15), none)] },
    { name := "C_L1L2_5_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 19), cons := [((.link 16), none)] },
    { name := "C_L1L2_6_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 20), cons := [((.link 17), none)] },
    { name := "A_L2L1_3", objSize := 4096, consSize := 4096, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (256 : Int)), ((16 : Int), (4 : Int)), ((4 : Int), (64 : Int)), ((4 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 3), cons := [((.worker 21), none), ((.worker 22), none), ((.worker 23), none), ((.worker 24), none), ((.worker 25), none), ((.worker 26), none), ((.worker 27), none)] },
    { name := "C_L1L2_0_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 21), cons := [((.link 11), none)] },
    { name := "C_L1L2_1_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 22), cons := [((.link 12), none)] },
    { name := "C_L1L2_2_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 23), cons := [((.link 13), none)] },
    { name := "C_L1L2_3_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 24), cons := [((.link 14), none)] },
    { name := "C_L1L2_4_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 25), cons := [((.link 15), none)] },
    { name := "C_L1L2_5_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 26), cons := [((.link 16), none)] },
    { name := "C_L1L2_6_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 27), cons := [((.link 17), none)] }]
  links := [
    { srcs := ["A_L3L2_0"], dsts := ["A_L2L1_0"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["A_L3L2_1"], dsts := ["A_L2L1_1"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["A_L3L2_2"], dsts := ["A_L2L1_2"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["A_L3L2_3"], dsts := ["A_L2L1_3"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["B_L3L2_0"], dsts := ["B_L2L1_0"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["B_L3L2_1"], dsts := ["B_L2L1_1"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["B_L3L2_2"], dsts := ["B_L2L1_2"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["B_L3L2_3"], dsts := ["B_L2L1_3"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["B_L3L2_4"], dsts := ["B_L2L1_4"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["B_L3L2_5"], dsts := ["B_L2L1_5"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["B_L3L2_6"], dsts := ["B_L2L1_6"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["C_L1L2_0_0", "C_L1L2_0_1", "C_L1L2_0_2", "C_L1L2_0_3"], dsts := ["C_L2L3_0"], srcOffsets := [0, 2048, 4096, 6144], dstOffsets := [] },
    { srcs := ["C_L1L2_1_0", "C_L1L2_1_1", "C_L1L2_1_2", "C_L1L2_1_3"], dsts := ["C_L2L3_1"], srcOffsets := [0, 2048, 4096, 6144], dstOffsets := [] },
    { srcs := ["C_L1L2_2_0", "C_L1L2_2_1", "C_L1L2_2_2", "C_L1L2_2_3"], dsts := ["C_L2L3_2"], srcOffsets := [0, 2048, 4096, 6144], dstOffsets := [] },
    { srcs := ["C_L1L2_3_0", "C_L1L2_3_1", "C_L1L2_3_2", "C_L1L2_3_3"], dsts := ["C_L2L3_3"], srcOffsets := [0, 2048, 4096, 6144], dstOffsets := [] },
    { srcs := ["C_L1L2_4_0", "C_L1L2_4_1", "C_L1L2_4_2", "C_L1L2_4_3"], dsts := ["C_L2L3_4"], srcOffsets := [0, 2048, 4096, 6144], dstOffsets := [] },
    { srcs := ["C_L1L2_5_0", "C_L1L2_5_1", "C_L1L2_5_2", "C_L1L2_5_3"], dsts := ["C_L2L3_5"], srcOffsets := [0, 2048, 4096, 6144], dstOffsets := [] },
    { srcs := ["C_L1L2_6_0", "C_L1L2_6_1", "C_L1L2_6_2", "C_L1L2_6_3"], dsts := ["C_L2L3_6"], srcOffsets := [0, 2048, 4096, 6144], dstOffsets := [] }]

def seq (M : Int) (K : Int) (N : Int) : Seq :=
  (.seq (.check (decide (M > (0 : Int))))
  (.seq (.check (decide (K > (0 : Int))))
  (.seq (.check (decide (N > (0 : Int))))
  (.seq (.check (decide ((Int.tmod M (256 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.tmod K (64 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.tmod N (224 : Int)) = (0 : Int))))
  (.seq (.check (decide (M ≥ (1 : Int))))
  (.seq (.check (decide (K ≥ (1 : Int))))
  (.seq (.check (decide ((M * K) > (0 : Int))))
  (.seq (.check (decide (M ≥ (1 : Int))))
  (.seq (.check (decide (K ≥ (1 : Int))))
  (.seq (.check (decide (((1 : Int) * K) ≥ (0 : Int))))
  (.seq (.check (decide ((Int.tmod M (256 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.tmod K (64 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)) ≥ (1 : Int))))
  (.seq (.check (decide ((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)) ≥ (1 : Int))))
  (.seq (.check (decide ((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)) ≥ (1 : Int))))
  (.seq (.check (decide ((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)) ≥ (1 : Int))))
  (.seq (.check (decide (K ≥ (1 : Int))))
  (.seq (.check (decide (N ≥ (1 : Int))))
  (.seq (.check (decide ((K * N) > (0 : Int))))
  (.seq (.check (decide (K ≥ (1 : Int))))
  (.seq (.check (decide (N ≥ (1 : Int))))
  (.seq (.check (decide (((1 : Int) * N) ≥ (0 : Int))))
  (.seq (.check (decide ((Int.tmod K (64 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.tmod N (32 : Int)) = (0 : Int))))
  (.seq (.check (decide (M ≥ (1 : Int))))
  (.seq (.check (decide (N ≥ (1 : Int))))
  (.seq (.check (decide ((M * N) > (0 : Int))))
  (.seq (.check (decide (M ≥ (1 : Int))))
  (.seq (.check (decide (N ≥ (1 : Int))))
  (.seq (.check (decide (((1 : Int) * N) ≥ (0 : Int))))
  (.seq (.check (decide ((Int.tmod M (256 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.tmod N (32 : Int)) = (0 : Int))))
  (.ite (decide ((Int.fdiv (Int.fdiv M (64 : Int)) (4 : Int)) > (0 : Int)))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (1 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (1 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (2 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (2 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (3 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (3 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (4 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (4 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (5 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (5 : Int))))
    (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (0 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (6 : Int))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
    (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (6 : Int))))
    (.seq (.xfer "C_L2L3_0" 2 { offset := (((0 : Int) + ((0 : Int) * (((1 : Int) * N) * (256 : Int)))) + (0 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "A_L3L2_0" 0 { offset := (((0 : Int) + ((0 : Int) * ((1 : Int) * K))) + ((0 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
    (.seq (.xfer "B_L3L2_0" 1 { offset := (0 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "C_L2L3_1" 2 { offset := (((0 : Int) + ((0 : Int) * (((1 : Int) * N) * (256 : Int)))) + (32 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (1 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "A_L3L2_1" 0 { offset := (((0 : Int) + ((64 : Int) * ((1 : Int) * K))) + ((0 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
    (.seq (.xfer "B_L3L2_1" 1 { offset := (32 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (1 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "C_L2L3_2" 2 { offset := (((0 : Int) + ((0 : Int) * (((1 : Int) * N) * (256 : Int)))) + (64 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (2 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "A_L3L2_2" 0 { offset := (((0 : Int) + ((128 : Int) * ((1 : Int) * K))) + ((0 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
    (.seq (.xfer "B_L3L2_2" 1 { offset := (64 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (2 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "C_L2L3_3" 2 { offset := (((0 : Int) + ((0 : Int) * (((1 : Int) * N) * (256 : Int)))) + (96 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (3 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "A_L3L2_3" 0 { offset := (((0 : Int) + ((192 : Int) * ((1 : Int) * K))) + ((0 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
    (.seq (.xfer "B_L3L2_3" 1 { offset := (96 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (3 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "C_L2L3_4" 2 { offset := (((0 : Int) + ((0 : Int) * (((1 : Int) * N) * (256 : Int)))) + (128 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (4 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "B_L3L2_4" 1 { offset := (128 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (4 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "C_L2L3_5" 2 { offset := (((0 : Int) + ((0 : Int) * (((1 : Int) * N) * (256 : Int)))) + (160 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (5 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "B_L3L2_5" 1 { offset := (160 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (5 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "C_L2L3_6" 2 { offset := (((0 : Int) + ((0 : Int) * (((1 : Int) * N) * (256 : Int)))) + (192 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (6 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "B_L3L2_6" 1 { offset := (192 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (6 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.ite (decide ((Int.fdiv (Int.fdiv M (64 : Int)) (4 : Int)) > (1 : Int)))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (2 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (2 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (3 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (3 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (4 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (4 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (5 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (5 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (6 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (6 : Int))))
      (.seq (.xfer "C_L2L3_0" 2 { offset := (((0 : Int) + ((1 : Int) * (((1 : Int) * N) * (256 : Int)))) + (0 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "A_L3L2_0" 0 { offset := (((0 : Int) + ((0 : Int) * ((1 : Int) * K))) + ((1 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
      (.seq (.xfer "B_L3L2_0" 1 { offset := (0 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "C_L2L3_1" 2 { offset := (((0 : Int) + ((1 : Int) * (((1 : Int) * N) * (256 : Int)))) + (32 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (1 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "A_L3L2_1" 0 { offset := (((0 : Int) + ((64 : Int) * ((1 : Int) * K))) + ((1 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
      (.seq (.xfer "B_L3L2_1" 1 { offset := (32 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (1 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "C_L2L3_2" 2 { offset := (((0 : Int) + ((1 : Int) * (((1 : Int) * N) * (256 : Int)))) + (64 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (2 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "A_L3L2_2" 0 { offset := (((0 : Int) + ((128 : Int) * ((1 : Int) * K))) + ((1 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
      (.seq (.xfer "B_L3L2_2" 1 { offset := (64 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (2 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "C_L2L3_3" 2 { offset := (((0 : Int) + ((1 : Int) * (((1 : Int) * N) * (256 : Int)))) + (96 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (3 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "A_L3L2_3" 0 { offset := (((0 : Int) + ((192 : Int) * ((1 : Int) * K))) + ((1 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
      (.seq (.xfer "B_L3L2_3" 1 { offset := (96 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (3 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "C_L2L3_4" 2 { offset := (((0 : Int) + ((1 : Int) * (((1 : Int) * N) * (256 : Int)))) + (128 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (4 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "B_L3L2_4" 1 { offset := (128 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (4 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "C_L2L3_5" 2 { offset := (((0 : Int) + ((1 : Int) * (((1 : Int) * N) * (256 : Int)))) + (160 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (5 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "B_L3L2_5" 1 { offset := (160 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (5 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "C_L2L3_6" 2 { offset := (((0 : Int) + ((1 : Int) * (((1 : Int) * N) * (256 : Int)))) + (192 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (6 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "B_L3L2_6" 1 { offset := (192 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (6 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.ite (decide ((Int.fdiv (Int.fdiv M (64 : Int)) (4 : Int)) > (2 : Int)))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (1 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (1 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (3 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (3 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (4 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (4 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (5 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (5 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (6 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (6 : Int))))
        (.seq (.xfer "C_L2L3_0" 2 { offset := (((0 : Int) + ((2 : Int) * (((1 : Int) * N) * (256 : Int)))) + (0 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "A_L3L2_0" 0 { offset := (((0 : Int) + ((0 : Int) * ((1 : Int) * K))) + ((2 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
        (.seq (.xfer "B_L3L2_0" 1 { offset := (0 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "C_L2L3_1" 2 { offset := (((0 : Int) + ((2 : Int) * (((1 : Int) * N) * (256 : Int)))) + (32 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (1 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "A_L3L2_1" 0 { offset := (((0 : Int) + ((64 : Int) * ((1 : Int) * K))) + ((2 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
        (.seq (.xfer "B_L3L2_1" 1 { offset := (32 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (1 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "C_L2L3_2" 2 { offset := (((0 : Int) + ((2 : Int) * (((1 : Int) * N) * (256 : Int)))) + (64 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (2 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "A_L3L2_2" 0 { offset := (((0 : Int) + ((128 : Int) * ((1 : Int) * K))) + ((2 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
        (.seq (.xfer "B_L3L2_2" 1 { offset := (64 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (2 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "C_L2L3_3" 2 { offset := (((0 : Int) + ((2 : Int) * (((1 : Int) * N) * (256 : Int)))) + (96 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (3 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "A_L3L2_3" 0 { offset := (((0 : Int) + ((192 : Int) * ((1 : Int) * K))) + ((2 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
        (.seq (.xfer "B_L3L2_3" 1 { offset := (96 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (3 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "C_L2L3_4" 2 { offset := (((0 : Int) + ((2 : Int) * (((1 : Int) * N) * (256 : Int)))) + (128 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (4 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "B_L3L2_4" 1 { offset := (128 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (4 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "C_L2L3_5" 2 { offset := (((0 : Int) + ((2 : Int) * (((1 : Int) * N) * (256 : Int)))) + (160 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (5 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "B_L3L2_5" 1 { offset := (160 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (5 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "C_L2L3_6" 2 { offset := (((0 : Int) + ((2 : Int) * (((1 : Int) * N) * (256 : Int)))) + (192 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (6 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "B_L3L2_6" 1 { offset := (192 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (6 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.loop (3 : Int) (Int.fdiv (Int.fdiv M (64 : Int)) (4 : Int)) fun i0 =>
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (1 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (1 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (2 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (2 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (3 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (3 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (4 : Int))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (4 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (5 : Int))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (5 : Int))))
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (6 : Int))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (6 : Int))))
          (.seq (.xfer "C_L2L3_0" 2 { offset := (((0 : Int) + (i0 * (((1 : Int) * N) * (256 : Int)))) + (0 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "A_L3L2_0" 0 { offset := (((0 : Int) + ((0 : Int) * ((1 : Int) * K))) + (i0 * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
          (.seq (.xfer "B_L3L2_0" 1 { offset := (0 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "C_L2L3_1" 2 { offset := (((0 : Int) + (i0 * (((1 : Int) * N) * (256 : Int)))) + (32 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (1 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "A_L3L2_1" 0 { offset := (((0 : Int) + ((64 : Int) * ((1 : Int) * K))) + (i0 * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
          (.seq (.xfer "B_L3L2_1" 1 { offset := (32 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (1 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "C_L2L3_2" 2 { offset := (((0 : Int) + (i0 * (((1 : Int) * N) * (256 : Int)))) + (64 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (2 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "A_L3L2_2" 0 { offset := (((0 : Int) + ((128 : Int) * ((1 : Int) * K))) + (i0 * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
          (.seq (.xfer "B_L3L2_2" 1 { offset := (64 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (2 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "C_L2L3_3" 2 { offset := (((0 : Int) + (i0 * (((1 : Int) * N) * (256 : Int)))) + (96 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (3 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "A_L3L2_3" 0 { offset := (((0 : Int) + ((192 : Int) * ((1 : Int) * K))) + (i0 * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (7 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((64 : Int), (((1 : Int) * K) * (1 : Int))), ((64 : Int), (1 : Int))] })
          (.seq (.xfer "B_L3L2_3" 1 { offset := (96 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (3 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "C_L2L3_4" 2 { offset := (((0 : Int) + (i0 * (((1 : Int) * N) * (256 : Int)))) + (128 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (4 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "B_L3L2_4" 1 { offset := (128 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (4 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "C_L2L3_5" 2 { offset := (((0 : Int) + (i0 * (((1 : Int) * N) * (256 : Int)))) + (160 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (5 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "B_L3L2_5" 1 { offset := (160 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (5 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "C_L2L3_6" 2 { offset := (((0 : Int) + (i0 * (((1 : Int) * N) * (256 : Int)))) + (192 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (6 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.xfer "B_L3L2_6" 1 { offset := (192 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (6 : Int)) - (1 : Int)) (7 : Int)) + (1 : Int)), (224 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] }))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
        .skip))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
      .skip))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
    .skip)))))))))))))))))))))))))))))))))))

def design : Design (Int × Int × Int) where
  net := net
  seq := fun (M, K, N) => seq M K N

def seqOfList : List Int → Option Seq
  | [M, K, N] => some (seq M K N)
  | _ => none

end IronVerify.Designs.WholeArrayC7
