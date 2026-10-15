-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Net

namespace IronVerify.Designs.WholeArrayC1

open IronVerify

-- Wide designs nest their runtime sequence deeper than Lean's default limit.
set_option maxRecDepth 20000

def params : List String := ["M", "K", "N"]

def net : Net where
  fifos := [
    { name := "A_L3L2_0", objSize := 16384, consSize := 16384, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 0), none)] },
    { name := "B_L3L2_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := .shim, cons := [((.link 1), none)] },
    { name := "C_L2L3_0", objSize := 8192, consSize := 8192, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 2), cons := [(.shim, none)] },
    { name := "A_L2L1_0", objSize := 4096, consSize := 4096, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (256 : Int)), ((16 : Int), (4 : Int)), ((4 : Int), (64 : Int)), ((4 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 0), cons := [((.worker 0), none)] },
    { name := "B_L2L1_0", objSize := 2048, consSize := 2048, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (128 : Int)), ((4 : Int), (8 : Int)), ((4 : Int), (32 : Int)), ((8 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 1), cons := [((.worker 0), none), ((.worker 1), none), ((.worker 2), none), ((.worker 3), none)] },
    { name := "C_L1L2_0_0", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 0), cons := [((.link 2), none)] },
    { name := "A_L2L1_1", objSize := 4096, consSize := 4096, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (256 : Int)), ((16 : Int), (4 : Int)), ((4 : Int), (64 : Int)), ((4 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 0), cons := [((.worker 1), none)] },
    { name := "C_L1L2_0_1", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 1), cons := [((.link 2), none)] },
    { name := "A_L2L1_2", objSize := 4096, consSize := 4096, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (256 : Int)), ((16 : Int), (4 : Int)), ((4 : Int), (64 : Int)), ((4 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 0), cons := [((.worker 2), none)] },
    { name := "C_L1L2_0_2", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 2), cons := [((.link 2), none)] },
    { name := "A_L2L1_3", objSize := 4096, consSize := 4096, toStream := (some { offset := (0 : Int), dims := [((16 : Int), (256 : Int)), ((16 : Int), (4 : Int)), ((4 : Int), (64 : Int)), ((4 : Int), (1 : Int))] }), repeatCount := 1, prod := (.link 0), cons := [((.worker 3), none)] },
    { name := "C_L1L2_0_3", objSize := 2048, consSize := 2048, toStream := none, repeatCount := 1, prod := (.worker 3), cons := [((.link 2), none)] }]
  links := [
    { srcs := ["A_L3L2_0"], dsts := ["A_L2L1_0", "A_L2L1_1", "A_L2L1_2", "A_L2L1_3"], srcOffsets := [], dstOffsets := [0, 4096, 8192, 12288] },
    { srcs := ["B_L3L2_0"], dsts := ["B_L2L1_0"], srcOffsets := [], dstOffsets := [0] },
    { srcs := ["C_L1L2_0_0", "C_L1L2_0_1", "C_L1L2_0_2", "C_L1L2_0_3"], dsts := ["C_L2L3_0"], srcOffsets := [0, 2048, 4096, 6144], dstOffsets := [] }]

def seq (M : Int) (K : Int) (N : Int) : Seq :=
  (.seq (.check (decide (M > (0 : Int))))
  (.seq (.check (decide (K > (0 : Int))))
  (.seq (.check (decide (N > (0 : Int))))
  (.seq (.check (decide ((Int.tmod M (256 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.tmod K (64 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.tmod N (32 : Int)) = (0 : Int))))
  (.seq (.check (decide (M ≥ (1 : Int))))
  (.seq (.check (decide (K ≥ (1 : Int))))
  (.seq (.check (decide ((M * K) > (0 : Int))))
  (.seq (.check (decide (M ≥ (1 : Int))))
  (.seq (.check (decide (K ≥ (1 : Int))))
  (.seq (.check (decide (((1 : Int) * K) ≥ (0 : Int))))
  (.seq (.check (decide ((Int.tmod M (256 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.tmod K (64 : Int)) = (0 : Int))))
  (.seq (.check (decide ((Int.fdiv (Int.fdiv N (32 : Int)) (1 : Int)) ≥ (1 : Int))))
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
    (.seq (.xfer "C_L2L3_0" 2 { offset := (((0 : Int) + ((0 : Int) * (((1 : Int) * N) * (256 : Int)))) + (0 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (1 : Int)) + (1 : Int)), (32 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.seq (.xfer "A_L3L2_0" 0 { offset := ((0 : Int) + ((0 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (1 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((256 : Int), ((1 : Int) * K)), ((64 : Int), (1 : Int))] })
    (.seq (.xfer "B_L3L2_0" 1 { offset := (0 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (1 : Int)) + (1 : Int)), (32 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
    (.ite (decide ((Int.fdiv (Int.fdiv M (64 : Int)) (4 : Int)) > (1 : Int)))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
      (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (1 : Int))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
      (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
      (.seq (.xfer "C_L2L3_0" 2 { offset := (((0 : Int) + ((1 : Int) * (((1 : Int) * N) * (256 : Int)))) + (0 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (1 : Int)) + (1 : Int)), (32 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.seq (.xfer "A_L3L2_0" 0 { offset := ((0 : Int) + ((1 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (1 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((256 : Int), ((1 : Int) * K)), ((64 : Int), (1 : Int))] })
      (.seq (.xfer "B_L3L2_0" 1 { offset := (0 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (1 : Int)) + (1 : Int)), (32 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
      (.ite (decide ((Int.fdiv (Int.fdiv M (64 : Int)) (4 : Int)) > (2 : Int)))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
        (.seq (.check (decide ((Int.fdiv M (256 : Int)) > (2 : Int))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
        (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
        (.seq (.xfer "C_L2L3_0" 2 { offset := (((0 : Int) + ((2 : Int) * (((1 : Int) * N) * (256 : Int)))) + (0 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (1 : Int)) + (1 : Int)), (32 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.seq (.xfer "A_L3L2_0" 0 { offset := ((0 : Int) + ((2 : Int) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (1 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((256 : Int), ((1 : Int) * K)), ((64 : Int), (1 : Int))] })
        (.seq (.xfer "B_L3L2_0" 1 { offset := (0 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (1 : Int)) + (1 : Int)), (32 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
        (.loop (3 : Int) (Int.fdiv (Int.fdiv M (64 : Int)) (4 : Int)) fun i0 =>
          (.seq (.check (decide (i0 ≥ (0 : Int))))
          (.seq (.check (decide (i0 < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
          (.seq (.check (decide (((i0 * (1 : Int)) + (0 : Int)) ≥ (0 : Int))))
          (.seq (.check (decide (((i0 * (1 : Int)) + (0 : Int)) < (Int.fdiv M (256 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) ≤ (Int.fdiv N (32 : Int)))))
          (.seq (.check (decide ((Int.fdiv N (32 : Int)) > (0 : Int))))
          (.seq (.xfer "C_L2L3_0" 2 { offset := (((0 : Int) + (i0 * (((1 : Int) * N) * (256 : Int)))) + (0 : Int)), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (1 : Int)) + (1 : Int)), (32 : Int)), ((256 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })
          (.seq (.xfer "A_L3L2_0" 0 { offset := ((0 : Int) + (((i0 * (1 : Int)) + (0 : Int)) * (((1 : Int) * K) * (256 : Int)))), dims := [((Int.fdiv (Int.fdiv N (32 : Int)) (1 : Int)), (0 : Int)), ((Int.fdiv K (64 : Int)), (64 : Int)), ((256 : Int), ((1 : Int) * K)), ((64 : Int), (1 : Int))] })
          (.xfer "B_L3L2_0" 1 { offset := (0 : Int), dims := [(((Int.fdiv (((Int.fdiv N (32 : Int)) - (0 : Int)) - (1 : Int)) (1 : Int)) + (1 : Int)), (32 : Int)), ((Int.fdiv K (64 : Int)), (((1 : Int) * N) * (64 : Int))), ((64 : Int), ((1 : Int) * N)), ((32 : Int), (1 : Int))] })))))))))))))))))))))
        .skip))))))))))
      .skip))))))))))
    .skip))))))))))))))))))))))))))))))))

def design : Design (Int × Int × Int) where
  net := net
  seq := fun (M, K, N) => seq M K N

def seqOfList : List Int → Option Seq
  | [M, K, N] => some (seq M K N)
  | _ => none

end IronVerify.Designs.WholeArrayC1
