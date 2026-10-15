-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Inst.Transfer

/-!
`wa_stream [facts]` proves that one shim channel of an export moves the same host elements as
`schema p`, for every shape: it unrolls the schema's row-block step over the concrete columns,
folds the export's `TaskGroup.pipelined` peels back into one loop (`peel3`) and closes the walk
arithmetic with `ring_nf`. `facts` are the export's design and parameter definitions plus the
literal division facts its index arithmetic needs (the generator emits them).
-/

namespace IronVerify.Schema.WholeArray

open IronVerify

theorem rowBlocks (p : Params) (h : 0 < p.rows * p.m) (Mt : Nat) :
    (((p.rows * p.m * Mt : Nat) : Int) / ((p.rows * p.m : Nat) : Int)).toNat = Mt := by
  rw [Nat.cast_mul, Int.mul_ediv_cancel_left _ (by exact_mod_cast h.ne')]
  simp

/-- The schema side of a channel, before the export is touched: its row blocks unrolled. -/
theorem schema_stream_dims (p : Params) (h : 0 < p.rows * p.m) (Mt Kt Nt : Nat) (ch : String) :
    ((schema p).seq (p.shapeOf Mt Kt Nt)).stream ch =
      ((List.range Mt).map fun rb =>
        (step p (p.rows * p.m * Mt : Nat) (p.k * Kt : Nat) (p.n * p.cols * Nt : Nat)
          (rb : Int)).stream ch).flatten := by
  show (wseq p _ _ _).stream ch = _
  rw [schemaStream, rowBlocks p h]

/-- `peel3` for a step that is the same every row block (B's transfers). -/
theorem peel3_rep {α : Type} (x : List α) (n : Nat) :
    (if 0 < n then x ++
      if 1 < n then x ++
        if 2 < n then x ++ (List.replicate (n - 3) x).flatten
        else []
      else []
    else []) = (List.replicate n x).flatten := by
  rcases n with _ | _ | _ | n
  · simp
  · simp
  · simp [List.replicate_succ]
  · simp [List.replicate_succ, show 1 < n + 1 + 1 + 1 by omega, show 2 < n + 1 + 1 + 1 by omega]

syntax "wa_stream" "[" Lean.Parser.Tactic.simpLemma,* "]" : tactic

macro_rules
  | `(tactic| wa_stream [$facts,*]) => `(tactic| (
      rw [schema_stream_dims _ (by decide)]
      simp (config := { decide := true }) only [step, Seq.ofList, Seq.stream, List.range_succ,
        List.range_zero, List.nil_append, List.flatMap_cons, List.flatMap_nil, List.append_nil,
        List.cons_append, List.singleton_append, Params.nA, aL3, bL3, cL3, Function.comp_def,
        Nat.lt_irrefl, decide_true, decide_false, ite_true, ite_false, min_self, $facts,*]
      simp [Int.fdiv_eq_ediv, Params.shapeOf, Design.host,
        Seq.stream, $facts,*]
      try simp only [flatMap_cast, List.map_map, Function.comp_def]
      first | (conv_rhs => rw [← peel3]) | (conv_rhs => rw [← peel3_rep])
      simp [tapA, tapB, tapC, nTiles, Params.shimFirst, Params.shimRows, Params.nA, $facts,*]
      try push_cast
      try ring_nf))

end IronVerify.Schema.WholeArray
