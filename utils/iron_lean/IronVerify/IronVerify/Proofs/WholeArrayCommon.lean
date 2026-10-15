-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Lemmas
import IronVerify.Spec.WholeArray

/-!
Facts every whole_array configuration shares: the shape `TaskGroup.pipelined(n, depth=4)` gives the
runtime sequence, the memtile-to-core walks of the matmul kernel's operands, and the segments of a
four-way join.
-/

namespace IronVerify.Proofs.WholeArray

open IronVerify Spec.WholeArray

theorem flatMap_cast (l : List Nat) :
    List.flatMap (fun a : Nat => [(Nat.cast a : Int)]) l = l.map (Nat.cast : Nat → Int) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [List.flatMap_cons, ih]; rfl

/-- `TaskGroup.pipelined(n, depth=4)`: three guarded peels, then a loop from 3. -/
theorem peel3 {α : Type} (g : Nat → List α) (n : Nat) :
    (if 0 < n then g 0 ++
      if 1 < n then g 1 ++
        if 2 < n then g 2 ++ ((List.range (n - 3)).map fun i => g (3 + i)).flatten
        else []
      else []
    else []) = ((List.range n).map g).flatten := by
  rcases n with _ | _ | _ | n
  · simp
  · simp
  · simp [List.range_succ]
  · rw [show n + 1 + 1 + 1 = 3 + n by omega, List.range_add]
    simp [List.range_succ, show 1 < 3 + n by omega, show 2 < 3 + n by omega, Function.comp_def]

/-- The memtile walk of an A segment onto the stream to a core (r × s = 4 × 4 blocks). -/
def wA : Walk := { offset := 0, dims := [(16, 256), (16, 4), (4, 64), (4, 1)] }

theorem wA_layoutA (a b : Nat) (ha : a < 64) (hb : b < 64) :
    layoutA a b < 4096 ∧ wA.idx (layoutA a b) = a * 64 + b := by
  have hfits : Fits wA.dims [a / 4, b / 4, a % 4, b % 4] := by
    simp [Fits, wA]; omega
  have henc : encode wA.dims [a / 4, b / 4, a % 4, b % 4] = layoutA a b := by
    simp [encode, wA, dimsLen, layoutA]; ring
  refine ⟨henc ▸ wA.encode_lt_len hfits, ?_⟩
  rw [← henc, wA.idx_encode hfits]
  simp [dot, wA]
  omega

/-- The memtile walk of a B object onto the stream to a core (s × t = 4 × 8 blocks). -/
def wB : Walk := { offset := 0, dims := [(16, 128), (4, 8), (4, 32), (8, 1)] }

theorem wB_layoutB (a b : Nat) (ha : a < 64) (hb : b < 32) :
    layoutB a b < 2048 ∧ wB.idx (layoutB a b) = a * 32 + b := by
  have hfits : Fits wB.dims [a / 4, b / 8, a % 4, b % 8] := by
    simp [Fits, wB]; omega
  have henc : encode wB.dims [a / 4, b / 8, a % 4, b % 8] = layoutB a b := by
    simp [encode, wB, dimsLen, layoutB]; ring
  refine ⟨henc ▸ wB.encode_lt_len hfits, ?_⟩
  rw [← henc, wB.idx_encode hfits]
  simp [dot, wB]
  omega

/-- The memtile walk of a C object onto the stream to the shim (r × t = 4 × 8 blocks). -/
def wC : Walk := { offset := 0, dims := [(16, 128), (4, 8), (4, 32), (8, 1)] }

theorem wC_layoutC (a b : Nat) (ha : a < 64) (hb : b < 32) :
    wC.idx (a * 32 + b) = layoutC a b ∧ layoutC a b < 2048 := by
  have hfits : Fits wC.dims [a / 4, a % 4, b / 8, b % 8] := by
    simp [Fits, wC]; omega
  have henc : encode wC.dims [a / 4, a % 4, b / 8, b % 8] = a * 32 + b := by
    simp [encode, wC, dimsLen]; omega
  rw [← henc, wC.idx_encode hfits]
  simp [dot, wC, layoutC]
  omega

theorem segmentOf_four (k x : Nat) (hx : x < 2048) (hk : k < 4) :
    segmentOf [0, 2048, 4096, 6144] (k * 2048 + x) = some k := by
  unfold segmentOf
  simp only [List.length_cons, List.length_nil]
  rw [show List.range 4 = [0, 1, 2, 3] from rfl]
  simp only [List.reverse_cons, List.reverse_nil, List.nil_append, List.cons_append, List.find?_cons]
  interval_cases k <;> simp <;> (repeat' split) <;> simp_all <;> omega

end IronVerify.Proofs.WholeArray
