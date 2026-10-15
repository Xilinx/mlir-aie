-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Mathlib
import IronVerify.Schema.WholeArray

/-!
The schema's fifo names with symbolic indices: each family `prefix ++ toString i` is injective,
the six families are pairwise distinct, and `Net.fifo?` finds each fifo of `wnet p` by name.
Everything else in the schema proofs compares names only through these lemmas.
-/

namespace IronVerify.Schema.WholeArray.Names

open IronVerify IronVerify.Schema.WholeArray

/-! ### Strings -/

theorem toString_inj {i j : Nat} : toString i = toString j ↔ i = j := by
  simp only [Nat.toString_eq_repr]; exact Nat.repr_inj

theorem us_not_mem_toString (n : Nat) : '_' ∉ (toString n).toList := by
  intro h
  simp [Nat.toString_eq_repr, Nat.repr] at h

theorem append_cancel {pre a b : String} : pre ++ a = pre ++ b ↔ a = b := by
  constructor
  · intro h
    rw [String.ext_iff, String.toList_append, String.toList_append] at h
    exact String.ext_iff.mpr (List.append_cancel_left h)
  · rintro rfl; rfl

/-- Two strings that start with different prefixes of the same length differ. -/
theorem append_ne {pre₁ pre₂ a b : String} (hl : pre₁.length = pre₂.length)
    (hne : pre₁ ≠ pre₂) : pre₁ ++ a ≠ pre₂ ++ b := by
  intro h
  rw [String.ext_iff, String.toList_append, String.toList_append] at h
  have := (List.append_inj h (by simpa [String.length] using hl)).1
  exact hne (String.ext_iff.mpr this)

/-- Splitting at the first `'_'`. -/
theorem split_us : ∀ {a a' b b' : List Char}, '_' ∉ a → '_' ∉ a' →
    a ++ '_' :: b = a' ++ '_' :: b' → a = a' ∧ b = b'
  | [], [], _, _, _, _, h => by simpa using h
  | [], x :: a', _, _, _, h', h => by
    simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
    exact absurd (h.1 ▸ List.mem_cons_self) h'
  | x :: a, [], _, _, h, _, h' => by
    simp only [List.nil_append, List.cons_append, List.cons.injEq] at h'
    exact absurd (h'.1 ▸ List.mem_cons_self) h
  | x :: a, y :: a', _, _, h, h', he => by
    simp only [List.cons_append, List.cons.injEq] at he
    obtain ⟨rfl, he⟩ := he
    have := split_us (fun hm => h (List.mem_cons_of_mem _ hm))
      (fun hm => h' (List.mem_cons_of_mem _ hm)) he
    exact ⟨by rw [this.1], this.2⟩

/-! ### The name families -/

theorem aL2_eq (i : Nat) : aL2 i = "A_L2L1_" ++ toString i := rfl
theorem aL3_eq (i : Nat) : aL3 i = "A_L3L2_" ++ toString i := rfl
theorem bL2_eq (i : Nat) : bL2 i = "B_L2L1_" ++ toString i := rfl
theorem bL3_eq (i : Nat) : bL3 i = "B_L3L2_" ++ toString i := rfl
theorem cL3_eq (i : Nat) : cL3 i = "C_L2L3_" ++ toString i := rfl
theorem cL1_eq (c row : Nat) : cL1 c row = "C_L1L2_" ++ (toString c ++ ("_" ++ toString row)) := by
  show "C_L1L2_" ++ toString c ++ "_" ++ toString row = _
  simp only [String.append_assoc]

@[simp] theorem aL2_inj {i j : Nat} : aL2 i = aL2 j ↔ i = j := by
  rw [aL2_eq, aL2_eq, append_cancel, toString_inj]
@[simp] theorem aL3_inj {i j : Nat} : aL3 i = aL3 j ↔ i = j := by
  rw [aL3_eq, aL3_eq, append_cancel, toString_inj]
@[simp] theorem bL2_inj {i j : Nat} : bL2 i = bL2 j ↔ i = j := by
  rw [bL2_eq, bL2_eq, append_cancel, toString_inj]
@[simp] theorem bL3_inj {i j : Nat} : bL3 i = bL3 j ↔ i = j := by
  rw [bL3_eq, bL3_eq, append_cancel, toString_inj]
@[simp] theorem cL3_inj {i j : Nat} : cL3 i = cL3 j ↔ i = j := by
  rw [cL3_eq, cL3_eq, append_cancel, toString_inj]

@[simp] theorem cL1_inj {c row c' row' : Nat} : cL1 c row = cL1 c' row' ↔ c = c' ∧ row = row' := by
  constructor
  · intro h
    rw [cL1_eq, cL1_eq, append_cancel, String.ext_iff] at h
    simp only [String.toList_append] at h
    have h' : (toString c).toList ++ '_' :: (toString row).toList =
        (toString c').toList ++ '_' :: (toString row').toList := by simpa using h
    obtain ⟨h1, h2⟩ := split_us (us_not_mem_toString c) (us_not_mem_toString c') h'
    exact ⟨toString_inj.mp (String.ext_iff.mpr h1), toString_inj.mp (String.ext_iff.mpr h2)⟩
  · rintro ⟨rfl, rfl⟩; rfl

/-- The family a name belongs to: its first seven characters. -/
def head7 (s : String) : List Char := s.toList.take 7

theorem head7_append {pre : String} (h : pre.length = 7) (a : String) :
    head7 (pre ++ a) = pre.toList := by
  have : pre.toList.length = 7 := by simpa [String.length] using h
  simp [head7, String.toList_append, List.take_append_of_le_length (by omega : 7 ≤ pre.toList.length),
    List.take_of_length_le (by omega : pre.toList.length ≤ 7)]

@[simp] theorem head7_aL2 (i : Nat) : head7 (aL2 i) = "A_L2L1_".toList := head7_append rfl _
@[simp] theorem head7_aL3 (i : Nat) : head7 (aL3 i) = "A_L3L2_".toList := head7_append rfl _
@[simp] theorem head7_bL2 (i : Nat) : head7 (bL2 i) = "B_L2L1_".toList := head7_append rfl _
@[simp] theorem head7_bL3 (i : Nat) : head7 (bL3 i) = "B_L3L2_".toList := head7_append rfl _
@[simp] theorem head7_cL3 (i : Nat) : head7 (cL3 i) = "C_L2L3_".toList := head7_append rfl _
@[simp] theorem head7_cL1 (c row : Nat) : head7 (cL1 c row) = "C_L1L2_".toList := by
  rw [cL1_eq]; exact head7_append rfl _

/-- Names from different families differ. -/
theorem ne_of_head7 {x y : String} (h : head7 x ≠ head7 y) : x ≠ y := fun e => h (e ▸ rfl)

/-! Pairwise distinct families, in both orientations, for `simp`. -/

section Distinct

variable {i j c row : Nat}

@[simp] theorem aL2_ne_aL3 : (aL2 i = aL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL2_ne_bL2 : (aL2 i = bL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL2_ne_bL3 : (aL2 i = bL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL2_ne_cL3 : (aL2 i = cL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL2_ne_cL1 : (aL2 i = cL1 c row) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL3_ne_aL2 : (aL3 i = aL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL3_ne_bL2 : (aL3 i = bL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL3_ne_bL3 : (aL3 i = bL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL3_ne_cL3 : (aL3 i = cL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem aL3_ne_cL1 : (aL3 i = cL1 c row) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL2_ne_aL2 : (bL2 i = aL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL2_ne_aL3 : (bL2 i = aL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL2_ne_bL3 : (bL2 i = bL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL2_ne_cL3 : (bL2 i = cL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL2_ne_cL1 : (bL2 i = cL1 c row) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL3_ne_aL2 : (bL3 i = aL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL3_ne_aL3 : (bL3 i = aL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL3_ne_bL2 : (bL3 i = bL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL3_ne_cL3 : (bL3 i = cL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem bL3_ne_cL1 : (bL3 i = cL1 c row) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL3_ne_aL2 : (cL3 i = aL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL3_ne_aL3 : (cL3 i = aL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL3_ne_bL2 : (cL3 i = bL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL3_ne_bL3 : (cL3 i = bL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL3_ne_cL1 : (cL3 i = cL1 c row) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL1_ne_aL2 : (cL1 c row = aL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL1_ne_aL3 : (cL1 c row = aL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL1_ne_bL2 : (cL1 c row = bL2 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL1_ne_bL3 : (cL1 c row = bL3 j) = False := eq_false (ne_of_head7 (by simp))
@[simp] theorem cL1_ne_cL3 : (cL1 c row = cL3 j) = False := eq_false (ne_of_head7 (by simp))

end Distinct

/-! ### Lookups in `wnet p` -/

/-- `find?` on `List.range n` for a predicate that holds exactly at `x < n`. -/
theorem find?_range_eq (P : Nat → Bool) {x n : Nat} (hP : ∀ i, P i = true ↔ i = x)
    (hx : x < n) : (List.range n).find? P = some x := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.find?_append]
    by_cases h : x < n
    · rw [ih h]; rfl
    · have hxn : x = n := by omega
      subst hxn
      have : (List.range x).find? P = none :=
        List.find?_eq_none.mpr fun i hi => by
          rw [hP]; have := List.mem_range.mp hi; omega
      rw [this]
      simp [(hP x).mpr rfl]

@[simp] theorem find?_false {α : Type} (l : List α) : l.find? (fun _ => false) = none :=
  List.find?_eq_none.mpr fun _ _ => Bool.false_ne_true

@[simp] theorem findSome?_none {α β : Type} (l : List α) :
    l.findSome? (fun _ => (none : Option β)) = none := by
  induction l <;> simp_all

/-- `findSome?` on `List.range n` for a function that is `some v` exactly at `x < n`. -/
theorem findSome?_range_eq {β : Type} (f : Nat → Option β) {x n : Nat} {v : β}
    (hf : ∀ i, i ≠ x → f i = none) (hx : f x = some v) (hxn : x < n) :
    (List.range n).findSome? f = some v := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.findSome?_append]
    by_cases h : x < n
    · rw [ih h]; rfl
    · have hxn : x = n := by omega
      subst hxn
      have : (List.range x).findSome? f = none :=
        List.findSome?_eq_none_iff.mpr fun i hi => hf i (by have := List.mem_range.mp hi; omega)
      rw [this]
      simp [hx]

variable (p : Params)

@[simp] theorem aL2Fifo_name (i : Nat) : (aL2Fifo p i).name = aL2 i := rfl
@[simp] theorem aL3Fifo_name (i : Nat) : (aL3Fifo p i).name = aL3 i := rfl
@[simp] theorem bL2Fifo_name (i : Nat) : (bL2Fifo p i).name = bL2 i := rfl
@[simp] theorem bL3Fifo_name (i : Nat) : (bL3Fifo p i).name = bL3 i := rfl
@[simp] theorem cL1Fifo_name (c row : Nat) : (cL1Fifo p c row).name = cL1 c row := rfl
@[simp] theorem cL3Fifo_name (i : Nat) : (cL3Fifo p i).name = cL3 i := rfl

theorem fifo?_aL2 {row : Nat} (h : row < p.rows) :
    (wnet p).fifo? (aL2 row) = some (aL2Fifo p row) := by
  simp only [Net.fifo?, wnet, List.find?_append, List.find?_map, List.find?_flatMap,
    Function.comp_def, aL2Fifo_name, aL3Fifo_name, bL2Fifo_name, bL3Fifo_name, cL1Fifo_name,
    cL3Fifo_name, aL2_inj, aL3_ne_aL2, bL2_ne_aL2, bL3_ne_aL2, cL1_ne_aL2, cL3_ne_aL2,
    decide_false, find?_false, Option.map_none, Option.or_none, findSome?_none]
  rw [find?_range_eq _ (fun i => by simp) h]
  rfl

theorem fifo?_aL3 {i : Nat} (h : i < p.nA) :
    (wnet p).fifo? (aL3 i) = some (aL3Fifo p i) := by
  simp only [Net.fifo?, wnet, List.find?_append, List.find?_map, List.find?_flatMap,
    Function.comp_def, aL2Fifo_name, aL3Fifo_name, bL2Fifo_name, bL3Fifo_name, cL1Fifo_name,
    cL3Fifo_name, aL3_inj, aL2_ne_aL3, bL2_ne_aL3, bL3_ne_aL3, cL1_ne_aL3, cL3_ne_aL3,
    decide_false, find?_false, Option.map_none, Option.or_none, Option.none_or, findSome?_none]
  rw [find?_range_eq _ (fun j => by simp) h]
  simp

theorem fifo?_bL2 {c : Nat} (h : c < p.cols) :
    (wnet p).fifo? (bL2 c) = some (bL2Fifo p c) := by
  simp only [Net.fifo?, wnet, List.find?_append, List.find?_map, List.find?_flatMap,
    Function.comp_def, aL2Fifo_name, aL3Fifo_name, bL2Fifo_name, bL3Fifo_name, cL1Fifo_name,
    cL3Fifo_name, bL2_inj, aL2_ne_bL2, aL3_ne_bL2, bL3_ne_bL2, cL1_ne_bL2, cL3_ne_bL2,
    decide_false, find?_false, Option.map_none, Option.or_none, Option.none_or, findSome?_none]
  rw [find?_range_eq _ (fun j => by simp) h]
  simp

theorem fifo?_bL3 {c : Nat} (h : c < p.cols) :
    (wnet p).fifo? (bL3 c) = some (bL3Fifo p c) := by
  simp only [Net.fifo?, wnet, List.find?_append, List.find?_map, List.find?_flatMap,
    Function.comp_def, aL2Fifo_name, aL3Fifo_name, bL2Fifo_name, bL3Fifo_name, cL1Fifo_name,
    cL3Fifo_name, bL3_inj, aL2_ne_bL3, aL3_ne_bL3, bL2_ne_bL3, cL1_ne_bL3, cL3_ne_bL3,
    decide_false, find?_false, Option.map_none, Option.or_none, Option.none_or, findSome?_none]
  rw [find?_range_eq _ (fun j => by simp) h]
  simp

theorem fifo?_cL3 {c : Nat} (h : c < p.cols) :
    (wnet p).fifo? (cL3 c) = some (cL3Fifo p c) := by
  simp only [Net.fifo?, wnet, List.find?_append, List.find?_map, List.find?_flatMap,
    Function.comp_def, aL2Fifo_name, aL3Fifo_name, bL2Fifo_name, bL3Fifo_name, cL1Fifo_name,
    cL3Fifo_name, cL3_inj, aL2_ne_cL3, aL3_ne_cL3, bL2_ne_cL3, bL3_ne_cL3, cL1_ne_cL3,
    decide_false, find?_false, Option.map_none, Option.or_none, Option.none_or, findSome?_none]
  rw [find?_range_eq _ (fun j => by simp) h]
  simp

theorem fifo?_cL1 {c row : Nat} (hc : c < p.cols) (hr : row < p.rows) :
    (wnet p).fifo? (cL1 c row) = some (cL1Fifo p c row) := by
  simp only [Net.fifo?, wnet, List.find?_append, List.find?_map, List.find?_flatMap,
    Function.comp_def, aL2Fifo_name, aL3Fifo_name, bL2Fifo_name, bL3Fifo_name, cL1Fifo_name,
    cL3Fifo_name, cL1_inj, aL2_ne_cL1, aL3_ne_cL1, bL2_ne_cL1, bL3_ne_cL1, cL3_ne_cL1,
    decide_false, find?_false, Option.map_none, Option.or_none, Option.none_or]
  refine findSome?_range_eq _ (fun x hx => ?_) ?_ hc
  · simp [hx]
  · rw [find?_range_eq _ (fun j => by simp) hr]; rfl

/-! ### Destinations of the links (`prodObj` locates a split's output with `idxOf?`) -/

/-- `idxOf?` on a mapped `List.range n` whose entries equal `a` exactly at `x < n`. -/
theorem idxOf?_map_range (g : Nat → String) {a : String} {x n : Nat}
    (hg : ∀ j, g j = a ↔ j = x) (hx : x < n) : ((List.range n).map g).idxOf? a = some x := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.idxOf?, List.findIdx?_append]
    by_cases h : x < n
    · rw [← List.idxOf?, ih h]; rfl
    · have hxn : x = n := by omega
      subst hxn
      have : ((List.range x).map g).findIdx? (· == a) = none :=
        List.findIdx?_eq_none_iff.mpr fun y hy => by
          obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hy
          have := List.mem_range.mp hj
          simpa [beq_iff_eq, hg] using (by omega : j ≠ x)
      rw [this]
      simp [(hg x).mpr rfl]

theorem idxOf?_aLink {i row : Nat} (h₁ : p.shimFirst i ≤ row)
    (h₂ : row < p.shimFirst i + p.shimRows i) :
    (aLinkD p i).dsts.idxOf? (aL2 row) = some (row - p.shimFirst i) :=
  idxOf?_map_range _ (fun j => by simp only [aL2_inj]; omega) (by omega)

@[simp] theorem idxOf?_bLink (c : Nat) : (bLinkD c).dsts.idxOf? (bL2 c) = some 0 := by
  simp [bLinkD, List.idxOf?]

@[simp] theorem idxOf?_cLink (c : Nat) : (cLinkD p c).dsts.idxOf? (cL3 c) = some 0 := by
  simp [cLinkD, List.idxOf?]

end IronVerify.Schema.WholeArray.Names
