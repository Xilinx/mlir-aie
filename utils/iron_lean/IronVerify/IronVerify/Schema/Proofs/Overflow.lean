-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Mathlib
import IronVerify.Schema.Defs
import IronVerify.Schema.Overflow

namespace IronVerify.Schema.WholeArray.Ovf

open IronVerify

theorem every_ofList (P : String → Nat → Walk → Prop) :
    ∀ xs : List Seq, (Seq.ofList xs).Every P ↔ ∀ x ∈ xs, x.Every P
  | [] => by simp [Seq.ofList, Seq.Every]
  | x :: xs => by simp [Seq.ofList, Seq.Every, every_ofList P xs]

variable (p : Params)

/-- Every transfer of the schema's sequence, for shapes given in tiles. -/
theorem every_wseq (P : String → Nat → Walk → Prop) (Mt : Nat) (K N : Int)
    (hrm : 0 < p.rows * p.m) :
    (wseq p ((p.rows * p.m * Mt : Nat) : Int) K N).Every P ↔
      ∀ rb < Mt, ∀ c < p.cols,
        P (cL3 c) 2 (tapC p ((p.rows * p.m * Mt : Nat) : Int) N rb c) ∧
        (c < p.nA → P (aL3 c) 0 (tapA p K N rb c)) ∧ P (bL3 c) 1 (tapB p K N c) := by
  have hdiv : ((p.rows * p.m * Mt : Nat) : Int) / ((p.rows * p.m : Nat) : Int) = Mt := by
    push_cast
    rw [mul_comm, Int.mul_ediv_cancel _ (by exact_mod_cast hrm.ne')]
  simp only [wseq, every_ofList, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq, Seq.Every, true_and, hdiv, sub_zero, Int.toNat_natCast]
  refine forall_congr' fun rb => imp_congr_right fun _ => ?_
  simp only [step, every_ofList, List.mem_flatMap, List.mem_range, zero_add]
  constructor
  · intro h c hc
    have h' := fun x hx => h x ⟨c, hc, hx⟩
    exact ⟨h' (.xfer _ _ _) (by simp), fun hcn => h' (.xfer (aL3 c) _ _) (by simp [hcn]),
      h' (.xfer (bL3 c) _ _) (by simp)⟩
  · rintro h x ⟨c, hc, hx⟩
    obtain ⟨hC, hA, hB⟩ := h c hc
    simp only [List.mem_append, List.mem_singleton] at hx
    split_ifs at hx with hcn
    · simp only [List.mem_singleton] at hx
      rcases hx with (rfl | rfl) | rfl <;> simp_all [Seq.Every]
    · simp only [List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl <;> simp_all [Seq.Every]

section Shims

variable {p}

theorem nA_pos (hp : p.Valid) : 0 < p.nA := Nat.lt_min.mpr ⟨hp.rows_pos, hp.cols_pos⟩

/-- Shim `i`'s run of rows ends where shim `i + 1`'s starts. -/
theorem shimEnd (i : Nat) :
    p.shimFirst i + p.shimRows i = (i + 1) * (p.rows / p.nA) + min (i + 1) (p.rows % p.nA) := by
  unfold Params.shimFirst Params.shimRows
  split_ifs with h
  · rw [min_eq_left h.le, min_eq_left (by omega : i + 1 ≤ p.rows % p.nA)]; ring
  · rw [min_eq_right (by omega), min_eq_right (by omega)]; ring

theorem shimEnd_le {i : Nat} (hi : i < p.nA) :
    p.shimFirst i + p.shimRows i ≤ p.rows := by
  rw [shimEnd]
  have hq := Nat.div_add_mod p.rows p.nA
  have : (i + 1) * (p.rows / p.nA) ≤ p.nA * (p.rows / p.nA) :=
    Nat.mul_le_mul_right _ (by omega)
  have : min (i + 1) (p.rows % p.nA) ≤ p.rows % p.nA := min_le_right _ _
  omega

theorem shimEnd_last (hp : p.Valid) :
    p.shimFirst (p.nA - 1) + p.shimRows (p.nA - 1) = p.rows := by
  have h0 := nA_pos hp
  rw [shimEnd, Nat.sub_add_cancel h0, min_eq_right (Nat.mod_lt _ h0).le]
  exact Nat.div_add_mod p.rows p.nA

end Shims

section Taps

variable {p} (hp : p.Valid) (Mt Kt Nt : Nat)
include hp

theorem nTiles_Nn : nTiles p (p.Nn Nt : Nat) = Nt := by
  have : (0 : Int) < (p.n * p.cols : Nat) := by
    exact_mod_cast Nat.mul_pos hp.n_pos hp.cols_pos
  unfold nTiles Params.Nn
  push_cast
  rw [Int.mul_ediv_cancel_left _ (by exact_mod_cast this.ne')]

theorem Kn_div : ((p.Kn Kt : Nat) : Int) / (p.k : Nat) = Kt := by
  unfold Params.Kn
  push_cast
  rw [Int.mul_ediv_cancel_left _ (by exact_mod_cast hp.k_pos.ne')]

theorem tapA_maxIdx (rb : Int) (i : Nat) :
    (tapA p (p.Kn Kt : Nat) (p.Nn Nt : Nat) rb i).maxIdx + 1 =
      (rb * (p.rows * p.m : Nat) + ((p.shimFirst i + p.shimRows i) * p.m : Nat)) *
        (p.Kn Kt : Nat) := by
  simp only [Walk.maxIdx, tapA, nTiles_Nn hp, Kn_div hp, List.map, List.sum_cons,
    List.sum_nil]
  unfold Params.Kn
  push_cast
  ring

theorem tapB_maxIdx (c : Nat) :
    (tapB p (p.Kn Kt : Nat) (p.Nn Nt : Nat) c).maxIdx + 1 =
      if p.bColMaj then ((c + 1) * p.n + (Nt - 1 : Int) * (p.n * p.cols : Nat)) * (p.Kn Kt : Nat)
      else ((p.Kn Kt : Nat) - 1 : Int) * (p.Nn Nt : Nat) + (Nt - 1 : Int) * (p.n * p.cols : Nat) +
        (c + 1) * p.n := by
  unfold tapB
  split_ifs <;>
  · simp only [Walk.maxIdx, nTiles_Nn hp, Kn_div hp, List.map, List.sum_cons, List.sum_nil]
    simp only [Params.Kn, Params.Nn]
    push_cast
    ring

theorem tapC_maxIdx (rb : Int) (c : Nat) :
    (tapC p (p.Mn Mt : Nat) (p.Nn Nt : Nat) rb c).maxIdx + 1 =
      if p.cColMaj then (rb + 1) * (p.rows * p.m : Nat) +
        ((c + 1) * p.n + (Nt - 1 : Int) * (p.n * p.cols : Nat) - 1) * (p.Mn Mt : Nat)
      else rb * (p.rows * p.m : Nat) * (p.Nn Nt : Nat) +
        ((p.rows * p.m : Nat) - 1 : Int) * (p.Nn Nt : Nat) +
        (Nt - 1 : Int) * (p.n * p.cols : Nat) + (c + 1) * p.n := by
  unfold tapC
  split_ifs <;>
  · simp only [Walk.maxIdx, nTiles_Nn hp, List.map, List.sum_cons, List.sum_nil]
    simp only [Params.Mn, Params.Nn]
    push_cast
    ring

/-- Every offset, size and stride of the schema's taps is non-negative. -/
def NonNeg (w : Walk) : Prop := 0 ≤ w.offset ∧ ∀ d ∈ w.dims, 0 ≤ d.1 ∧ 0 ≤ d.2

theorem tapA_nonneg (rb i : Nat) : NonNeg (tapA p (p.Kn Kt : Nat) (p.Nn Nt : Nat) rb i) := by
  simp only [NonNeg, tapA, nTiles_Nn hp, Kn_div hp, List.mem_cons, List.mem_nil_iff, or_false,
    forall_eq_or_imp, forall_eq]
  refine ⟨by positivity, ?_⟩
  simp only [Params.Kn]
  refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> positivity

theorem tapB_nonneg (c : Nat) : NonNeg (tapB p (p.Kn Kt : Nat) (p.Nn Nt : Nat) c) := by
  unfold tapB
  split_ifs <;>
  · simp only [NonNeg, nTiles_Nn hp, Kn_div hp, List.mem_cons, List.mem_nil_iff, or_false,
      forall_eq_or_imp, forall_eq]
    refine ⟨by positivity, ?_⟩
    refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> positivity

theorem tapC_nonneg (rb c : Nat) : NonNeg (tapC p (p.Mn Mt : Nat) (p.Nn Nt : Nat) rb c) := by
  unfold tapC
  split_ifs <;>
  · simp only [NonNeg, nTiles_Nn hp, List.mem_cons, List.mem_nil_iff, or_false,
      forall_eq_or_imp, forall_eq]
    refine ⟨by positivity, ?_⟩
    first
    | refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> positivity
    | refine ⟨⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩⟩ <;> positivity

end Taps

section Bounds

variable {p} (hp : p.Valid) {Mt Kt Nt : Nat}
include hp

theorem tapA_le {rb i : Nat} (hrb : rb < Mt) (hi : i < p.nA) :
    (tapA p (p.Kn Kt : Nat) (p.Nn Nt : Nat) rb i).maxIdx + 1 ≤ (p.Mn Mt * p.Kn Kt : Nat) := by
  rw [tapA_maxIdx hp]
  have h1 := shimEnd_le hi
  have h2 : rb * (p.rows * p.m) + (p.shimFirst i + p.shimRows i) * p.m ≤ p.rows * p.m * Mt := by
    have := Nat.mul_le_mul_right (p.rows * p.m) (show rb + 1 ≤ Mt from hrb)
    have := Nat.mul_le_mul_right p.m h1
    nlinarith
  have h3 := Nat.mul_le_mul_right (p.Kn Kt) h2
  simp only [Params.Mn]
  exact_mod_cast h3

theorem tapA_last (hM : 0 < Mt) :
    (tapA p (p.Kn Kt : Nat) (p.Nn Nt : Nat) (Mt - 1 : Nat) (p.nA - 1)).maxIdx + 1 =
      (p.Mn Mt * p.Kn Kt : Nat) := by
  rw [tapA_maxIdx hp, shimEnd_last hp]
  simp only [Params.Mn]
  push_cast [Nat.cast_sub hM]
  ring

theorem tapB_le {c : Nat} (hc : c < p.cols) :
    (tapB p (p.Kn Kt : Nat) (p.Nn Nt : Nat) c).maxIdx + 1 ≤ (p.Kn Kt * p.Nn Nt : Nat) := by
  rw [tapB_maxIdx hp]
  have hc' : (c : Int) + 1 ≤ p.cols := by exact_mod_cast hc
  have hn : (0 : Int) ≤ p.n := by positivity
  have hK : (0 : Int) ≤ (p.Kn Kt : Nat) := by positivity
  have := mul_le_mul_of_nonneg_right hc' hn
  split_ifs
  · simp only [Params.Nn] at *
    push_cast at *
    nlinarith [mul_le_mul_of_nonneg_right this hK]
  · simp only [Params.Nn] at *
    push_cast at *
    nlinarith

theorem tapB_last :
    (tapB p (p.Kn Kt : Nat) (p.Nn Nt : Nat) (p.cols - 1)).maxIdx + 1 = (p.Kn Kt * p.Nn Nt : Nat) := by
  rw [tapB_maxIdx hp]
  have := hp.cols_pos
  split_ifs <;>
  · simp only [Params.Nn]
    push_cast [Nat.cast_sub (show 1 ≤ p.cols from this)]
    ring

theorem tapC_le {rb c : Nat} (hrb : rb < Mt) (hc : c < p.cols) (hN : 0 < Nt) :
    (tapC p (p.Mn Mt : Nat) (p.Nn Nt : Nat) rb c).maxIdx + 1 ≤ (p.Mn Mt * p.Nn Nt : Nat) := by
  rw [tapC_maxIdx hp]
  have hc' : (c : Int) + 1 ≤ p.cols := by exact_mod_cast hc
  have hrb' : (rb : Int) + 1 ≤ Mt := by exact_mod_cast hrb
  have hN' : (1 : Int) ≤ Nt := by exact_mod_cast hN
  have hn : (0 : Int) ≤ p.n := by positivity
  have hR : (0 : Int) ≤ p.rows * p.m := by positivity
  have hRpos : (1 : Int) ≤ p.rows * p.m := by
    exact_mod_cast Nat.mul_pos hp.rows_pos hp.m_pos
  have hcols : (0 : Int) ≤ p.cols := by positivity
  have hNn : (0 : Int) ≤ (p.Nn Nt : Nat) := by positivity
  have hMn : (0 : Int) ≤ (p.Mn Mt : Nat) := by positivity
  have h1 := mul_le_mul_of_nonneg_right hc' hn
  split_ifs
  · simp only [Params.Mn, Params.Nn] at *
    push_cast at *
    have h2 := mul_le_mul_of_nonneg_right hrb' hR
    have h3 : ((c : Int) + 1) * p.n + (Nt - 1) * (p.n * p.cols) - 1 ≤
        p.n * p.cols * Nt - 1 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_right h3 hMn]
  · simp only [Params.Mn, Params.Nn] at *
    push_cast at *
    have h2 := mul_le_mul_of_nonneg_right hrb' (mul_nonneg hR hNn)
    nlinarith

theorem tapC_last (hM : 0 < Mt) :
    (tapC p (p.Mn Mt : Nat) (p.Nn Nt : Nat) (Mt - 1 : Nat) (p.cols - 1)).maxIdx + 1 =
      (p.Mn Mt * p.Nn Nt : Nat) := by
  rw [tapC_maxIdx hp]
  have := hp.cols_pos
  split_ifs <;>
  · simp only [Params.Mn, Params.Nn]
    push_cast [Nat.cast_sub (show 1 ≤ p.cols from this), Nat.cast_sub hM]
    ring

end Bounds

variable {p}

/-- `Theorems.hostFits_iff`: the compiler's runtime host-buffer guard accepts the schema's
transfers exactly when A, B and C fit their host buffers. -/
theorem hostFits_iff_aux (p : Params) (hp : p.Valid) (Mt Kt Nt : Nat) (hM : 0 < Mt) (_hK : 0 < Kt)
    (hN : 0 < Nt) (cap : Nat → Nat) :
    (wseq p (p.Mn Mt) (p.Kn Kt) (p.Nn Nt)).HostFits cap ↔
      p.Mn Mt * p.Kn Kt ≤ cap 0 ∧ p.Kn Kt * p.Nn Nt ≤ cap 1 ∧ p.Mn Mt * p.Nn Nt ≤ cap 2 := by
  have hrm : 0 < p.rows * p.m := Nat.mul_pos hp.rows_pos hp.m_pos
  unfold Seq.HostFits
  have hMn : ((p.Mn Mt : Nat) : Int) = ((p.rows * p.m * Mt : Nat) : Int) := rfl
  rw [hMn, every_wseq p _ Mt _ _ hrm, ← hMn]
  have hlt (x : Int) (b : Nat) : x < (b : Int) ↔ x + 1 ≤ (b : Int) := by omega
  constructor
  · intro h
    have hc := hp.cols_pos
    have hA := nA_pos hp
    have hnc : p.nA ≤ p.cols := min_le_right _ _
    obtain ⟨hC, -, hB⟩ := h (Mt - 1) (by omega) (p.cols - 1) (by omega)
    obtain ⟨-, hA', -⟩ := h (Mt - 1) (by omega) (p.nA - 1) (by omega)
    have hA'' := (hA' (by omega)).2.2
    rw [hlt, tapA_last hp hM] at hA''
    rw [hlt, tapB_last hp] at hB
    rw [hlt, tapC_last hp hM] at hC
    exact ⟨by exact_mod_cast hA'', by exact_mod_cast hB.2.2, by exact_mod_cast hC.2.2⟩
  · rintro ⟨h0, h1, h2⟩ rb hrb c hc
    refine ⟨⟨(tapC_nonneg hp Mt Nt rb c).1, (tapC_nonneg hp Mt Nt rb c).2, ?_⟩,
      fun hcn => ⟨(tapA_nonneg hp Kt Nt rb c).1, (tapA_nonneg hp Kt Nt rb c).2, ?_⟩,
      ⟨(tapB_nonneg hp Kt Nt c).1, (tapB_nonneg hp Kt Nt c).2, ?_⟩⟩
    · rw [hlt]; exact (tapC_le hp hrb hc hN).trans (by exact_mod_cast h2)
    · rw [hlt]; exact (tapA_le hp hrb hcn).trans (by exact_mod_cast h0)
    · rw [hlt]; exact (tapB_le hp hc).trans (by exact_mod_cast h1)

theorem ok_ofList : ∀ xs : List Seq, (Seq.ofList xs).ok = xs.all Seq.ok
  | [] => rfl
  | x :: xs => by simp [Seq.ofList, Seq.ok, ok_ofList xs]

/-- The shape behind the guards: `(wseq p M K N).ok` gives `M`, `K` and `N` in whole tiles. -/
theorem shape_of_ok (hp : p.Valid) {M K N : Int} (hok : (wseq p M K N).ok = true) :
    ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧
      M = (p.Mn Mt : Nat) ∧ K = (p.Kn Kt : Nat) ∧ N = (p.Nn Nt : Nat) := by
  simp only [wseq, ok_ofList, List.all_cons, List.all_nil, Seq.ok, Bool.and_true,
    Bool.and_eq_true, decide_eq_true_eq] at hok
  obtain ⟨hM, hK, hN, hMd, hKd, hNd, -⟩ := hok
  obtain ⟨a, ha⟩ := Int.dvd_of_emod_eq_zero hMd
  obtain ⟨b, hb⟩ := Int.dvd_of_emod_eq_zero hKd
  obtain ⟨c, hc⟩ := Int.dvd_of_emod_eq_zero hNd
  have hrm : (0 : Int) < (p.rows * p.m : Nat) := by exact_mod_cast Nat.mul_pos hp.rows_pos hp.m_pos
  have hk : (0 : Int) < (p.k : Nat) := by exact_mod_cast hp.k_pos
  have hnc : (0 : Int) < (p.n * p.cols : Nat) := by exact_mod_cast Nat.mul_pos hp.n_pos hp.cols_pos
  have ha0 : 0 < a := by rw [ha] at hM; exact pos_of_mul_pos_right hM hrm.le
  have hb0 : 0 < b := by rw [hb] at hK; exact pos_of_mul_pos_right hK hk.le
  have hc0 : 0 < c := by rw [hc] at hN; exact pos_of_mul_pos_right hN hnc.le
  refine ⟨a.toNat, b.toNat, c.toNat, by omega, by omega, by omega, ?_, ?_, ?_⟩ <;>
  simp only [Params.Mn, Params.Kn, Params.Nn, ha, hb, hc] <;>
  push_cast [Int.toNat_of_nonneg ha0.le, Int.toNat_of_nonneg hb0.le, Int.toNat_of_nonneg hc0.le] <;>
  ring

theorem fitsS64_of {x : Int} (h0 : 0 ≤ x) (h : x ≤ 2 ^ 62) : fitsS 64 x = true := by
  simp only [fitsS, decide_eq_true_eq]; constructor <;> omega

/-- A walk whose sizes are positive and whose values all lie in `[0, 2^62]` fits `i64`
everywhere, offset included (it is at most the furthest element). -/
theorem fits_of_bounds (w : Walk) (hnn : NonNeg w) (hsz : ∀ d ∈ w.dims, 1 ≤ d.1)
    (hd : ∀ d ∈ w.dims, d.1 ≤ 2 ^ 62 ∧ d.2 ≤ 2 ^ 62) (hmax : w.maxIdx ≤ 2 ^ 62) :
    fitsS 64 w.offset ∧ (∀ d ∈ w.dims, fitsS 64 d.1 ∧ fitsS 64 d.2) ∧ fitsS 64 w.maxIdx := by
  have hsum : 0 ≤ (w.dims.map fun (s, t) => (s - 1) * t).sum := by
    apply List.sum_nonneg
    intro x hx
    obtain ⟨⟨s, t⟩, hm, rfl⟩ := List.mem_map.mp hx
    exact mul_nonneg (by linarith [hsz _ hm]) (hnn.2 _ hm).2
  have hoff : w.offset ≤ w.maxIdx := by unfold Walk.maxIdx; linarith
  refine ⟨fitsS64_of hnn.1 (hoff.trans hmax), fun d hm => ?_, fitsS64_of (hnn.1.trans hoff) hmax⟩
  exact ⟨fitsS64_of (hnn.2 d hm).1 (hd d hm).1, fitsS64_of (hnn.2 d hm).2 (hd d hm).2⟩

theorem fitsS32_lt {x : Int} (h : fitsS 32 x = true) : x < 2 ^ 31 := by
  simp only [fitsS, decide_eq_true_eq] at h; exact h.2

set_option maxRecDepth 20000 in
/-- `Theorems.noWrap64`. -/
theorem noWrap64_aux (p : Params) (hp : p.Valid) (M K N : Int) (hM : fitsS 32 M) (hK : fitsS 32 K)
    (hN : fitsS 32 N) (hok : (wseq p M K N).ok = true)
    (hsmall : p.rows * p.m < 2 ^ 16 ∧ p.k < 2 ^ 16 ∧ p.n * p.cols < 2 ^ 16) :
    (wseq p M K N).Every fun _ _ w =>
      fitsS 64 w.offset ∧ (∀ d ∈ w.dims, fitsS 64 d.1 ∧ fitsS 64 d.2) ∧ fitsS 64 w.maxIdx := by
  obtain ⟨Mt, Kt, Nt, hMt, hKt, hNt, rfl, rfl, rfl⟩ := shape_of_ok hp hok
  have hM' := fitsS32_lt hM
  have hK' := fitsS32_lt hK
  have hN' := fitsS32_lt hN
  obtain ⟨hrm, hk, hnc⟩ := hsmall
  have hrm0 : 0 < p.rows * p.m := Nat.mul_pos hp.rows_pos hp.m_pos
  simp only [Params.Mn, Params.Kn, Params.Nn] at hM' hK' hN'
  push_cast at hM' hK' hN'
  have hMn : ((p.Mn Mt : Nat) : Int) = ((p.rows * p.m * Mt : Nat) : Int) := rfl
  rw [hMn, every_wseq p _ Mt _ _ hrm0, ← hMn]
  have hMi : ((p.Mn Mt : Nat) : Int) < 2 ^ 31 := by simp only [Params.Mn]; push_cast; linarith
  have hKi : ((p.Kn Kt : Nat) : Int) < 2 ^ 31 := by simp only [Params.Kn]; push_cast; linarith
  have hNi : ((p.Nn Nt : Nat) : Int) < 2 ^ 31 := by simp only [Params.Nn]; push_cast; linarith
  have h0M : (0 : Int) ≤ (p.Mn Mt : Nat) := by positivity
  have h0K : (0 : Int) ≤ (p.Kn Kt : Nat) := by positivity
  have h0N : (0 : Int) ≤ (p.Nn Nt : Nat) := by positivity
  have hprod {x y : Int} (hx : x < 2 ^ 31) (hy : y < 2 ^ 31) (h0 : 0 ≤ y) :
      x * y ≤ 2 ^ 62 :=
    calc x * y ≤ 2 ^ 31 * y := mul_le_mul_of_nonneg_right hx.le h0
      _ ≤ 2 ^ 31 * 2 ^ 31 := mul_le_mul_of_nonneg_left hy.le (by norm_num)
      _ = 2 ^ 62 := by norm_num
  have hMK : ((p.Mn Mt * p.Kn Kt : Nat) : Int) ≤ 2 ^ 62 := by push_cast; exact hprod hMi hKi h0K
  have hKN : ((p.Kn Kt * p.Nn Nt : Nat) : Int) ≤ 2 ^ 62 := by push_cast; exact hprod hKi hNi h0N
  have hMN : ((p.Mn Mt * p.Nn Nt : Nat) : Int) ≤ 2 ^ 62 := by push_cast; exact hprod hMi hNi h0N
  have hsm : ((p.rows * p.m : Nat) : Int) < 2 ^ 31 := by
    have : ((p.rows * p.m : Nat) : Int) < 2 ^ 16 := by exact_mod_cast hrm
    linarith
  have hsk : (p.k : Int) < 2 ^ 31 := by
    have : (p.k : Int) < 2 ^ 16 := by exact_mod_cast hk
    linarith
  have hsn : ((p.n * p.cols : Nat) : Int) < 2 ^ 31 := by
    have : ((p.n * p.cols : Nat) : Int) < 2 ^ 16 := by exact_mod_cast hnc
    linarith
  have hr1 : 1 ≤ p.rows := hp.rows_pos
  have hm1 : 1 ≤ p.m := hp.m_pos
  have hn1 : 1 ≤ p.n := hp.n_pos
  have hc1 : 1 ≤ p.cols := hp.cols_pos
  have hk1 : 1 ≤ p.k := hp.k_pos
  have hrow : p.rows ≤ p.rows * p.m := Nat.le_mul_of_pos_right _ hm1
  have hmm : p.m ≤ p.rows * p.m := Nat.le_mul_of_pos_left _ hr1
  have hnn : p.n ≤ p.n * p.cols := Nat.le_mul_of_pos_right _ hc1
  have hcc : p.cols ≤ p.n * p.cols := Nat.le_mul_of_pos_left _ hn1
  have hNtle : Nt ≤ p.Nn Nt := Nat.le_mul_of_pos_left _ (Nat.mul_pos hn1 hc1)
  have hKtle : Kt ≤ p.Kn Kt := Nat.le_mul_of_pos_left _ hk1
  have hMi' : p.Mn Mt < 2 ^ 31 := by exact_mod_cast hMi
  have hKi' : p.Kn Kt < 2 ^ 31 := by exact_mod_cast hKi
  have hNi' : p.Nn Nt < 2 ^ 31 := by exact_mod_cast hNi
  have hP1 : ((p.n * p.cols : Nat) : Int) * (p.Mn Mt : Nat) ≤ 2 ^ 62 := hprod hsn hMi h0M
  have hP2 : ((p.n * p.cols : Nat) : Int) * (p.Kn Kt : Nat) ≤ 2 ^ 62 := hprod hsn hKi h0K
  have hP3 : (p.k : Int) * (p.Nn Nt : Nat) ≤ 2 ^ 62 := hprod hsk hNi h0N
  have big (x : Nat) (hx : x < 2 ^ 31) : (x : Int) ≤ 2 ^ 62 := by
    have : (x : Int) < 2 ^ 31 := by exact_mod_cast hx
    linarith
  have hQ1 : (p.cols : Int) * p.n * (p.Mn Mt : Nat) ≤ 2 ^ 62 := by
    have := hP1; push_cast at this; linarith
  have hQ2 : (p.cols : Int) * p.n * (p.Kn Kt : Nat) ≤ 2 ^ 62 := by
    have := hP2; push_cast at this; linarith
  have hQ3 : (p.k : Int) * (p.Nn Nt : Nat) ≤ 2 ^ 62 := hP3
  have hQ4 : (p.n : Int) * p.cols ≤ 2 ^ 62 := by push_cast at hsn; linarith
  have hQ5 : (p.rows : Int) * p.m ≤ 2 ^ 62 := by push_cast at hsm; linarith
  have hQ6 : (1 : Int) ≤ (p.rows : Int) * p.m := by exact_mod_cast hrm0
  have hl1 : (1 : Int) ≤ Nt := by exact_mod_cast hNt
  have hl2 : (1 : Int) ≤ Kt := by exact_mod_cast hKt
  have hl3 : (1 : Int) ≤ p.n := by exact_mod_cast hn1
  have hl4 : (1 : Int) ≤ p.m := by exact_mod_cast hm1
  have hl5 : (1 : Int) ≤ p.k := by exact_mod_cast hk1
  have hl6 : (1 : Int) ≤ p.rows := by exact_mod_cast hr1
  have hu1 : (Nt : Int) ≤ (p.Nn Nt : Nat) := by exact_mod_cast hNtle
  have hu2 : (Kt : Int) ≤ (p.Kn Kt : Nat) := by exact_mod_cast hKtle
  have hu3 : (p.rows : Int) ≤ p.rows * p.m := by exact_mod_cast hrow
  have hu4 : (p.m : Int) ≤ p.rows * p.m := by exact_mod_cast hmm
  have hu5 : (p.n : Int) ≤ p.n * p.cols := by exact_mod_cast hnn
  have hu6 : (p.k : Int) < 2 ^ 31 := hsk
  intro rb hrb c hc
  refine ⟨fits_of_bounds _ (tapC_nonneg hp Mt Nt rb c) ?_ ?_
      (by linarith [tapC_le hp hrb hc hNt]), fun hcn => ?_,
      fits_of_bounds _ (tapB_nonneg hp Kt Nt c) ?_ ?_
      (by linarith [tapB_le (Kt := Kt) (Nt := Nt) hp hc])⟩
  rotate_left 2
  · have hs1 : 1 ≤ p.shimRows c := by
      unfold Params.shimRows
      have : 1 ≤ p.rows / p.nA :=
        (Nat.one_le_div_iff (nA_pos hp)).mpr (min_le_left _ _)
      omega
    have hs2 : p.shimRows c ≤ p.rows := by have := shimEnd_le hcn; omega
    have hs3 : (1 : Int) ≤ (p.shimRows c : Int) * p.m := by
      have : 1 ≤ p.shimRows c * p.m := Nat.one_le_iff_ne_zero.mpr (by positivity)
      exact_mod_cast this
    have hs4 : (p.shimRows c : Int) * p.m ≤ p.rows * p.m := by
      exact_mod_cast Nat.mul_le_mul_right _ hs2
    refine fits_of_bounds _ (tapA_nonneg hp Kt Nt rb c) ?_ ?_
      (by linarith [tapA_le (Kt := Kt) (Nt := Nt) hp hrb hcn])
    all_goals
      unfold tapA
      simp only [nTiles_Nn hp, Kn_div hp, List.mem_cons, List.mem_nil_iff, or_false,
        forall_eq_or_imp, forall_eq]
      repeat' apply And.intro
      all_goals first | (norm_num; done) | (push_cast; linarith)
  all_goals
    first
    | unfold tapC
    | unfold tapB
    split_ifs
    all_goals
      simp only [nTiles_Nn hp, Kn_div hp, List.mem_cons, List.mem_nil_iff, or_false,
        forall_eq_or_imp, forall_eq]
      repeat' apply And.intro
      all_goals first | (norm_num; done) | (push_cast; linarith)

end IronVerify.Schema.WholeArray.Ovf
