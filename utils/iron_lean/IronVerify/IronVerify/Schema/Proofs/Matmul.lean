-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Schema.Defs
import IronVerify.Schema.Proofs.Arith
import IronVerify.Schema.Proofs.PathA
import IronVerify.Schema.Proofs.PathB
import IronVerify.Schema.Proofs.PathC

/-!
C = A @ B for the schema. T-C names the core, tile and slot behind each drain write; T-A and
T-B say that core's `kk`-th objects are exactly the spec's operands, so the drained value is the
kernel folded over the spec's operands (`orderedAcc`, any kernel). For a kernel whose output at
`(a, b)` depends only on `C a b` the fold collapses to a scalar recurrence: a sum for the exact
kernel (`tMM`, and `tMM_wrap` in `ZMod (2 ^ w)`), a clamped sum that never clamps for the
saturating one (`tMM_sat`).
-/

namespace IronVerify.Schema.WholeArray.Matmul

open IronVerify IronVerify.Schema.WholeArray

/-- A kernel that is pointwise in its C tile at `(a, b)` folds to a scalar recurrence there. -/
theorem foldl_pointwise {R : Type*} (call : KernelCall R) (Af Bf : Nat → Nat → R)
    (step : Nat → R → R) (Z : Nat → Nat → R) (a b : Nat)
    (h : ∀ kk C, call (Af kk) (Bf kk) C a b = step kk (C a b)) :
    ∀ K, ((List.range K).foldl (fun C kk => call (Af kk) (Bf kk) C) Z) a b =
      (List.range K).foldl (fun x kk => step kk x) (Z a b)
  | 0 => rfl
  | K + 1 => by
    rw [List.range_succ, List.foldl_append, List.foldl_append, List.foldl_cons, List.foldl_nil,
      List.foldl_cons, List.foldl_nil, h, foldl_pointwise call Af Bf step Z a b h K]

theorem foldl_add {R : Type*} [AddCommMonoid R] (f : Nat → R) (z : R) :
    ∀ K, (List.range K).foldl (fun x kk => x + f kk) z = z + ∑ kk ∈ Finset.range K, f kk
  | 0 => by simp
  | K + 1 => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, foldl_add f z K,
      Finset.sum_range_succ, add_assoc]

variable (p : Params)

variable (hp : p.Valid)
include hp

/-- `w = worker row col` reads A on its row's fifo and B on its column's. -/
private theorem worker_div {row col : Nat} (hcol : col < p.cols) :
    p.worker row col / p.cols = row ∧ p.worker row col % p.cols = col := by
  unfold Params.worker
  have hc := hp.cols_pos
  refine ⟨?_, ?_⟩
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hc, Nat.div_eq_of_lt hcol, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hcol]

/-- The A operand a core reads for its `kk`-th call on tile `τ` is the spec's. -/
private theorem opA_eq {R : Type*} [Zero R] (hA hB : Nat → R) (Mt Kt Nt row col τ kk i0 : Nat)
    (hrow : row < p.rows) (hcol : col < p.cols) (hτ : τ < Mt * Nt) (hkk : kk < Kt)
    (hi0 : p.i0 Nt row τ = i0) :
    opA p hA hB Mt Kt Nt (p.worker row col) (τ * Kt + kk) = specA p hA Kt i0 kk := by
  funext s
  unfold opA specA
  rw [(worker_div p hp hcol).1]
  by_cases hs : s < p.m * p.k
  · simp only [hs, ↓reduceIte]
    have h := PathA.tA p hp Mt Kt Nt row col τ kk _ _ hrow hcol hτ hkk
      (hp.unlayoutA_fst_lt hs) (hp.unlayoutA_snd_lt hs)
    rw [hp.layoutA_unlayoutA hs] at h
    rw [h, hi0]
    simp [hostVal]
  · simp only [hs, ↓reduceIte]

/-- The B operand a core reads for its `kk`-th call on tile `τ` is the spec's. -/
private theorem opB_eq {R : Type*} [Zero R] (hA hB : Nat → R) (Mt Kt Nt row col τ kk j0 : Nat)
    (hrow : row < p.rows) (hcol : col < p.cols) (hτ : τ < Mt * Nt) (hkk : kk < Kt)
    (hj0 : p.j0 Nt col τ = j0) :
    opB p hA hB Mt Kt Nt (p.worker row col) (τ * Kt + kk) = specB p hB Kt Nt j0 kk := by
  funext s
  unfold opB specB
  rw [(worker_div p hp hcol).2]
  by_cases hs : s < p.k * p.n
  · simp only [hs, ↓reduceIte]
    have h := PathB.tB p hp Mt Kt Nt row col τ kk _ _ hrow hcol hτ hkk
      (hp.unlayoutB_fst_lt hs) (hp.unlayoutB_snd_lt hs)
    rw [hp.layoutB_unlayoutB hs] at h
    rw [h, hj0]
    simp [hostVal]
  · simp only [hs, ↓reduceIte]

/-- Ordered accumulation, for any kernel (any dtype, any rounding): the value written for
`C[i, j]` is the kernel folded over `kk = 0 … Kt - 1` on exactly the A row block and B column
block the spec names, in order. -/
theorem orderedAcc {R : Type*} [Zero R] (hA hB : Nat → R) (call : KernelCall R)
    (zero : Nat → Nat → R) (Mt Kt Nt i j c q : Nat) (hi : i < p.Mn Mt) (hj : j < p.Nn Nt)
    (hc : c < p.cols)
    (hq : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[q]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int))) :
    drainVal p hA hB Mt Kt Nt call zero c q =
      (List.range Kt).foldl
        (fun C kk => call (specA p hA Kt (i - i % p.m) kk) (specB p hB Kt Nt (j - j % p.n) kk) C)
        zero (i % p.m) (j % p.n) := by
  obtain ⟨-, hsrc, hi_eq, hj_eq⟩ := PathC.tC p hp Mt Kt Nt i j hi hj
  obtain ⟨hcj, hstream⟩ := hsrc c q hc hq
  subst hcj
  have hτ : p.tauC Nt i j < Mt * Nt := hp.tauC_lt hi hj
  unfold drainVal
  rw [hstream]
  dsimp only
  rw [hp.unlayoutC_layoutC (Nat.mod_lt _ hp.m_pos) (Nat.mod_lt _ hp.n_pos)]
  unfold coreTile
  dsimp only
  rw [List.foldl_ext _ _ _ fun C kk hkk => by
    rw [opA_eq p hp hA hB Mt Kt Nt _ _ _ kk (i - i % p.m) (hp.rowC_lt i) (hp.colC_lt j) hτ
        (List.mem_range.1 hkk) (by omega),
      opB_eq p hp hA hB Mt Kt Nt _ _ _ kk (j - j % p.n) (hp.rowC_lt i) (hp.colC_lt j) hτ
        (List.mem_range.1 hkk) (by omega)]]

/-- The spec's `kk`-th call contributes the `kk`-th `k`-block of `C[i, j]`'s dot product. -/
private theorem spec_inner {R : Type*} [CommSemiring R] (hA hB : Nat → R) (Kt Nt i j kk : Nat) :
    ∑ c ∈ Finset.range p.k,
        specA p hA Kt (i - i % p.m) kk (p.layoutA (i % p.m) c) *
          specB p hB Kt Nt (j - j % p.n) kk (p.layoutB c (j % p.n)) =
      ∑ c ∈ Finset.range p.k,
        hA (p.aIdx (p.Kn Kt) i (kk * p.k + c)) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) (kk * p.k + c) j) := by
  refine Finset.sum_congr rfl fun c hc => ?_
  have hc := Finset.mem_range.1 hc
  have ha := Nat.mod_lt i hp.m_pos
  have hb := Nat.mod_lt j hp.n_pos
  unfold specA specB
  simp only [hp.layoutA_lt ha hc, hp.layoutB_lt hc hb, ↓reduceIte]
  rw [hp.unlayoutA_layoutA hc,
    hp.unlayoutB_layoutB hc hb, Nat.sub_add_cancel (Nat.mod_le i p.m),
    Nat.sub_add_cancel (Nat.mod_le j p.n)]

/-- Exact `C = A @ B` over any commutative semiring (ℤ, or `Wrap w` for `w`-bit integers with
saturation off). -/
theorem tMM {R : Type*} [CommSemiring R] (hA hB : Nat → R) (Mt Kt Nt i j c q : Nat)
    (hi : i < p.Mn Mt) (hj : j < p.Nn Nt) (hc : c < p.cols)
    (hq : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[q]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int))) :
    drainVal p hA hB Mt Kt Nt p.matmul (fun _ _ => 0) c q =
      ∑ x ∈ Finset.range (p.Kn Kt),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) := by
  rw [orderedAcc p hp hA hB p.matmul _ Mt Kt Nt i j c q hi hj hc hq,
    foldl_pointwise _ _ _ (fun kk x => x + ∑ c ∈ Finset.range p.k,
      specA p hA Kt (i - i % p.m) kk (p.layoutA (i % p.m) c) *
        specB p hB Kt Nt (j - j % p.n) kk (p.layoutB c (j % p.n))) _ _ _ (fun _ _ => rfl),
    foldl_add, zero_add]
  simp only [spec_inner p hp]
  rw [sum_blocks p.k (fun x => hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j))]
  rfl

/-- `w`-bit integers with saturation off (the contract assumption): the stored C is the exact
integer product reduced mod `2 ^ w`. -/
theorem tMM_wrap (w : Nat) (hA hB : Nat → ℤ) (Mt Kt Nt i j c q : Nat)
    (hi : i < p.Mn Mt) (hj : j < p.Nn Nt) (hc : c < p.cols)
    (hq : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[q]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int))) :
    drainVal p (fun x => (hA x : Wrap w)) (fun x => (hB x : Wrap w)) Mt Kt Nt p.matmul
        (fun _ _ => 0) c q =
      ((∑ x ∈ Finset.range (p.Kn Kt),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) : ℤ) : Wrap w) := by
  rw [tMM p hp _ _ Mt Kt Nt i j c q hi hj hc hq]
  push_cast
  rfl

-- `hlo`/`hup` are kept to match the statement in `Theorems.lean`; `hpart` at `kk = 0` implies them.
set_option linter.unusedVariables false in
/-- Saturating stores: exact over ℤ as long as no partial sum of `C[i, j]` (after any number of
k steps) leaves `[lo, hi]`. -/
theorem tMM_sat (lo up : ℤ) (hlo : lo ≤ 0) (hup : 0 ≤ up) (hA hB : Nat → ℤ)
    (Mt Kt Nt i j c q : Nat) (hi : i < p.Mn Mt) (hj : j < p.Nn Nt) (hc : c < p.cols)
    (hq : ((schema p).host (p.dims Mt Kt Nt) (cL3 c))[q]? =
      some (2, ((p.cIdx (p.Mn Mt) (p.Nn Nt) i j : Nat) : Int)))
    (hpart : ∀ kk ≤ Kt, lo ≤ ∑ x ∈ Finset.range (kk * p.k),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) ∧
      ∑ x ∈ Finset.range (kk * p.k),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) ≤ up) :
    drainVal p hA hB Mt Kt Nt (p.matmulSat lo up) (fun _ _ => 0) c q =
      ∑ x ∈ Finset.range (p.Kn Kt),
        hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j) := by
  set f : Nat → ℤ := fun x => hA (p.aIdx (p.Kn Kt) i x) * hB (p.bIdx (p.Kn Kt) (p.Nn Nt) x j)
  set inner : Nat → ℤ := fun kk => ∑ c ∈ Finset.range p.k,
    specA p hA Kt (i - i % p.m) kk (p.layoutA (i % p.m) c) *
      specB p hB Kt Nt (j - j % p.n) kk (p.layoutB c (j % p.n)) with hinner
  have hblk : ∀ K, ∑ kk ∈ Finset.range K, inner kk = ∑ x ∈ Finset.range (K * p.k), f x := by
    intro K
    simp only [hinner, spec_inner p hp]
    rw [sum_blocks p.k f K, Nat.mul_comm]
  -- Below the bound the clamp never fires, so the fold is the partial sum.
  have hfold : ∀ K ≤ Kt, (List.range K).foldl (fun x kk => max lo (min up (x + inner kk))) 0 =
      ∑ x ∈ Finset.range (K * p.k), f x := by
    intro K
    induction K with
    | zero => intro _; simp
    | succ K ih =>
      intro hK
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, ih (by omega),
        ← hblk, ← Finset.sum_range_succ, hblk]
      obtain ⟨h1, h2⟩ := hpart (K + 1) hK
      rw [min_eq_right h2, max_eq_right h1]
  rw [orderedAcc p hp hA hB (p.matmulSat lo up) _ Mt Kt Nt i j c q hi hj hc hq,
    foldl_pointwise _ _ _ (fun kk x => max lo (min up (x + inner kk))) _ _ _ (fun _ _ => rfl),
    hfold Kt le_rfl]
  show _ = ∑ x ∈ Finset.range (p.k * Kt), f x
  rw [Nat.mul_comm]

end IronVerify.Schema.WholeArray.Matmul
