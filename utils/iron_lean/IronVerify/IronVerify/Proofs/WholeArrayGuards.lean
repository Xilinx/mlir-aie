-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Proofs.WholeArrayC1
import IronVerify.Proofs.WholeArrayC2
import IronVerify.Proofs.WholeArrayC4
import IronVerify.Proofs.WholeArrayC8

/-!
The theorems in `WholeArrayC*.lean` quantify over `M = 256·Mt`, `K = 64·Kt`, `N = 32·cols·Nt`.
Here that hypothesis is derived from the exported runtime sequence itself: whenever its
`cf.assert`s pass (`(seq M K N).ok`, which includes the design's `require(...)` guards), `M`, `K`
and `N` have that form. The `_ok` theorems restate T-A, T-B and T-C over `M K N : Int` with that
single hypothesis.
-/

namespace IronVerify.Proofs.WholeArrayGuards

open IronVerify Spec.WholeArray

theorem multiple_of_tmod {M d : Int} (hd : 0 < d) (hM : M > 0) (h : Int.tmod M d = 0) :
    ∃ t : Nat, 0 < t ∧ M = d * t := by
  rw [Int.tmod_eq_emod] at h
  simp [le_of_lt hM] at h
  obtain ⟨q, hq⟩ := h
  have hq0 : 0 < q := by
    rcases lt_or_ge 0 q with h | h
    · exact h
    · nlinarith
  exact ⟨q.toNat, by omega, by rw [Int.toNat_of_nonneg hq0.le]; exact hq⟩

/-- Every whole_array sequence opens with the design's six guards. If they pass, `M`, `K` and `N`
are positive multiples of their tile sizes. -/
theorem shape_of_ok {M K N dm dk dn : Int} {rest : Seq} (hdm : 0 < dm) (hdk : 0 < dk)
    (hdn : 0 < dn)
    (h : (Seq.seq (.check (decide (M > 0))) (.seq (.check (decide (K > 0)))
      (.seq (.check (decide (N > 0))) (.seq (.check (decide (Int.tmod M dm = 0)))
      (.seq (.check (decide (Int.tmod K dk = 0))) (.seq (.check (decide (Int.tmod N dn = 0)))
        rest)))))).ok = true) :
    ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧ M = dm * Mt ∧ K = dk * Kt ∧ N = dn * Nt := by
  simp only [Seq.ok, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨hM, hK, hN, hm, hk, hn, -⟩ := h
  obtain ⟨Mt, hMt, rfl⟩ := multiple_of_tmod hdm hM hm
  obtain ⟨Kt, hKt, rfl⟩ := multiple_of_tmod hdk hK hk
  obtain ⟨Nt, hNt, rfl⟩ := multiple_of_tmod hdn hN hn
  exact ⟨Mt, Kt, Nt, hMt, hKt, hNt, rfl, rfl, rfl⟩

namespace C1

open Designs.WholeArrayC1 Proofs.WholeArrayC1

theorem shape (M K N : Int) (h : (seq M K N).ok = true) :
    ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧
      M = ((256 * Mt : Nat) : Int) ∧ K = ((64 * Kt : Nat) : Int) ∧ N = ((32 * Nt : Nat) : Int) := by
  unfold seq at h
  obtain ⟨Mt, Kt, Nt, h1, h2, h3, hM, hK, hN⟩ :=
    shape_of_ok (by norm_num) (by norm_num) (by norm_num) h
  exact ⟨Mt, Kt, Nt, h1, h2, h3, by push_cast; exact hM, by push_cast; exact hK,
    by push_cast; exact hN⟩

theorem tA_ok (M K N : Int) (hok : (seq M K N).ok = true) (row : Nat) (hrow : row < 4) (f : FifoD)
    (hf : net.fifo? (aName row) = some f) (τ kk a b : Nat)
    (hτ : τ < (M / 256).toNat * (N / 32).toNat) (hkk : kk < (K / 64).toNat) (ha : a < 64)
    (hb : b < 64) :
    design.deliver (M, K, N) f row (τ * (K / 64).toNat + kk) (layoutA a b) =
      some (.host 0 ((i0 (N / 32).toNat row τ + a : Nat) * K + (kk * 64 + b : Nat))) := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e1 : (((256 * Mt : Nat) : Int) / 256).toNat = Mt := by omega
  have e2 : (((64 * Kt : Nat) : Int) / 64).toNat = Kt := by omega
  have e3 : (((32 * Nt : Nat) : Int) / 32).toNat = Nt := by omega
  rw [e1, e3] at hτ; rw [e2] at hkk; rw [e2, e3]
  exact tA Mt Kt Nt row hrow f hf τ kk a b hτ hkk ha hb

theorem tB_ok (M K N : Int) (hok : (seq M K N).ok = true) (w : Nat) (hw : w < 4) (f : FifoD)
    (hf : net.fifo? "B_L2L1_0" = some f) (τ kk a b : Nat)
    (hτ : τ < (M / 256).toNat * (N / 32).toNat) (hkk : kk < (K / 64).toNat) (ha : a < 64)
    (hb : b < 32) :
    design.deliver (M, K, N) f w (τ * (K / 64).toNat + kk) (layoutB a b) =
      some (.host 1 ((kk * 64 + a : Nat) * N + (j0 1 (N / 32).toNat 0 τ + b : Nat))) := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e1 : (((256 * Mt : Nat) : Int) / 256).toNat = Mt := by omega
  have e2 : (((64 * Kt : Nat) : Int) / 64).toNat = Kt := by omega
  have e3 : (((32 * Nt : Nat) : Int) / 32).toNat = Nt := by omega
  rw [e1, e3] at hτ; rw [e2] at hkk; rw [e2, e3]
  exact tB Mt Kt Nt w hw f hf τ kk a b hτ hkk ha hb

theorem tC_ok (M K N : Int) (hok : (seq M K N).ok = true) (f : FifoD)
    (hf : net.fifo? "C_L2L3_0" = some f) (i j : Nat) (hi : (i : Int) < M) (hj : (j : Int) < N) :
    (∃! p : Nat, (design.host (M, K, N) "C_L2L3_0")[p]? = some (2, i * N + j)) ∧
    design.stream (M, K, N) f (posC (N / 32).toNat i j) =
      some (.core ((i % 256) / 64) (cName ((i % 256) / 64)) (tauC (N / 32).toNat i j)
        (layoutC (i % 64) (j % 32))) ∧
    i = i0 (N / 32).toNat ((i % 256) / 64) (tauC (N / 32).toNat i j) + i % 64 ∧
    j = j0 1 (N / 32).toNat 0 (tauC (N / 32).toNat i j) + j % 32 := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e3 : (((32 * Nt : Nat) : Int) / 32).toNat = Nt := by omega
  have hc : ((i * (32 * Nt) + j : Nat) : Int) = (i : Int) * ((32 * Nt : Nat) : Int) + j := by
    push_cast; ring
  rw [e3, ← hc]
  exact tC Mt Kt Nt f hf i j (by omega) (by omega)

end C1

namespace C2

open Designs.WholeArrayC2 Proofs.WholeArrayC2

theorem shape (M K N : Int) (h : (seq M K N).ok = true) :
    ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧
      M = ((256 * Mt : Nat) : Int) ∧ K = ((64 * Kt : Nat) : Int) ∧ N = ((64 * Nt : Nat) : Int) := by
  unfold seq at h
  obtain ⟨Mt, Kt, Nt, h1, h2, h3, hM, hK, hN⟩ :=
    shape_of_ok (by norm_num) (by norm_num) (by norm_num) h
  exact ⟨Mt, Kt, Nt, h1, h2, h3, by push_cast; exact hM, by push_cast; exact hK,
    by push_cast; exact hN⟩

theorem tA_ok (M K N : Int) (hok : (seq M K N).ok = true) (row col : Nat) (hrow : row < 4)
    (hcol : col < 2) (f : FifoD) (hf : net.fifo? (aL2 row) = some f) (τ kk a b : Nat)
    (hτ : τ < (M / 256).toNat * (N / 64).toNat) (hkk : kk < (K / 64).toNat) (ha : a < 64)
    (hb : b < 64) :
    design.deliver (M, K, N) f (row * 2 + col) (τ * (K / 64).toNat + kk) (layoutA a b) =
      some (.host 0 ((i0 (N / 64).toNat row τ + a : Nat) * K + (kk * 64 + b : Nat))) := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e1 : (((256 * Mt : Nat) : Int) / 256).toNat = Mt := by omega
  have e2 : (((64 * Kt : Nat) : Int) / 64).toNat = Kt := by omega
  have e3 : (((64 * Nt : Nat) : Int) / 64).toNat = Nt := by omega
  rw [e1, e3] at hτ; rw [e2] at hkk; rw [e2, e3]
  exact tA Mt Kt Nt row col hrow hcol f hf τ kk a b hτ hkk ha hb

theorem tB_ok (M K N : Int) (hok : (seq M K N).ok = true) (row col : Nat) (hrow : row < 4)
    (hcol : col < 2) (f : FifoD) (hf : net.fifo? (bL2 col) = some f) (τ kk a b : Nat)
    (hτ : τ < (M / 256).toNat * (N / 64).toNat) (hkk : kk < (K / 64).toNat) (ha : a < 64)
    (hb : b < 32) :
    design.deliver (M, K, N) f (row * 2 + col) (τ * (K / 64).toNat + kk) (layoutB a b) =
      some (.host 1 ((kk * 64 + a : Nat) * N + (j0 2 (N / 64).toNat col τ + b : Nat))) := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e1 : (((256 * Mt : Nat) : Int) / 256).toNat = Mt := by omega
  have e2 : (((64 * Kt : Nat) : Int) / 64).toNat = Kt := by omega
  have e3 : (((64 * Nt : Nat) : Int) / 64).toNat = Nt := by omega
  rw [e1, e3] at hτ; rw [e2] at hkk; rw [e2, e3]
  exact tB Mt Kt Nt row col hrow hcol f hf τ kk a b hτ hkk ha hb

theorem tC_ok (M K N : Int) (hok : (seq M K N).ok = true) (i j : Nat) (hi : (i : Int) < M)
    (hj : (j : Int) < N) (f : FifoD) (hf : net.fifo? (cL3 (colC j)) = some f) :
    (∃! cp : Nat × Nat, cp.1 < 2 ∧
      (design.host (M, K, N) (cL3 cp.1))[cp.2]? = some (2, i * N + j)) ∧
    design.stream (M, K, N) f (posC (N / 64).toNat i j) =
      some (.core ((i % 256) / 64 * 2 + colC j) (cL1 (colC j) ((i % 256) / 64))
        (tauC (N / 64).toNat i j) (layoutC (i % 64) (j % 32))) ∧
    i = i0 (N / 64).toNat ((i % 256) / 64) (tauC (N / 64).toNat i j) + i % 64 ∧
    j = j0 2 (N / 64).toNat (colC j) (tauC (N / 64).toNat i j) + j % 32 := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e3 : (((64 * Nt : Nat) : Int) / 64).toNat = Nt := by omega
  have hc : ((i * (64 * Nt) + j : Nat) : Int) = (i : Int) * ((64 * Nt : Nat) : Int) + j := by
    push_cast; ring
  rw [e3, ← hc]
  exact tC Mt Kt Nt i j (by omega) (by omega) f hf

end C2

namespace C4

open Designs.WholeArrayC4 Proofs.WholeArrayC4

theorem shape (M K N : Int) (h : (seq M K N).ok = true) :
    ∃ Mt Kt Nq : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nq ∧
      M = ((256 * Mt : Nat) : Int) ∧ K = ((64 * Kt : Nat) : Int) ∧ N = ((128 * Nq : Nat) : Int) := by
  unfold seq at h
  obtain ⟨Mt, Kt, Nq, h1, h2, h3, hM, hK, hN⟩ :=
    shape_of_ok (by norm_num) (by norm_num) (by norm_num) h
  exact ⟨Mt, Kt, Nq, h1, h2, h3, by push_cast; exact hM, by push_cast; exact hK,
    by push_cast; exact hN⟩

theorem tA_ok (M K N : Int) (hok : (seq M K N).ok = true) (row col : Nat) (hrow : row < 4)
    (hcol : col < 4) (f : FifoD) (hf : net.fifo? (aL2 row) = some f) (τ kk a b : Nat)
    (hτ : τ < (M / 256).toNat * (N / 128).toNat) (hkk : kk < (K / 64).toNat) (ha : a < 64)
    (hb : b < 64) :
    design.deliver (M, K, N) f (row * 4 + col) (τ * (K / 64).toNat + kk) (layoutA a b) =
      some (.host 0 ((i0 (N / 128).toNat row τ + a : Nat) * K + (kk * 64 + b : Nat))) := by
  obtain ⟨Mt, Kt, Nq, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e1 : (((256 * Mt : Nat) : Int) / 256).toNat = Mt := by omega
  have e2 : (((64 * Kt : Nat) : Int) / 64).toNat = Kt := by omega
  have e3 : (((128 * Nq : Nat) : Int) / 128).toNat = Nq := by omega
  rw [e1, e3] at hτ; rw [e2] at hkk; rw [e2, e3]
  exact tA Mt Kt Nq row col hrow hcol f hf τ kk a b hτ hkk ha hb

theorem tB_ok (M K N : Int) (hok : (seq M K N).ok = true) (row col : Nat) (hrow : row < 4)
    (hcol : col < 4) (f : FifoD) (hf : net.fifo? (bL2 col) = some f) (τ kk a b : Nat)
    (hτ : τ < (M / 256).toNat * (N / 128).toNat) (hkk : kk < (K / 64).toNat) (ha : a < 64)
    (hb : b < 32) :
    design.deliver (M, K, N) f (row * 4 + col) (τ * (K / 64).toNat + kk) (layoutB a b) =
      some (.host 1 ((kk * 64 + a : Nat) * N + (j0 4 (N / 128).toNat col τ + b : Nat))) := by
  obtain ⟨Mt, Kt, Nq, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e1 : (((256 * Mt : Nat) : Int) / 256).toNat = Mt := by omega
  have e2 : (((64 * Kt : Nat) : Int) / 64).toNat = Kt := by omega
  have e3 : (((128 * Nq : Nat) : Int) / 128).toNat = Nq := by omega
  rw [e1, e3] at hτ; rw [e2] at hkk; rw [e2, e3]
  exact tB Mt Kt Nq row col hrow hcol f hf τ kk a b hτ hkk ha hb

theorem tC_ok (M K N : Int) (hok : (seq M K N).ok = true) (i j : Nat) (hi : (i : Int) < M)
    (hj : (j : Int) < N) (f : FifoD) (hf : net.fifo? (cL3 (colC j)) = some f) :
    (∃! cp : Nat × Nat, cp.1 < 4 ∧
      (design.host (M, K, N) (cL3 cp.1))[cp.2]? = some (2, i * N + j)) ∧
    design.stream (M, K, N) f (posC (N / 128).toNat i j) =
      some (.core ((i % 256) / 64 * 4 + colC j) (cL1 (colC j) ((i % 256) / 64))
        (tauC (N / 128).toNat i j) (layoutC (i % 64) (j % 32))) ∧
    i = i0 (N / 128).toNat ((i % 256) / 64) (tauC (N / 128).toNat i j) + i % 64 ∧
    j = j0 4 (N / 128).toNat (colC j) (tauC (N / 128).toNat i j) + j % 32 := by
  obtain ⟨Mt, Kt, Nq, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e3 : (((128 * Nq : Nat) : Int) / 128).toNat = Nq := by omega
  have hc : ((i * (128 * Nq) + j : Nat) : Int) = (i : Int) * ((128 * Nq : Nat) : Int) + j := by
    push_cast; ring
  rw [e3, ← hc]
  exact tC Mt Kt Nq i j (by omega) (by omega) f hf

end C4

namespace C8

open Designs.WholeArrayC8 Proofs.WholeArrayC8

theorem shape (M K N : Int) (h : (seq M K N).ok = true) :
    ∃ Mt Kt Nt : Nat, 0 < Mt ∧ 0 < Kt ∧ 0 < Nt ∧
      M = ((256 * Mt : Nat) : Int) ∧ K = ((64 * Kt : Nat) : Int) ∧ N = ((256 * Nt : Nat) : Int) := by
  unfold seq at h
  obtain ⟨Mt, Kt, Nt, h1, h2, h3, hM, hK, hN⟩ :=
    shape_of_ok (by norm_num) (by norm_num) (by norm_num) h
  exact ⟨Mt, Kt, Nt, h1, h2, h3, by push_cast; exact hM, by push_cast; exact hK,
    by push_cast; exact hN⟩

theorem tA_ok (M K N : Int) (hok : (seq M K N).ok = true) (row col : Nat) (hrow : row < 4)
    (hcol : col < 8) (f : FifoD) (hf : net.fifo? (aL2 row) = some f) (τ kk a b : Nat)
    (hτ : τ < (M / 256).toNat * (N / 256).toNat) (hkk : kk < (K / 64).toNat) (ha : a < 64)
    (hb : b < 64) :
    design.deliver (M, K, N) f (row * 8 + col) (τ * (K / 64).toNat + kk) (layoutA a b) =
      some (.host 0 ((i0 (N / 256).toNat row τ + a : Nat) * K + (kk * 64 + b : Nat))) := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e1 : (((256 * Mt : Nat) : Int) / 256).toNat = Mt := by omega
  have e2 : (((64 * Kt : Nat) : Int) / 64).toNat = Kt := by omega
  have e3 : (((256 * Nt : Nat) : Int) / 256).toNat = Nt := by omega
  rw [e1, e3] at hτ; rw [e2] at hkk; rw [e2, e3]
  exact tA Mt Kt Nt row col hrow hcol f hf τ kk a b hτ hkk ha hb

theorem tB_ok (M K N : Int) (hok : (seq M K N).ok = true) (row col : Nat) (hrow : row < 4)
    (hcol : col < 8) (f : FifoD) (hf : net.fifo? (bL2 col) = some f) (τ kk a b : Nat)
    (hτ : τ < (M / 256).toNat * (N / 256).toNat) (hkk : kk < (K / 64).toNat) (ha : a < 64)
    (hb : b < 32) :
    design.deliver (M, K, N) f (row * 8 + col) (τ * (K / 64).toNat + kk) (layoutB a b) =
      some (.host 1 ((kk * 64 + a : Nat) * N + (j0 8 (N / 256).toNat col τ + b : Nat))) := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e1 : (((256 * Mt : Nat) : Int) / 256).toNat = Mt := by omega
  have e2 : (((64 * Kt : Nat) : Int) / 64).toNat = Kt := by omega
  have e3 : (((256 * Nt : Nat) : Int) / 256).toNat = Nt := by omega
  rw [e1, e3] at hτ; rw [e2] at hkk; rw [e2, e3]
  exact tB Mt Kt Nt row col hrow hcol f hf τ kk a b hτ hkk ha hb

theorem tC_ok (M K N : Int) (hok : (seq M K N).ok = true) (i j : Nat) (hi : (i : Int) < M)
    (hj : (j : Int) < N) (f : FifoD) (hf : net.fifo? (cL3 (colC j)) = some f) :
    (∃! cp : Nat × Nat, cp.1 < 8 ∧
      (design.host (M, K, N) (cL3 cp.1))[cp.2]? = some (2, i * N + j)) ∧
    design.stream (M, K, N) f (posC (N / 256).toNat i j) =
      some (.core ((i % 256) / 64 * 8 + colC j) (cL1 (colC j) ((i % 256) / 64))
        (tauC (N / 256).toNat i j) (layoutC (i % 64) (j % 32))) ∧
    i = i0 (N / 256).toNat ((i % 256) / 64) (tauC (N / 256).toNat i j) + i % 64 ∧
    j = j0 8 (N / 256).toNat (colC j) (tauC (N / 256).toNat i j) + j % 32 := by
  obtain ⟨Mt, Kt, Nt, -, -, -, rfl, rfl, rfl⟩ := shape M K N hok
  have e3 : (((256 * Nt : Nat) : Int) / 256).toNat = Nt := by omega
  have hc : ((i * (256 * Nt) + j : Nat) : Int) = (i : Int) * ((256 * Nt : Nat) : Int) + j := by
    push_cast; ring
  rw [e3, ← hc]
  exact tC Mt Kt Nt i j (by omega) (by omega) f hf

end C8

end IronVerify.Proofs.WholeArrayGuards
