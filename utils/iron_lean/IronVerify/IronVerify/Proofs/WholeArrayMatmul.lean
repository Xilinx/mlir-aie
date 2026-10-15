-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Spec.Contract
import IronVerify.Proofs.WholeArrayGuards

/-!
`C = A @ B`: given the kernel and core contracts of `Spec/Contract.lean`, whenever the exported
sequence's `cf.assert`s pass, every element of C is written exactly once and the value written to
`C[i, j]` is `Σ_k A[i, k] · B[k, j]`. A, B and C are the row-major host buffers 0, 1 and 2.
-/

namespace IronVerify.Proofs.WholeArrayMatmul

open IronVerify Spec.WholeArray Spec.Contract

variable {R : Type*} [CommSemiring R]

theorem foldl_matmul (A B : Nat → Nat → R) (base : Nat) (C0 : Nat → Nat → R) (a b : Nat) :
    ∀ n, (List.range n).foldl (fun C kk => matmul (A (base + kk)) (B (base + kk)) C) C0 a b =
      C0 a b + ∑ kk ∈ Finset.range n, ∑ c ∈ Finset.range 64,
        A (base + kk) (layoutA a c) * B (base + kk) (layoutB c b)
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, matmul,
      foldl_matmul A B base C0 a b n,
      Finset.sum_range_succ (fun kk => ∑ c ∈ Finset.range 64,
        A (base + kk) (layoutA a c) * B (base + kk) (layoutB c b)) n, add_assoc]

/-- `Kt` blocks of 64 consecutive indices are the first `64·Kt` indices. -/
theorem sum_blocks (f : Nat → R) :
    ∀ Kt, ∑ kk ∈ Finset.range Kt, ∑ c ∈ Finset.range 64, f (kk * 64 + c) =
      ∑ k ∈ Finset.range (64 * Kt), f k
  | 0 => by simp
  | Kt + 1 => by
    rw [Finset.sum_range_succ, sum_blocks f Kt, show 64 * (Kt + 1) = 64 * Kt + 64 by ring,
      Finset.sum_range_add]
    congr 1
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Nat.mul_comm]

/-- A core's output tile, when its A objects carry row `row` of A and its B objects column `col` of
B, `kk·64 ..` at k-step `kk`, is their dot product. -/
theorem coreTile_eq (hA hB : Nat → R) (A B : Nat → Nat → R) (Kt τ a b row col K N : Nat)
    (hAv : ∀ kk < Kt, ∀ c < 64, A (τ * Kt + kk) (layoutA a c) = hA (row * K + (kk * 64 + c)))
    (hBv : ∀ kk < Kt, ∀ c < 64, B (τ * Kt + kk) (layoutB c b) = hB ((kk * 64 + c) * N + col)) :
    coreTile A B Kt τ a b = ∑ k ∈ Finset.range (64 * Kt), hA (row * K + k) * hB (k * N + col) := by
  rw [coreTile, foldl_matmul, zero_add,
    ← sum_blocks (fun k => hA (row * K + k) * hB (k * N + col))]
  refine Finset.sum_congr rfl fun kk hkk => Finset.sum_congr rfl fun c hc => ?_
  rw [Finset.mem_range] at hkk hc
  rw [hAv kk hkk c hc, hBv kk hkk c hc]

theorem toNat_cast_mul_add (x y z : Nat) : ((x : Int) * (y : Int) + (z : Int)).toNat = x * y + z := by
  rw [← Nat.cast_mul, ← Nat.cast_add, Int.toNat_natCast]

namespace C1

open Designs.WholeArrayC1 Proofs.WholeArrayC1

/-- Slot `s` of worker `w`'s `j`-th A object, as a value. -/
def opA (hA hB : Nat → R) (P : Int × Int × Int) (w j s : Nat) : R :=
  match net.fifo? (aName w) with
  | some f => hostVal hA hB (design.deliver P f w j s)
  | none => 0

/-- Slot `s` of worker `w`'s `j`-th B object, as a value. -/
def opB (hA hB : Nat → R) (P : Int × Int × Int) (w j s : Nat) : R :=
  match net.fifo? "B_L2L1_0" with
  | some f => hostVal hA hB (design.deliver P f w j s)
  | none => 0

/-- The value the C drain carries at position `p`. Each core runs `K / 64` k-steps per tile. -/
def drainVal (hA hB : Nat → R) (P : Int × Int × Int) (p : Nat) : R :=
  match net.fifo? "C_L2L3_0" with
  | some f => coreVal (opA hA hB P) (opB hA hB P) (P.2.1 / 64).toNat (design.stream P f p)
  | none => 0

/-- T-MM, one column: under the contracts, for every M, K, N the sequence's asserts accept,
exactly one drain position writes `C[i, j]`, and it writes `Σ_k A[i, k] · B[k, j]`. -/
theorem tMM (hA hB : Nat → R) (M K N : Int) (hok : (seq M K N).ok = true) (i j : Nat)
    (hi : (i : Int) < M) (hj : (j : Int) < N) :
    (∃! p : Nat, (design.host (M, K, N) "C_L2L3_0")[p]? = some (2, i * N + j)) ∧
    ∀ p : Nat, (design.host (M, K, N) "C_L2L3_0")[p]? = some (2, i * N + j) →
      drainVal hA hB (M, K, N) p =
        ∑ k ∈ Finset.range K.toNat, hA (i * K.toNat + k) * hB (k * N.toNat + j) := by
  obtain ⟨Mt, Kt, Nt, hMt, hKt, hNt, rfl, rfl, rfl⟩ := WholeArrayGuards.C1.shape M K N hok
  have hi' : i < 256 * Mt := by omega
  have hj' : j < 32 * Nt := by omega
  have hcast : ((i * (32 * Nt) + j : Nat) : Int) = (i : Int) * ((32 * Nt : Nat) : Int) + j := by
    push_cast; ring
  obtain ⟨fC, hfC⟩ : ∃ f, net.fifo? "C_L2L3_0" = some f := ⟨_, rfl⟩
  obtain ⟨huniq, hsrc, hrow, hcol⟩ := tC Mt Kt Nt fC hfC i j hi' hj'
  rw [← hcast]
  refine ⟨huniq, fun p hp => ?_⟩
  have hpos : (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) "C_L2L3_0")[
      posC Nt i j]? = some (2, ((i * (32 * Nt) + j : Nat) : Int)) := by
    simp only [Design.host, design]; push_cast
    rw [List.getElem?_toArray]; exact_mod_cast hostC_posC Mt Kt Nt i j hi' hj'
  rw [huniq.unique hp hpos, drainVal, hfC]
  simp only [Int.toNat_natCast]
  rw [show (((64 * Kt : Nat) : Int) / 64).toNat = Kt by omega, hsrc, coreVal,
    unlayoutC_layoutC _ _ (by omega)]
  set r := (i % 256) / 64 with hr_def
  set τ := tauC Nt i j with hτ_def
  have hr : r < 4 := by omega
  have hτ : τ < Mt * Nt := by
    have h1 : i / 256 < Mt := by omega
    have h2 : j / 32 < Nt := by omega
    simp only [hτ_def, tauC]; nlinarith
  obtain ⟨fA, hfA⟩ : ∃ f, net.fifo? (aName r) = some f := by
    interval_cases r <;> exact ⟨_, rfl⟩
  obtain ⟨fB, hfB⟩ : ∃ f, net.fifo? "B_L2L1_0" = some f := ⟨_, rfl⟩
  have key := coreTile_eq hA hB
    (opA hA hB ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) r)
    (opB hA hB ((256 * Mt : Nat), (64 * Kt : Nat), (32 * Nt : Nat)) r) Kt τ (i % 64) (j % 32)
    (i0 Nt r τ + i % 64) (j0 1 Nt 0 τ + j % 32) (64 * Kt) (32 * Nt)
    (fun kk hkk c hc => by
      simp only [opA, hfA]
      rw [tA Mt Kt Nt r hr fA hfA τ kk (i % 64) c hτ hkk (by omega) hc, hostVal,
        toNat_cast_mul_add])
    (fun kk hkk c hc => by
      simp only [opB, hfB]
      rw [tB Mt Kt Nt r hr fB hfB τ kk c (j % 32) hτ hkk hc (by omega), hostVal,
        toNat_cast_mul_add])
  rw [key, ← hrow, ← hcol]

end C1

namespace C2

open Designs.WholeArrayC2 Proofs.WholeArrayC2

/-- Slot `s` of worker `w = row·2 + col`'s `j`-th A object, as a value. -/
def opA (hA hB : Nat → R) (P : Int × Int × Int) (w j s : Nat) : R :=
  match net.fifo? (aL2 (w / 2)) with
  | some f => hostVal hA hB (design.deliver P f w j s)
  | none => 0

/-- Slot `s` of worker `w = row·2 + col`'s `j`-th B object, as a value. -/
def opB (hA hB : Nat → R) (P : Int × Int × Int) (w j s : Nat) : R :=
  match net.fifo? (bL2 (w % 2)) with
  | some f => hostVal hA hB (design.deliver P f w j s)
  | none => 0

/-- The value drain `c` carries at position `p`. Each core runs `K / 64` k-steps per tile. -/
def drainVal (hA hB : Nat → R) (P : Int × Int × Int) (c p : Nat) : R :=
  match net.fifo? (cL3 c) with
  | some f => coreVal (opA hA hB P) (opB hA hB P) (P.2.1 / 64).toNat (design.stream P f p)
  | none => 0

/-- T-MM, 2 columns: under the contracts, for every M, K, N the sequence's asserts accept,
exactly one position on one of the 2 drains writes `C[i, j]`, and it writes
`Σ_k A[i, k] · B[k, j]`. -/
theorem tMM (hA hB : Nat → R) (M K N : Int) (hok : (seq M K N).ok = true) (i j : Nat)
    (hi : (i : Int) < M) (hj : (j : Int) < N) :
    (∃! cp : Nat × Nat, cp.1 < 2 ∧
      (design.host (M, K, N) (cL3 cp.1))[cp.2]? = some (2, i * N + j)) ∧
    ∀ c p : Nat, c < 2 → (design.host (M, K, N) (cL3 c))[p]? = some (2, i * N + j) →
      drainVal hA hB (M, K, N) c p =
        ∑ k ∈ Finset.range K.toNat, hA (i * K.toNat + k) * hB (k * N.toNat + j) := by
  obtain ⟨Mt, Kt, Nt, hMt, hKt, hNt, rfl, rfl, rfl⟩ := WholeArrayGuards.C2.shape M K N hok
  have hi' : i < 256 * Mt := by omega
  have hj' : j < 64 * Nt := by omega
  have hcast : ((i * (64 * Nt) + j : Nat) : Int) = (i : Int) * ((64 * Nt : Nat) : Int) + j := by
    push_cast; ring
  set cc := colC j with hcc_def
  have hcc : cc < 2 := by simp only [hcc_def, colC]; omega
  obtain ⟨fC, hfC⟩ : ∃ f, net.fifo? (cL3 cc) = some f := by
    interval_cases cc <;> exact ⟨_, rfl⟩
  obtain ⟨huniq, hsrc, hrow, hcol⟩ := tC Mt Kt Nt i j hi' hj' fC hfC
  rw [← hcc_def] at hsrc hcol
  rw [← hcast]
  refine ⟨huniq, fun c p hc hp => ?_⟩
  have hpos : (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (64 * Nt : Nat)) (cL3 cc))[
      posC Nt i j]? = some (2, ((i * (64 * Nt) + j : Nat) : Int)) := by
    simp only [Design.host, design]; push_cast
    rw [List.getElem?_toArray]; exact_mod_cast hostC_posC Mt Kt Nt i j hi' hj'
  have hcp : (c, p) = (cc, posC Nt i j) := huniq.unique ⟨hc, hp⟩ ⟨hcc, hpos⟩
  simp only [Prod.mk.injEq] at hcp
  obtain ⟨rfl, rfl⟩ := hcp
  rw [drainVal, hfC]
  simp only [Int.toNat_natCast]
  rw [show (((64 * Kt : Nat) : Int) / 64).toNat = Kt by omega, hsrc, coreVal,
    unlayoutC_layoutC _ _ (by omega)]
  set r := (i % 256) / 64 with hr_def
  set τ := tauC Nt i j with hτ_def
  have hr : r < 4 := by omega
  have hτ : τ < Mt * Nt := by
    have h1 : i / 256 < Mt := by omega
    have h2 : j / 64 < Nt := by omega
    simp only [hτ_def, tauC]; nlinarith
  have hwdiv : (r * 2 + cc) / 2 = r := by omega
  have hwmod : (r * 2 + cc) % 2 = cc := by omega
  obtain ⟨fA, hfA⟩ : ∃ f, net.fifo? (aL2 r) = some f := by
    interval_cases r <;> exact ⟨_, rfl⟩
  obtain ⟨fB, hfB⟩ : ∃ f, net.fifo? (bL2 cc) = some f := by
    interval_cases cc <;> exact ⟨_, rfl⟩
  have key := coreTile_eq hA hB
    (opA hA hB ((256 * Mt : Nat), (64 * Kt : Nat), (64 * Nt : Nat)) (r * 2 + cc))
    (opB hA hB ((256 * Mt : Nat), (64 * Kt : Nat), (64 * Nt : Nat)) (r * 2 + cc)) Kt τ
    (i % 64) (j % 32) (i0 Nt r τ + i % 64) (j0 2 Nt cc τ + j % 32) (64 * Kt) (64 * Nt)
    (fun kk hkk c' hc' => by
      simp only [opA, hwdiv, hfA]
      rw [tA Mt Kt Nt r cc hr hcc fA hfA τ kk (i % 64) c' hτ hkk (by omega) hc', hostVal,
        toNat_cast_mul_add])
    (fun kk hkk c' hc' => by
      simp only [opB, hwmod, hfB]
      rw [tB Mt Kt Nt r cc hr hcc fB hfB τ kk c' (j % 32) hτ hkk hc' (by omega), hostVal,
        toNat_cast_mul_add])
  rw [key, ← hrow, ← hcol]

end C2

namespace C4

open Designs.WholeArrayC4 Proofs.WholeArrayC4

/-- Slot `s` of worker `w = row·4 + col`'s `j`-th A object, as a value. -/
def opA (hA hB : Nat → R) (P : Int × Int × Int) (w j s : Nat) : R :=
  match net.fifo? (aL2 (w / 4)) with
  | some f => hostVal hA hB (design.deliver P f w j s)
  | none => 0

/-- Slot `s` of worker `w = row·4 + col`'s `j`-th B object, as a value. -/
def opB (hA hB : Nat → R) (P : Int × Int × Int) (w j s : Nat) : R :=
  match net.fifo? (bL2 (w % 4)) with
  | some f => hostVal hA hB (design.deliver P f w j s)
  | none => 0

/-- The value drain `c` carries at position `p`. Each core runs `K / 64` k-steps per tile. -/
def drainVal (hA hB : Nat → R) (P : Int × Int × Int) (c p : Nat) : R :=
  match net.fifo? (cL3 c) with
  | some f => coreVal (opA hA hB P) (opB hA hB P) (P.2.1 / 64).toNat (design.stream P f p)
  | none => 0

/-- T-MM, four columns: under the contracts, for every M, K, N the sequence's asserts accept,
exactly one position on one of the four drains writes `C[i, j]`, and it writes
`Σ_k A[i, k] · B[k, j]`. -/
theorem tMM (hA hB : Nat → R) (M K N : Int) (hok : (seq M K N).ok = true) (i j : Nat)
    (hi : (i : Int) < M) (hj : (j : Int) < N) :
    (∃! cp : Nat × Nat, cp.1 < 4 ∧
      (design.host (M, K, N) (cL3 cp.1))[cp.2]? = some (2, i * N + j)) ∧
    ∀ c p : Nat, c < 4 → (design.host (M, K, N) (cL3 c))[p]? = some (2, i * N + j) →
      drainVal hA hB (M, K, N) c p =
        ∑ k ∈ Finset.range K.toNat, hA (i * K.toNat + k) * hB (k * N.toNat + j) := by
  obtain ⟨Mt, Kt, Nq, hMt, hKt, hNq, rfl, rfl, rfl⟩ := WholeArrayGuards.C4.shape M K N hok
  have hi' : i < 256 * Mt := by omega
  have hj' : j < 128 * Nq := by omega
  have hcast : ((i * (128 * Nq) + j : Nat) : Int) = (i : Int) * ((128 * Nq : Nat) : Int) + j := by
    push_cast; ring
  set cc := colC j with hcc_def
  have hcc : cc < 4 := by simp only [hcc_def, colC]; omega
  obtain ⟨fC, hfC⟩ : ∃ f, net.fifo? (cL3 cc) = some f := by
    interval_cases cc <;> exact ⟨_, rfl⟩
  obtain ⟨huniq, hsrc, hrow, hcol⟩ := tC Mt Kt Nq i j hi' hj' fC hfC
  rw [← hcc_def] at hsrc hcol
  rw [← hcast]
  refine ⟨huniq, fun c p hc hp => ?_⟩
  have hpos : (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) (cL3 cc))[
      posC Nq i j]? = some (2, ((i * (128 * Nq) + j : Nat) : Int)) := by
    simp only [Design.host, design]; push_cast
    rw [List.getElem?_toArray]; exact_mod_cast hostC_posC Mt Kt Nq i j hi' hj'
  have hcp : (c, p) = (cc, posC Nq i j) := huniq.unique ⟨hc, hp⟩ ⟨hcc, hpos⟩
  simp only [Prod.mk.injEq] at hcp
  obtain ⟨rfl, rfl⟩ := hcp
  rw [drainVal, hfC]
  simp only [Int.toNat_natCast]
  rw [show (((64 * Kt : Nat) : Int) / 64).toNat = Kt by omega, hsrc, coreVal,
    unlayoutC_layoutC _ _ (by omega)]
  set r := (i % 256) / 64 with hr_def
  set τ := tauC Nq i j with hτ_def
  have hr : r < 4 := by omega
  have hτ : τ < Mt * Nq := by
    have h1 : i / 256 < Mt := by omega
    have h2 : j / 128 < Nq := by omega
    simp only [hτ_def, tauC]; nlinarith
  have hwdiv : (r * 4 + cc) / 4 = r := by omega
  have hwmod : (r * 4 + cc) % 4 = cc := by omega
  obtain ⟨fA, hfA⟩ : ∃ f, net.fifo? (aL2 r) = some f := by
    interval_cases r <;> exact ⟨_, rfl⟩
  obtain ⟨fB, hfB⟩ : ∃ f, net.fifo? (bL2 cc) = some f := by
    interval_cases cc <;> exact ⟨_, rfl⟩
  have key := coreTile_eq hA hB
    (opA hA hB ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) (r * 4 + cc))
    (opB hA hB ((256 * Mt : Nat), (64 * Kt : Nat), (128 * Nq : Nat)) (r * 4 + cc)) Kt τ
    (i % 64) (j % 32) (i0 Nq r τ + i % 64) (j0 4 Nq cc τ + j % 32) (64 * Kt) (128 * Nq)
    (fun kk hkk c' hc' => by
      simp only [opA, hwdiv, hfA]
      rw [tA Mt Kt Nq r cc hr hcc fA hfA τ kk (i % 64) c' hτ hkk (by omega) hc', hostVal,
        toNat_cast_mul_add])
    (fun kk hkk c' hc' => by
      simp only [opB, hwmod, hfB]
      rw [tB Mt Kt Nq r cc hr hcc fB hfB τ kk c' (j % 32) hτ hkk hc' (by omega), hostVal,
        toNat_cast_mul_add])
  rw [key, ← hrow, ← hcol]

end C4

namespace C8

open Designs.WholeArrayC8 Proofs.WholeArrayC8

/-- Slot `s` of worker `w = row·8 + col`'s `j`-th A object, as a value. -/
def opA (hA hB : Nat → R) (P : Int × Int × Int) (w j s : Nat) : R :=
  match net.fifo? (aL2 (w / 8)) with
  | some f => hostVal hA hB (design.deliver P f w j s)
  | none => 0

/-- Slot `s` of worker `w = row·8 + col`'s `j`-th B object, as a value. -/
def opB (hA hB : Nat → R) (P : Int × Int × Int) (w j s : Nat) : R :=
  match net.fifo? (bL2 (w % 8)) with
  | some f => hostVal hA hB (design.deliver P f w j s)
  | none => 0

/-- The value drain `c` carries at position `p`. Each core runs `K / 64` k-steps per tile. -/
def drainVal (hA hB : Nat → R) (P : Int × Int × Int) (c p : Nat) : R :=
  match net.fifo? (cL3 c) with
  | some f => coreVal (opA hA hB P) (opB hA hB P) (P.2.1 / 64).toNat (design.stream P f p)
  | none => 0

/-- T-MM, 8 columns: under the contracts, for every M, K, N the sequence's asserts accept,
exactly one position on one of the 8 drains writes `C[i, j]`, and it writes
`Σ_k A[i, k] · B[k, j]`. -/
theorem tMM (hA hB : Nat → R) (M K N : Int) (hok : (seq M K N).ok = true) (i j : Nat)
    (hi : (i : Int) < M) (hj : (j : Int) < N) :
    (∃! cp : Nat × Nat, cp.1 < 8 ∧
      (design.host (M, K, N) (cL3 cp.1))[cp.2]? = some (2, i * N + j)) ∧
    ∀ c p : Nat, c < 8 → (design.host (M, K, N) (cL3 c))[p]? = some (2, i * N + j) →
      drainVal hA hB (M, K, N) c p =
        ∑ k ∈ Finset.range K.toNat, hA (i * K.toNat + k) * hB (k * N.toNat + j) := by
  obtain ⟨Mt, Kt, Nt, hMt, hKt, hNt, rfl, rfl, rfl⟩ := WholeArrayGuards.C8.shape M K N hok
  have hi' : i < 256 * Mt := by omega
  have hj' : j < 256 * Nt := by omega
  have hcast : ((i * (256 * Nt) + j : Nat) : Int) = (i : Int) * ((256 * Nt : Nat) : Int) + j := by
    push_cast; ring
  set cc := colC j with hcc_def
  have hcc : cc < 8 := by simp only [hcc_def, colC]; omega
  obtain ⟨fC, hfC⟩ : ∃ f, net.fifo? (cL3 cc) = some f := by
    interval_cases cc <;> exact ⟨_, rfl⟩
  obtain ⟨huniq, hsrc, hrow, hcol⟩ := tC Mt Kt Nt i j hi' hj' fC hfC
  rw [← hcc_def] at hsrc hcol
  rw [← hcast]
  refine ⟨huniq, fun c p hc hp => ?_⟩
  have hpos : (design.host ((256 * Mt : Nat), (64 * Kt : Nat), (256 * Nt : Nat)) (cL3 cc))[
      posC Nt i j]? = some (2, ((i * (256 * Nt) + j : Nat) : Int)) := by
    simp only [Design.host, design]; push_cast
    rw [List.getElem?_toArray]; exact_mod_cast hostC_posC Mt Kt Nt i j hi' hj'
  have hcp : (c, p) = (cc, posC Nt i j) := huniq.unique ⟨hc, hp⟩ ⟨hcc, hpos⟩
  simp only [Prod.mk.injEq] at hcp
  obtain ⟨rfl, rfl⟩ := hcp
  rw [drainVal, hfC]
  simp only [Int.toNat_natCast]
  rw [show (((64 * Kt : Nat) : Int) / 64).toNat = Kt by omega, hsrc, coreVal,
    unlayoutC_layoutC _ _ (by omega)]
  set r := (i % 256) / 64 with hr_def
  set τ := tauC Nt i j with hτ_def
  have hr : r < 4 := by omega
  have hτ : τ < Mt * Nt := by
    have h1 : i / 256 < Mt := by omega
    have h2 : j / 256 < Nt := by omega
    simp only [hτ_def, tauC]; nlinarith
  have hwdiv : (r * 8 + cc) / 8 = r := by omega
  have hwmod : (r * 8 + cc) % 8 = cc := by omega
  obtain ⟨fA, hfA⟩ : ∃ f, net.fifo? (aL2 r) = some f := by
    interval_cases r <;> exact ⟨_, rfl⟩
  obtain ⟨fB, hfB⟩ : ∃ f, net.fifo? (bL2 cc) = some f := by
    interval_cases cc <;> exact ⟨_, rfl⟩
  have key := coreTile_eq hA hB
    (opA hA hB ((256 * Mt : Nat), (64 * Kt : Nat), (256 * Nt : Nat)) (r * 8 + cc))
    (opB hA hB ((256 * Mt : Nat), (64 * Kt : Nat), (256 * Nt : Nat)) (r * 8 + cc)) Kt τ
    (i % 64) (j % 32) (i0 Nt r τ + i % 64) (j0 8 Nt cc τ + j % 32) (64 * Kt) (256 * Nt)
    (fun kk hkk c' hc' => by
      simp only [opA, hwdiv, hfA]
      rw [tA Mt Kt Nt r cc hr hcc fA hfA τ kk (i % 64) c' hτ hkk (by omega) hc', hostVal,
        toNat_cast_mul_add])
    (fun kk hkk c' hc' => by
      simp only [opB, hwmod, hfB]
      rw [tB Mt Kt Nt r cc hr hcc fB hfB τ kk c' (j % 32) hτ hkk hc' (by omega), hostVal,
        toNat_cast_mul_add])
  rw [key, ← hrow, ← hcol]

end C8

end IronVerify.Proofs.WholeArrayMatmul
