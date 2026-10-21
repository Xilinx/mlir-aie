-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import IronVerify.Kernel
import IronVerify.Schema.Spec

/-!
How each value domain's stored outputs relate to the exact kernel `p.matmul`, per call and
over a core's whole accumulation (`coreTile`):

- `wrap w`: the exact result read as a `w`-bit two's-complement integer is the exact result
  in `ZMod (2 ^ w)`, and computing in `ZMod (2 ^ w)` throughout gives the same value
  (`matmul_map`, `coreTile_map`).
- `saturate lo hi`: as long as every partial result of an element stays in range, the saturating
  accumulation equals the exact one (`coreTile_sat`).
- `float`: no arithmetic claim; `orderedAcc` fixes the operands and their order.
-/

namespace IronVerify

section Domain

/-- On a core where every kernel has a contract and none leaves a mode set, a kernel that
sets no mode of its own narrows in the boot mode: it wraps. -/
theorem KernelD.narrowingOn_boot (kd : KernelD) (onCore : List KernelD)
    (hs : kd.saturation = none) (hd : kd.declared = true)
    (hcore : ∀ k ∈ onCore, k.declared = true ∧ k.leavesSaturation = none) :
    kd.narrowingOn onCore = some .wrap := by
  have hall : onCore.all (fun k => k.declared && k.leavesSaturation.isNone) = true := by
    simp only [List.all_eq_true, Bool.and_eq_true, Option.isNone_iff_eq_none]
    exact fun k hk => hcore k hk
  simp [KernelD.narrowingOn, hs, hd, hall]

theorem KernelD.domainOn_int (kd : KernelD) (onCore : List KernelD) (o : KernelArg) (w : Nat)
    (signed : Bool) (mode : Narrowing) (ho : kd.output? = some o)
    (hw : intWidth? o.dtype = some (w, signed)) (hn : kd.narrowingOn onCore = some mode) :
    kd.domainOn onCore = some (ValueDomain.ofInt w signed mode) := by
  simp [KernelD.domainOn, ho, hw, hn]

theorem KernelD.domainOn_float (kd : KernelD) (onCore : List KernelD) (o : KernelArg)
    (ho : kd.output? = some o) (hw : intWidth? o.dtype = none)
    (hf : isFloatDtype o.dtype = true) :
    kd.domainOn onCore = some (.float o.dtype) := by
  simp [KernelD.domainOn, ho, hw, hf]

/-- A `w`-bit two's-complement read of `v` is `v` in `ZMod (2 ^ w)`. -/
theorem ValueDomain.read_wrap (w : Nat) (v : Int) :
    ((ValueDomain.read (.wrap w) v : Int) : ZMod (2 ^ w)) = (v : ZMod (2 ^ w)) := by
  simp only [ValueDomain.read]
  rw [ZMod.intCast_eq_intCast_iff_dvd_sub]
  have hb : Int.bmod v (2 ^ w) % (2 ^ w : Int) = v % (2 ^ w : Int) := by
    simpa using Int.bmod_emod (x := v) (m := 2 ^ w)
  exact Int.ModEq.dvd (by simpa [Int.ModEq] using hb)

/-- Inside `[lo, hi]` a saturating read is the identity. -/
theorem ValueDomain.read_sat_of_mem (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) :
    ValueDomain.read (.saturate lo hi) v = v := by
  simp [ValueDomain.read, min_eq_right h2, max_eq_right h1]

end Domain

end IronVerify

namespace IronVerify.Schema.WholeArray

open IronVerify

namespace Params

variable (p : Params)

/-- One exact call commutes with any ring hom (e.g. `ℤ → ZMod (2 ^ w)`). -/
theorem matmul_map {R S : Type*} [CommSemiring R] [CommSemiring S] (f : R →+* S)
    (A B : Nat → R) (C : Nat → Nat → R) (a b : Nat) :
    p.matmul (fun q => f (A q)) (fun q => f (B q)) (fun x y => f (C x y)) a b =
      f (p.matmul A B C a b) := by
  simp [matmul, map_add, map_sum, map_mul]

/-- A core's whole accumulation commutes with any ring hom. -/
theorem coreTile_map {R S : Type*} [CommSemiring R] [CommSemiring S] (f : R →+* S)
    (zero : Nat → Nat → R) (opA opB : Nat → Nat → R) (Kt τ : Nat) (a b : Nat) :
    coreTile p.matmul (fun x y => f (zero x y)) (fun j q => f (opA j q)) (fun j q => f (opB j q))
        Kt τ a b =
      f (coreTile p.matmul zero opA opB Kt τ a b) := by
  unfold coreTile
  suffices h : ∀ (l : List Nat) (C : Nat → Nat → R),
      l.foldl (fun C kk => p.matmul (fun q => f (opA (τ * Kt + kk) q))
          (fun q => f (opB (τ * Kt + kk) q)) C) (fun x y => f (C x y)) =
        fun x y => f (l.foldl (fun C kk => p.matmul (opA (τ * Kt + kk)) (opB (τ * Kt + kk)) C)
          C x y) by
    exact congrFun (congrFun (h _ zero) a) b
  intro l
  induction l with
  | nil => intro C; rfl
  | cons kk l ih =>
    intro C
    simp only [List.foldl_cons]
    rw [← ih]
    congr 1
    funext x y
    exact matmul_map p f _ _ _ x y

/-- A saturating call whose exact result is in range is the exact call. -/
theorem matmulSat_of_mem (lo hi : ℤ) (A B : Nat → ℤ) (C : Nat → Nat → ℤ) (a b : Nat)
    (h1 : lo ≤ p.matmul A B C a b) (h2 : p.matmul A B C a b ≤ hi) :
    p.matmulSat lo hi A B C a b = p.matmul A B C a b := by
  simp [matmulSat, min_eq_right h2, max_eq_right h1]

/-- One exact call reads its C tile only at the element it writes. -/
theorem matmul_congr_C {R : Type*} [CommSemiring R] (A B : Nat → R) (C C' : Nat → Nat → R)
    (a b : Nat) (h : C a b = C' a b) : p.matmul A B C a b = p.matmul A B C' a b := by
  simp [matmul, h]

/-- The exact accumulation after the first `n` k-steps, at element `(a, b)`. -/
def partialTile (zero : Nat → Nat → ℤ) (opA opB : Nat → Nat → ℤ) (Kt τ n : Nat) :
    Nat → Nat → ℤ :=
  (List.range n).foldl (fun C kk => p.matmul (opA (τ * Kt + kk)) (opB (τ * Kt + kk)) C) zero

/-- Saturating stores give the exact accumulation at `(a, b)` as long as every partial result
there, after `1 … Kt` k-steps, stays in `[lo, hi]`. -/
theorem coreTile_sat (lo hi : ℤ) (zero : Nat → Nat → ℤ) (opA opB : Nat → Nat → ℤ)
    (Kt τ a b : Nat)
    (hpart : ∀ n, 0 < n → n ≤ Kt →
      lo ≤ p.partialTile zero opA opB Kt τ n a b ∧ p.partialTile zero opA opB Kt τ n a b ≤ hi) :
    coreTile (p.matmulSat lo hi) zero opA opB Kt τ a b =
      coreTile p.matmul zero opA opB Kt τ a b := by
  unfold coreTile
  suffices h : ∀ n, n ≤ Kt →
      (List.range n).foldl (fun C kk => p.matmulSat lo hi (opA (τ * Kt + kk)) (opB (τ * Kt + kk)) C)
          zero a b =
        p.partialTile zero opA opB Kt τ n a b from h Kt le_rfl
  intro n hn
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [partialTile, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    have hn' := hpart (n + 1) (Nat.succ_pos n) hn
    simp only [partialTile, List.range_succ, List.foldl_append, List.foldl_cons,
      List.foldl_nil] at hn'
    rw [matmulSat_of_mem p lo hi _ _ _ a b ?_ ?_]
    · exact matmul_congr_C p _ _ _ _ a b (ih (Nat.le_of_succ_le hn))
    · rw [matmul_congr_C p _ _ _ _ a b (ih (Nat.le_of_succ_le hn))]; exact hn'.1
    · rw [matmul_congr_C p _ _ _ _ a b (ih (Nat.le_of_succ_le hn))]; exact hn'.2

end Params

end IronVerify.Schema.WholeArray
