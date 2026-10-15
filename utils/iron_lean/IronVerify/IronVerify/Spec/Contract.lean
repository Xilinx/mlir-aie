-- Copyright (C) 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Mathlib
import IronVerify.Net
import IronVerify.Spec.WholeArray

/-!
What whole_array's cores compute, stated rather than derived: the matmul kernel's contract and the
order the core program applies it in. These mirror `core_tiles` in `utils/iron_lean/matmul_check.py`
and are part of the trusted base. Values live in any commutative semiring `R`: `ℤ`, or
`ZMod (2^16)` for i16 wraparound.
-/

namespace IronVerify.Spec.Contract

open Spec.WholeArray

variable {R : Type*} [CommSemiring R]

/-- Kernel contract: one `matmul` call adds `A · B` to the 64 × 32 C tile, reading the A and B
buffers in the kernel's blocked layouts. `C a b` is row `a`, column `b` of the tile. -/
def matmul (A B : Nat → R) (C : Nat → Nat → R) (a b : Nat) : R :=
  C a b + ∑ c ∈ Finset.range 64, A (layoutA a c) * B (layoutB c b)

/-- Core contract: output tile `τ` starts from `zero` and applies `matmul` to the core's A and B
objects `τ·Kt + kk`, for `kk = 0 … Kt − 1` in order. -/
def coreTile (A B : Nat → Nat → R) (Kt τ : Nat) : Nat → Nat → R :=
  (List.range Kt).foldl (fun C kk => matmul (A (τ * Kt + kk)) (B (τ * Kt + kk)) C) fun _ _ => 0

/-- The row and column the C buffer holds at slot `s` (the inverse of `layoutC`). -/
def unlayoutC (s : Nat) : Nat × Nat := ((s / 128) * 4 + (s % 32) / 8, ((s % 128) / 32) * 8 + s % 8)

/-- The value of an element that came from the host: entry `idx` of buffer 0 is `hA idx`, of
buffer 1 `hB idx`. -/
def hostVal (hA hB : Nat → R) : Option Src → R
  | some (.host 0 idx) => hA idx.toNat
  | some (.host 1 idx) => hB idx.toNat
  | _ => 0

/-- The value of an element a core produced: slot `s` of worker `w`'s output tile `τ`, given each
worker's A and B operands (`opA w j s` is slot `s` of its `j`-th A object). -/
def coreVal (opA opB : Nat → Nat → Nat → R) (Kt : Nat) : Option Src → R
  | some (.core w _ τ s) => coreTile (opA w) (opB w) Kt τ (unlayoutC s).1 (unlayoutC s).2
  | _ => 0

theorem unlayoutC_layoutC (a b : Nat) (hb : b < 32) :
    unlayoutC (layoutC a b) = (a, b) := by
  simp only [unlayoutC, layoutC, Prod.mk.injEq]
  omega

end IronVerify.Spec.Contract
